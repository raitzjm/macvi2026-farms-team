#!/usr/bin/env python3
"""
Generate pseudo-labels from a Co-DETR checkpoint on the training images.

Runs inference on all training-split images, filters predictions against
existing GT annotations (conf >= threshold, IoU < iou_thr vs GT, cap per image),
and writes a COCO-format pseudo-label JSON for merging with GT on retraining.

Usage:
  python make_pseudo_labels.py \
    --checkpoint artifacts/checkpoints/T-004-codetr-tensordock/best.pth \
    --output /tmp/pseudo_labels_T-004.json \
    --conf 0.5 --iou-thr 0.3 --cap 3
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHALLENGE_ROOT = HERE.parent
MMDET_SRC = CHALLENGE_ROOT / "envs" / "mmdet-src"
CODETR_PROJ = MMDET_SRC / "projects" / "CO-DETR"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODETR_PROJ))
sys.path.insert(0, str(MMDET_SRC))

from _paths import TRAIN_IMGS, TRAIN_JSON  # noqa: E402
from split_manifest import default_split_manifest_path, load_manifest  # noqa: E402


def iou(box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box [x1,y1,x2,y2] and N GT boxes."""
    if gt_boxes.shape[0] == 0:
        return np.zeros(0)
    ix1 = np.maximum(box[0], gt_boxes[:, 0])
    iy1 = np.maximum(box[1], gt_boxes[:, 1])
    ix2 = np.minimum(box[2], gt_boxes[:, 2])
    iy2 = np.minimum(box[3], gt_boxes[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_box + area_gt - inter
    return np.where(union > 0, inter / union, 0.0)


def resolve_config(checkpoint: Path) -> Path:
    sibling = checkpoint.parent / "config_used.py"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        f"No config_used.py found beside {checkpoint}. "
        "Copy the resolved config there or pass --config.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--conf", type=float, default=0.5,
                    help="Minimum confidence for pseudo-label acceptance")
    ap.add_argument("--iou-thr", type=float, default=0.3,
                    help="Max IoU vs GT for a box to be accepted (lower = stricter)")
    ap.add_argument("--cap", type=int, default=3,
                    help="Max pseudo-label boxes per image")
    ap.add_argument("--device", default="0")
    ap.add_argument("--split-manifest", default=str(default_split_manifest_path()))
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else resolve_config(checkpoint)

    print(f"[pl] checkpoint: {checkpoint}")
    print(f"[pl] config:     {config_path}")
    print(f"[pl] conf>={args.conf}  IoU<{args.iou_thr}  cap={args.cap}")

    # Register CO-DETR modules
    import codetr  # noqa: F401
    sys.modules.setdefault("projects", types.ModuleType("projects"))
    sys.modules.setdefault("projects.CO-DETR", types.ModuleType("projects.CO-DETR"))
    sys.modules["projects.CO-DETR.codetr"] = codetr

    import torch
    # PyTorch 2.6+ defaults weights_only=True; mmengine checkpoints need False
    _real_load = torch.load
    torch.load = lambda *a, **kw: _real_load(*a, **{**kw, "weights_only": False})

    from mmdet.apis import inference_detector, init_detector

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    print(f"[pl] loading model on {device}")
    model = init_detector(str(config_path), str(checkpoint), device=device)
    model.eval()

    # Load COCO data
    coco = json.loads(TRAIN_JSON.read_text())
    img_by_id = {im["id"]: im for im in coco["images"]}
    cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])
    idx_to_cat_id = {i: c["id"] for i, c in enumerate(cats_sorted)}

    # GT boxes per image: {img_id: np.array([[x1,y1,x2,y2], ...])}
    gt_by_img: dict[int, np.ndarray] = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        box = np.array([x, y, x + w, y + h])
        if a["image_id"] not in gt_by_img:
            gt_by_img[a["image_id"]] = []
        gt_by_img[a["image_id"]].append(box)
    for k in gt_by_img:
        gt_by_img[k] = np.array(gt_by_img[k])

    # Load shared split manifest — only infer on TRAINING images
    manifest = load_manifest(Path(args.split_manifest))
    train_ids = set(manifest["train_ids"])
    train_imgs = [im for im in coco["images"] if im["id"] in train_ids]
    print(f"[pl] inferring on {len(train_imgs)} training images")

    pseudo_anns = []
    ann_id = 1
    n_accepted = 0
    n_rejected_conf = 0
    n_rejected_iou = 0
    n_rejected_cap = 0

    for i, im in enumerate(train_imgs):
        img_path = TRAIN_IMGS / im["file_name"]
        if not img_path.exists():
            continue

        result = inference_detector(model, str(img_path))
        pred = result.pred_instances

        bboxes = pred.bboxes.cpu().numpy()   # [N, 4] xyxy
        scores = pred.scores.cpu().numpy()   # [N]
        labels = pred.labels.cpu().numpy()   # [N]

        gt_boxes = gt_by_img.get(im["id"], np.zeros((0, 4)))
        accepted = []

        # Sort by confidence descending
        order = np.argsort(-scores)
        for idx in order:
            s = float(scores[idx])
            if s < args.conf:
                n_rejected_conf += 1
                continue
            box_xyxy = bboxes[idx]  # [x1, y1, x2, y2]
            # Check IoU against all GT boxes
            if gt_boxes.shape[0] > 0:
                max_iou = float(iou(box_xyxy, gt_boxes).max())
            else:
                max_iou = 0.0
            if max_iou >= args.iou_thr:
                n_rejected_iou += 1
                continue
            if len(accepted) >= args.cap:
                n_rejected_cap += 1
                continue
            x1, y1, x2, y2 = box_xyxy
            bw, bh = float(x2 - x1), float(y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            accepted.append({
                "id": ann_id,
                "image_id": im["id"],
                "category_id": int(idx_to_cat_id[int(labels[idx])]),
                "bbox": [float(x1), float(y1), bw, bh],
                "area": bw * bh,
                "score": s,
                "iscrowd": 0,
                "is_pseudo": True,
            })
            ann_id += 1

        pseudo_anns.extend(accepted)
        n_accepted += len(accepted)

        if (i + 1) % 200 == 0 or (i + 1) == len(train_imgs):
            print(f"[pl] {i+1}/{len(train_imgs)}  accepted={n_accepted}  "
                  f"rej_conf={n_rejected_conf} rej_iou={n_rejected_iou} rej_cap={n_rejected_cap}")

    print(f"\n[pl] DONE: {n_accepted} pseudo-labels from {len(train_imgs)} images")
    print(f"[pl]   avg per image: {n_accepted/len(train_imgs):.2f}")

    # Quality check
    from collections import Counter
    cat_counts = Counter(a["category_id"] for a in pseudo_anns)
    total = len(pseudo_anns)
    print("[pl] top-5 categories by pseudo-label count:")
    for cat_id, cnt in cat_counts.most_common(5):
        cat_name = next((c["name"] for c in coco["categories"] if c["id"] == cat_id), str(cat_id))
        print(f"  {cat_name}: {cnt} ({100*cnt/total:.1f}%)")

    # Write COCO-format pseudo-label JSON
    out = {
        "images": [img_by_id[a["image_id"]] for a in {a["image_id"]: a for a in pseudo_anns}.values()],
        "annotations": pseudo_anns,
        "categories": coco["categories"],
        "info": {"pseudo_label_source": str(checkpoint), "conf_thr": args.conf,
                 "iou_thr": args.iou_thr, "cap": args.cap},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out))
    print(f"[pl] wrote {len(pseudo_anns)} pseudo-annotations → {args.output}")


if __name__ == "__main__":
    main()

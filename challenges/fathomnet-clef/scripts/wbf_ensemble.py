"""
Weighted Box Fusion ensemble for FathomNet-CLEF 2026.

Usage:
    python wbf_ensemble.py \
        --inputs <csv1> <csv2> \
        --weights 2.0 1.0 \
        --iou-thr 0.55 \
        --skip-box-thr 0.001 \
        --output <output.csv> \
        --test-json /home/farms/data/datasets/fathomnet-2026/test_dataset.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ensemble_boxes import weighted_boxes_fusion


def load_submissions(paths: list[str]) -> dict[int, dict[int, list]]:
    """Load CSVs → {img_id: {cat_id: [[x1,y1,x2,y2,score], ...]}}"""
    all_preds = []
    for path in paths:
        preds: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = int(row["image_id"])
                cat_id = int(row["category_id"])
                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_width"])
                h = float(row["bbox_height"])
                score = float(row["score"])
                preds[img_id][cat_id].append([x, y, x + w, y + h, score])
        all_preds.append(preds)
    return all_preds


def run_wbf(
    all_preds: list,
    img_dims: dict[int, tuple[int, int]],
    img_ids: list[int],
    cat_ids: list[int],
    weights: list[float],
    iou_thr: float,
    skip_box_thr: float,
) -> list[dict]:
    rows = []
    ann_id = 1
    for img_id in img_ids:
        W, H = img_dims[img_id]
        for cat_id in cat_ids:
            boxes_list, scores_list, labels_list = [], [], []
            for model_preds in all_preds:
                raw = model_preds.get(img_id, {}).get(cat_id, [])
                if raw:
                    boxes = [[max(0, b[0] / W), max(0, b[1] / H),
                              min(1, b[2] / W), min(1, b[3] / H)] for b in raw]
                    scores = [b[4] for b in raw]
                    labels = [0] * len(raw)
                else:
                    boxes, scores, labels = [], [], []
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)

            if all(len(b) == 0 for b in boxes_list):
                continue

            fused_boxes, fused_scores, _ = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
            for box, score in zip(fused_boxes, fused_scores):
                x1, y1, x2, y2 = box
                bx = x1 * W
                by = y1 * H
                bw = (x2 - x1) * W
                bh = (y2 - y1) * H
                rows.append({
                    "annotation_id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox_x": bx,
                    "bbox_y": by,
                    "bbox_width": bw,
                    "bbox_height": bh,
                    "score": score,
                })
                ann_id += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--iou-thr", type=float, default=0.55)
    ap.add_argument("--skip-box-thr", type=float, default=0.001)
    ap.add_argument("--output", required=True)
    ap.add_argument("--test-json", required=True)
    args = ap.parse_args()

    print(f"Loading test metadata from {args.test_json}")
    with open(args.test_json) as f:
        meta = json.load(f)
    img_dims = {img["id"]: (img["width"], img["height"]) for img in meta["images"]}
    img_ids = sorted(img_dims.keys())
    cat_ids = sorted(c["id"] for c in meta["categories"]) if "categories" in meta else list(range(1, 33))

    print(f"Loading {len(args.inputs)} submission CSVs…")
    all_preds = load_submissions(args.inputs)

    print(f"Running WBF: weights={args.weights}, iou_thr={args.iou_thr}, skip_box_thr={args.skip_box_thr}")
    rows = run_wbf(all_preds, img_dims, img_ids, cat_ids, args.weights, args.iou_thr, args.skip_box_thr)
    print(f"WBF produced {len(rows)} predictions")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "score"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

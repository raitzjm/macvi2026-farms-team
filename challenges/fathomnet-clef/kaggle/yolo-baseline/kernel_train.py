#!/usr/bin/env python3
"""
FathomNet-CLEF 2026 — YOLO11l baseline (Kaggle T4x2).

Setup required ONCE in the Kaggle UI before pushing/running:
  1. Add a Kaggle Secret named WANDB_API_KEY (Add-ons > Secrets).
  2. Set Accelerator to "GPU T4 x2" (the kaggle-cli kernel-metadata.json
     cannot pin the dual-GPU type — only enable_gpu=true. Verify in UI).
  3. Internet ON, Persistence Off, Environment "Pin to original".

Inputs (mounted automatically from kernel-metadata.json):
  /kaggle/input/fathomnet-2026/                  -> JSONs only (no images)
    train_dataset.json, test_dataset.json, download.py
  /kaggle/input/fathomnetclef2026/images/{train,test}/<uuid>.jpg
                                                 -> mirror dataset with images
                                                    (cosheimil/fathomnetclef2026)

Outputs:
  /kaggle/working/idx_to_cat_id.json   inverse class map (idx -> original COCO cat_id)
  /kaggle/working/best.pt              best YOLO weights
  /kaggle/working/submission.csv       Kaggle-format predictions on test_set
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Install deps (cite challenge requirements-ultralytics.txt; pinned subset).
# ---------------------------------------------------------------------------
PIP_PKGS = [
    "ultralytics==8.4.41",
    "sahi==0.11.36",
    "wandb==0.26.1",
    "pycocotools==2.0.11",
    "ensemble-boxes==1.0.9",
]
print("[setup] pip installing deps")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PIP_PKGS])

# ---------------------------------------------------------------------------
# 2. WandB key from Kaggle Secrets.
# ---------------------------------------------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    USE_WANDB = True
    print("[setup] WANDB_API_KEY loaded from Kaggle Secrets")
except Exception as e:
    print(f"[setup] WandB disabled (no secret): {e}")
    USE_WANDB = False

# ---------------------------------------------------------------------------
# 3. Paths.
# ---------------------------------------------------------------------------
COMP_ROOT = Path("/kaggle/input/fathomnet-2026")
IMG_ROOT = Path("/kaggle/input/fathomnetclef2026/images")  # train/, test/
TRAIN_JSON = COMP_ROOT / "train_dataset.json"
TEST_JSON = COMP_ROOT / "test_dataset.json"
WORK = Path("/kaggle/working")
YOLO_DATA = WORK / "yolo_data"

assert TRAIN_JSON.exists(), f"Missing {TRAIN_JSON}"
assert TEST_JSON.exists(), f"Missing {TEST_JSON}"
assert IMG_ROOT.exists(), f"Missing image mirror dataset at {IMG_ROOT}"

# Resolve image filename: JSON references *.png; cosheimil mirror is *.jpg.
# Build a stem -> path index once for fast lookup.
def _build_stem_index(folder: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in folder.iterdir():
        idx[p.stem] = p
    return idx

print("[data] indexing image mirrors")
TRAIN_IMG_INDEX = _build_stem_index(IMG_ROOT / "train")
TEST_IMG_INDEX = _build_stem_index(IMG_ROOT / "test")
print(f"[data] {len(TRAIN_IMG_INDEX)} train imgs / {len(TEST_IMG_INDEX)} test imgs in mirror")


# ---------------------------------------------------------------------------
# 4. COCO -> YOLO conversion (adapted from scripts/train_yolo.py).
# ---------------------------------------------------------------------------
def coco_to_yolo(coco_json: Path, dst_root: Path,
                 stem_index: dict[str, Path],
                 val_split: float = 0.1, seed: int = 42):
    coco = json.loads(coco_json.read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    idx_to_cat_id = {i: c["id"] for i, c in enumerate(cats)}
    class_names = [c["name"] for c in cats]

    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    img_ids = sorted(img_by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(img_ids)
    cut = int(len(img_ids) * (1 - val_split))
    splits = {"train": img_ids[:cut], "val": img_ids[cut:]}

    for split, ids in splits.items():
        labels_dir = dst_root / "labels" / split
        images_dir = dst_root / "images" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        missing = 0
        for iid in ids:
            im = img_by_id[iid]
            stem = Path(im["file_name"]).stem
            src = stem_index.get(stem)
            if src is None:
                missing += 1
                continue
            dst = images_dir / src.name
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
            lines = []
            for a in anns_by_img.get(iid, []):
                x, y, w, h = a["bbox"]
                cx = (x + w / 2) / im["width"]
                cy = (y + h / 2) / im["height"]
                wn = w / im["width"]
                hn = h / im["height"]
                cls = cat_id_to_idx[a["category_id"]]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
        if missing:
            print(f"[data] WARN: {missing} images missing from mirror in split={split}")

    return class_names, splits, idx_to_cat_id


print("[data] building YOLO dataset")
class_names, splits, idx_to_cat_id = coco_to_yolo(TRAIN_JSON, YOLO_DATA, TRAIN_IMG_INDEX)
print(f"[data] {len(splits['train'])} train / {len(splits['val'])} val / {len(class_names)} classes")

(WORK / "idx_to_cat_id.json").write_text(
    json.dumps({str(k): v for k, v in idx_to_cat_id.items()})
)

yaml_path = YOLO_DATA / "fathomnet.yaml"
yaml_path.write_text(
    f"path: {YOLO_DATA}\n"
    f"train: images/train\n"
    f"val: images/val\n"
    f"nc: {len(class_names)}\n"
    f"names: {class_names}\n"
)

# ---------------------------------------------------------------------------
# 5. WandB pre-init (so ultralytics callback uses our run name).
# ---------------------------------------------------------------------------
RUN_NAME = "fathomnet-yolo11l-baseline"
if USE_WANDB:
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"wandb": True})
    except Exception as e:
        print(f"[wandb] settings update failed: {e}")
    import wandb
    wandb.init(project="fathomnet2026", entity="little_farms",
               name=RUN_NAME, dir=str(WORK))

# ---------------------------------------------------------------------------
# 6. Train.
# ---------------------------------------------------------------------------
import torch
from ultralytics import YOLO

device = [0, 1] if torch.cuda.device_count() >= 2 else 0
print(f"[train] device={device} cuda_count={torch.cuda.device_count()}")

model = YOLO("yolo11l.pt")  # auto-downloaded by ultralytics
model.train(
    data=str(yaml_path),
    epochs=50,
    patience=20,
    imgsz=1280,
    rect=True,
    batch=8,                  # per-GPU batch on T4x2 -> effective 16
    workers=2,
    optimizer="AdamW",
    lr0=0.001,
    project=str(WORK),
    name="run",
    exist_ok=True,
    device=device,
    amp=True,
)

best_pt = WORK / "run" / "weights" / "best.pt"
final_pt = WORK / "best.pt"
if best_pt.exists():
    import shutil
    shutil.copy2(best_pt, final_pt)
    print(f"[train] copied best weights -> {final_pt}")
else:
    print(f"[train] WARN: best.pt not found at {best_pt}")

# ---------------------------------------------------------------------------
# 7. Inference on test_set -> submission.csv.
# ---------------------------------------------------------------------------
print("[infer] running test-set inference")
test_coco = json.loads(TEST_JSON.read_text())
test_images = test_coco["images"]
stem_to_image_id = {Path(im["file_name"]).stem: im["id"] for im in test_images}

# Resolve test image paths via mirror index.
test_paths: list[Path] = []
test_ids: list[int] = []
missing = 0
for im in test_images:
    stem = Path(im["file_name"]).stem
    p = TEST_IMG_INDEX.get(stem)
    if p is None:
        missing += 1
        continue
    test_paths.append(p)
    test_ids.append(im["id"])
print(f"[infer] {len(test_paths)} test imgs found / {missing} missing")

infer_model = YOLO(str(final_pt) if final_pt.exists() else str(best_pt))
rows: list[dict] = []
ann_id = 1
BATCH = 16
for i in range(0, len(test_paths), BATCH):
    chunk = test_paths[i : i + BATCH]
    chunk_ids = test_ids[i : i + BATCH]
    results = infer_model.predict(
        source=[str(p) for p in chunk],
        imgsz=1280,
        conf=0.001,
        iou=0.6,
        device=0,
        verbose=False,
        augment=False,
    )
    for img_id, r in zip(chunk_ids, results):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
            rows.append({
                "annotation_id": ann_id,
                "image_id": int(img_id),
                "category_id": int(idx_to_cat_id[int(c)]),
                "bbox_x": float(x1),
                "bbox_y": float(y1),
                "bbox_width": float(x2 - x1),
                "bbox_height": float(y2 - y1),
                "score": float(s),
            })
            ann_id += 1

import csv
SUB = WORK / "submission.csv"
with SUB.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_width", "bbox_height", "score",
    ])
    w.writeheader()
    w.writerows(rows)
print(f"[infer] wrote {len(rows)} rows")
print(f"Submission CSV: /kaggle/working/submission.csv")

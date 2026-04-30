#!/usr/bin/env python3
"""
FathomNet-CLEF 2026 — YOLO26l baseline (Kaggle T4x2).

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
  /kaggle/working/best.pt              best YOLO26l weights
  /kaggle/working/submission.csv       Kaggle-format predictions on test_set
"""
from __future__ import annotations

import atexit
import json
import os
import random
import subprocess
import sys
from pathlib import Path
import shutil

# ---------------------------------------------------------------------------
# 0. Pin torch + torchvision before ultralytics is imported.
#    torch 2.4.1+cu121 covers SM 7.5 (T4) and SM 8.0 (A100). cu121 wheels
#    run fine on any CUDA 12.x driver. Always re-exec so new wheels are active.
# ---------------------------------------------------------------------------
if not os.environ.get("_TORCH_COMPAT"):
    print("[setup] Pinning torch 2.4.1+cu121 + torchvision 0.19.1 (T4/A100)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "torch==2.4.1", "torchvision==0.19.1",
                           "--index-url", "https://download.pytorch.org/whl/cu121"])
    os.environ["_TORCH_COMPAT"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ---------------------------------------------------------------------------
# 1. Install deps. ultralytics>=8.4.35 required for YOLO26 support.
# ---------------------------------------------------------------------------
print("[setup] pip installing deps")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "ultralytics==8.4.42", "--no-deps"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "sahi==0.11.36", "wandb==0.26.1",
                       "pycocotools==2.0.11", "ensemble-boxes==1.0.9"])

# ---------------------------------------------------------------------------
# 2. WandB key from Kaggle Secrets.
# ---------------------------------------------------------------------------
WORK = Path("/kaggle/working")


class _TeeWriter:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


_raw_stdout = sys.stdout
_raw_stderr = sys.stderr
_log_file = (WORK / "train.log").open("a", encoding="utf-8", buffering=1)
sys.stdout = _TeeWriter(_raw_stdout, _log_file)
sys.stderr = _TeeWriter(_raw_stderr, _log_file)
atexit.register(_log_file.close)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    USE_WANDB = True
    print("[setup] WANDB_API_KEY loaded from Kaggle Secrets")
except Exception:
    # removed - use Kaggle Secret WANDB_API_KEY instead of embedding credentials
    os.environ.pop("WANDB_API_KEY", None)
    USE_WANDB = False
    print("[setup] WANDB disabled: Kaggle Secret WANDB_API_KEY not available")

# ---------------------------------------------------------------------------
# 3. Paths.
# ---------------------------------------------------------------------------
YOLO_DATA = WORK / "yolo_data"

# Locate competition JSONs.
_candidates = sorted(Path("/kaggle/input").rglob("train_dataset.json"))
assert _candidates, f"train_dataset.json not found under /kaggle/input/. Contents: {list(Path('/kaggle/input').iterdir())}"
COMP_ROOT = _candidates[0].parent
print(f"[data] competition root: {COMP_ROOT}")
TRAIN_JSON = COMP_ROOT / "train_dataset.json"
TEST_JSON = COMP_ROOT / "test_dataset.json"
assert TEST_JSON.exists(), f"Missing {TEST_JSON}"

# Locate image mirror.
_img_candidates = [
    Path("/kaggle/input/fathomnetclef2026/images"),
    Path("/kaggle/input/fathomnetclef2026"),
]
IMG_ROOT = next((p for p in _img_candidates if (p / "train").exists()), None)
if IMG_ROOT is None:
    for _p in Path("/kaggle/input").rglob("train"):
        if _p.is_dir() and any(_p.iterdir()):
            IMG_ROOT = _p.parent
            break
assert IMG_ROOT is not None, f"Image mirror not found. /kaggle/input contents: {list(Path('/kaggle/input').iterdir())}"
print(f"[data] image root: {IMG_ROOT}")

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
# 4. COCO -> YOLO conversion.
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

_write_json(
    WORK / "metadata.json",
    {
        "config_source": "kaggle/yolo26-baseline/kernel_train.py",
        "kind": "training+inference",
        "platform": "kaggle",
        "resolved_config": "resolved-config.yaml",
        "results_artifact": "results.csv",
        "selected_checkpoint": "best.pt",
        "stdout_log": "train.log",
        "submission_artifact": "submission.csv",
        "track": "YOLO26l",
    },
)

# ---------------------------------------------------------------------------
# 5. WandB setup.
# ---------------------------------------------------------------------------
RUN_NAME = "fathomnet-yolo26l-baseline"
WANDB_PROJECT = "fathomnet2026"
if USE_WANDB:
    os.environ["WANDB_ENTITY"] = "little_farms"
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS.update({"wandb": True})
    except Exception as e:
        print(f"[wandb] settings update failed: {e}")

# ---------------------------------------------------------------------------
# 6. Train.
# ---------------------------------------------------------------------------
import torch
from ultralytics import YOLO

device = [0, 1] if torch.cuda.device_count() >= 2 else 0
print(f"[train] device={device} cuda_count={torch.cuda.device_count()}")

model = YOLO("yolo26l.pt")  # auto-downloaded by ultralytics
model.train(
    data=str(yaml_path),
    epochs=50,
    patience=20,
    imgsz=640,
    rect=True,
    batch=32,                 # per-GPU batch on T4x2 -> effective 64
    workers=2,
    optimizer="AdamW",
    lr0=0.001,
    project=WANDB_PROJECT,
    name=RUN_NAME,
    exist_ok=True,
    device=device,
    amp=True,
)

# Get actual save_dir from trainer — ultralytics prefixes runs/detect/ regardless
# of project= kwarg so constructing the path manually is unreliable.
save_dir = Path(model.trainer.save_dir)
best_pt = save_dir / "weights" / "best.pt"
final_pt = WORK / "best.pt"
print(f"[train] trainer save_dir: {save_dir}")
if best_pt.exists():
    shutil.copy2(best_pt, final_pt)
    print(f"[train] copied best weights -> {final_pt}")
else:
    candidates = list(WORK.rglob("best.pt"))
    if candidates:
        best_pt = candidates[0]
        shutil.copy2(best_pt, final_pt)
        print(f"[train] copied best weights (fallback glob) from {best_pt} -> {final_pt}")
    else:
        raise FileNotFoundError(f"best.pt not found under {WORK}. Contents: {list(WORK.iterdir())}")

resolved_cfg_src = save_dir / "args.yaml"
if resolved_cfg_src.exists():
    shutil.copy2(resolved_cfg_src, WORK / "resolved-config.yaml")
results_csv_src = save_dir / "results.csv"
if results_csv_src.exists():
    shutil.copy2(results_csv_src, WORK / "results.csv")

# ---------------------------------------------------------------------------
# 7. Inference on test_set -> submission.csv.
# ---------------------------------------------------------------------------
print("[infer] running test-set inference")
test_coco = json.loads(TEST_JSON.read_text())
test_images = test_coco["images"]

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

infer_model = YOLO(str(final_pt))
rows: list[dict] = []
ann_id = 1
BATCH = 64
for i in range(0, len(test_paths), BATCH):
    chunk = test_paths[i : i + BATCH]
    chunk_ids = test_ids[i : i + BATCH]
    results = infer_model.predict(
        source=[str(p) for p in chunk],
        imgsz=640,
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

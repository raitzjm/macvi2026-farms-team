#!/usr/bin/env python3
"""
FathomNet-CLEF 2026 — T-014 YOLO26s k-fold training kernel template.

This kernel is training-only. Checkpoints and run artifacts are pulled back
locally after completion; inference and pseudo-label post-processing happen off
Kaggle for this experiment family.
"""
from __future__ import annotations

import atexit
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


if not os.environ.get("_TORCH_COMPAT"):
    print("[setup] Pinning torch 2.4.1+cu121 + torchvision 0.19.1 (T4/A100)")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch==2.4.1",
            "torchvision==0.19.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    os.environ["_TORCH_COMPAT"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

print("[setup] pip installing deps")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.42", "--no-deps"])
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "wandb==0.26.1",
        "pycocotools==2.0.11",
    ]
)

WORK = Path("/kaggle/working")
KERNEL_ROOT = Path(__file__).resolve().parent
FOLD_CONFIG = __FOLD_CONFIG_JSON__
MANIFEST = __MANIFEST_JSON__

WANDB_API_KEY = "__WANDB_KEY_PLACEHOLDER__"
if WANDB_API_KEY and WANDB_API_KEY != "__WANDB_KEY_PLACEHOLDER__":
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    USE_WANDB = True
    print("[setup] WANDB_API_KEY set via push-time injection")
else:
    os.environ.pop("WANDB_API_KEY", None)
    USE_WANDB = False
    print("[setup] WANDB disabled: no key injected at push time")


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


_raw_stdout = sys.stdout
_raw_stderr = sys.stderr
_log_file = (WORK / "train.log").open("a", encoding="utf-8", buffering=1)
sys.stdout = _TeeWriter(_raw_stdout, _log_file)
sys.stderr = _TeeWriter(_raw_stderr, _log_file)
atexit.register(_log_file.close)


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_stem_index(folder: Path) -> dict[str, Path]:
    return {path.stem: path for path in folder.iterdir()}


def validate_source(coco: dict):
    expected = MANIFEST["source_stats"]
    actual = {
        "image_count": len(coco["images"]),
        "annotation_count": len(coco["annotations"]),
        "category_count": len(coco["categories"]),
        "sha256": canonical_sha256(coco),
    }
    if actual != expected:
        raise RuntimeError(f"Manifest/source mismatch: expected={expected} actual={actual}")


def coco_to_yolo(coco: dict, dst_root: Path, stem_index: dict[str, Path], train_ids: list[int], val_ids: list[int]):
    cats = sorted(coco["categories"], key=lambda cat: cat["id"])
    cat_id_to_idx = {cat["id"]: index for index, cat in enumerate(cats)}
    idx_to_cat_id = {index: cat["id"] for index, cat in enumerate(cats)}
    class_names = [cat["name"] for cat in cats]
    img_by_id = {int(image["id"]): image for image in coco["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    split_map = {"train": train_ids, "val": val_ids}
    for split_name, image_ids in split_map.items():
        labels_dir = dst_root / "labels" / split_name
        images_dir = dst_root / "images" / split_name
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            image = img_by_id[image_id]
            stem = Path(image["file_name"]).stem
            src = stem_index[stem]
            dst = images_dir / src.name
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
            lines = []
            for ann in anns_by_img.get(image_id, []):
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / image["width"]
                cy = (y + h / 2) / image["height"]
                wn = w / image["width"]
                hn = h / image["height"]
                cls = cat_id_to_idx[int(ann["category_id"])]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
    return class_names, idx_to_cat_id


_candidates = sorted(Path("/kaggle/input").rglob("train_dataset.json"))
assert _candidates, "train_dataset.json not found under /kaggle/input"
COMP_ROOT = _candidates[0].parent
TRAIN_JSON = COMP_ROOT / "train_dataset.json"
TEST_JSON = COMP_ROOT / "test_dataset.json"
assert TEST_JSON.exists(), f"Missing {TEST_JSON}"

_img_candidates = [
    Path("/kaggle/input/fathomnetclef2026/images"),
    Path("/kaggle/input/fathomnetclef2026"),
]
IMG_ROOT = next((path for path in _img_candidates if (path / "train").exists()), None)
if IMG_ROOT is None:
    raise RuntimeError("Image mirror not found under /kaggle/input")

print(f"[cfg] run_id={FOLD_CONFIG['run_id']} fold={FOLD_CONFIG['fold_name']}")
print(f"[cfg] manifest={FOLD_CONFIG['manifest_file']}")
print(f"[cfg] machine_shape={FOLD_CONFIG['machine_shape']}")

coco = json.loads(TRAIN_JSON.read_text())
validate_source(coco)

fold = next(fold for fold in MANIFEST["folds"] if int(fold["fold_index"]) == int(FOLD_CONFIG["fold_index"]))
train_ids = [int(image_id) for image_id in fold["train_ids"]]
val_ids = [int(image_id) for image_id in fold["val_ids"]]
print(f"[data] train={len(train_ids)} val={len(val_ids)}")

YOLO_DATA = WORK / "yolo_data"
TRAIN_IMG_INDEX = build_stem_index(IMG_ROOT / "train")
class_names, idx_to_cat_id = coco_to_yolo(coco, YOLO_DATA, TRAIN_IMG_INDEX, train_ids, val_ids)
(WORK / "idx_to_cat_id.json").write_text(json.dumps({str(k): v for k, v in idx_to_cat_id.items()}))

yaml_path = YOLO_DATA / "fathomnet.yaml"
yaml_path.write_text(
    f"path: {YOLO_DATA}\n"
    "train: images/train\n"
    "val: images/val\n"
    f"nc: {len(class_names)}\n"
    f"names: {class_names}\n"
)

write_json(
    WORK / "metadata.json",
    {
        "kind": "training",
        "track": "YOLO26s",
        "platform": "kaggle",
        "training_run_id": FOLD_CONFIG["run_id"],
        "fold_index": FOLD_CONFIG["fold_index"],
        "fold_name": FOLD_CONFIG["fold_name"],
        "config_source": f"{FOLD_CONFIG['kernel_dir_name']}/kernel_train.py",
        "manifest_name": MANIFEST["name"],
        "manifest_file": FOLD_CONFIG["manifest_file"],
        "resolved_config": "resolved-config.yaml",
        "results_artifact": "results.csv",
        "selected_checkpoint": "best.pt",
        "last_checkpoint": "last.pt",
        "periodic_checkpoint_dir": "checkpoints",
        "stdout_log": "train.log",
    },
)

RUN_NAME = FOLD_CONFIG["run_name"]
WANDB_PROJECT = "fathomnet2026"
if USE_WANDB:
    import wandb

    wandb.login(key=WANDB_API_KEY)
    os.environ["WANDB_ENTITY"] = "little_farms"
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"wandb": True})
    except Exception as exc:
        print(f"[wandb] settings update failed: {exc}")
    print(f"[wandb] logged in — entity=little_farms project={WANDB_PROJECT} run={RUN_NAME}")

import torch
from ultralytics import YOLO

device = [0, 1] if torch.cuda.device_count() >= 2 else 0
print(f"[train] device={device} cuda_count={torch.cuda.device_count()}")

model = YOLO("yolo26s.pt")
model.train(
    data=str(yaml_path),
    epochs=50,
    patience=0,
    imgsz=1024,
    rect=False,
    batch=16,
    save_period=5,
    workers=4,
    optimizer="auto",
    lr0=0.01,
    cos_lr=True,
    mosaic=0.85,
    close_mosaic=10,
    hsv_h=0.01,
    hsv_s=0.2,
    hsv_v=0.15,
    translate=0.05,
    scale=0.5,
    fliplr=0.5,
    project=WANDB_PROJECT,
    name=RUN_NAME,
    exist_ok=True,
    device=device,
    amp=True,
    plots=True,
)

save_dir = Path(model.trainer.save_dir)
weights_dir = save_dir / "weights"
for src_name, dst_name in [("best.pt", "best.pt"), ("last.pt", "last.pt")]:
    src = weights_dir / src_name
    if src.exists():
        shutil.copy2(src, WORK / dst_name)
        print(f"[train] copied {src_name} -> /kaggle/working/{dst_name}")
ckpt_dst = WORK / "checkpoints"
ckpt_dst.mkdir(exist_ok=True)
for checkpoint in sorted(weights_dir.glob("epoch*.pt")):
    shutil.copy2(checkpoint, ckpt_dst / checkpoint.name)
    print(f"[train] checkpoint saved: checkpoints/{checkpoint.name}")

resolved_cfg_src = save_dir / "args.yaml"
if resolved_cfg_src.exists():
    shutil.copy2(resolved_cfg_src, WORK / "resolved-config.yaml")
results_csv_src = save_dir / "results.csv"
if results_csv_src.exists():
    shutil.copy2(results_csv_src, WORK / "results.csv")

summary_csv = WORK / "fold-summary.csv"
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["run_id", "fold_index", "fold_name", "train_images", "val_images"])
    writer.writeheader()
    writer.writerow(
        {
            "run_id": FOLD_CONFIG["run_id"],
            "fold_index": FOLD_CONFIG["fold_index"],
            "fold_name": FOLD_CONFIG["fold_name"],
            "train_images": len(train_ids),
            "val_images": len(val_ids),
        }
    )

try:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
except Exception:
    pass

print("[done] training-only kernel complete")

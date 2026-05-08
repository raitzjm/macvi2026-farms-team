#!/usr/bin/env python3
"""
FathomNet-CLEF 2026 — T-019 final YOLO26s retrain on Kaggle.

Training-only kernel. Checkpoints and artifacts are pulled back locally after
completion; inference and post-processing remain local.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


RUN_ID = "T-019"
RUN_NAME = "t014-yolo26s-final-T-019"
WANDB_PROJECT = "fathomnet2026"
WORK = Path("/kaggle/working")
IMAGE_DATASET_ROOT = Path("/kaggle/input/fathomnetclef2026")


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
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["_TORCH_COMPAT"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

print("[setup] pip installing deps")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.42", "--no-deps"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb==0.26.1", "pycocotools==2.0.11"])

WANDB_API_KEY = "__WANDB_KEY_PLACEHOLDER__"
if WANDB_API_KEY.startswith("wandb_"):
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    USE_WANDB = True
    print("[setup] WANDB_API_KEY set via push-time injection")
else:
    raise RuntimeError("WANDB key was not injected; push this kernel with --wandb-key")


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


def resolve_uploaded_input(filename: str) -> Path:
    candidates = sorted(Path("/kaggle/input").rglob(filename))
    if not candidates:
        raise RuntimeError(
            f"Required uploaded file {filename} not found under /kaggle/input; "
            f"top-level entries={sorted(path.name for path in Path('/kaggle/input').iterdir())}"
        )
    if len(candidates) > 1:
        print(f"[data] multiple candidates for {filename}, using first: {candidates[0]}")
    return candidates[0]


def validate_image_index(coco: dict, stem_index: dict[str, Path], image_ids: list[int]):
    img_by_id = {int(image["id"]): image for image in coco["images"]}
    missing = []
    for image_id in image_ids:
        image = img_by_id[image_id]
        stem = Path(image["file_name"]).stem
        if stem not in stem_index:
            missing.append(stem)
    if missing:
        sample = ", ".join(missing[:10])
        raise RuntimeError(f"Missing {len(missing)} image stems from mounted mirror; first missing: {sample}")


def coco_to_yolo(coco: dict, dst_root: Path, stem_index: dict[str, Path], image_ids: list[int]):
    cats = sorted(coco["categories"], key=lambda cat: cat["id"])
    cat_id_to_idx = {cat["id"]: index for index, cat in enumerate(cats)}
    idx_to_cat_id = {index: cat["id"] for index, cat in enumerate(cats)}
    class_names = [cat["name"] for cat in cats]
    img_by_id = {int(image["id"]): image for image in coco["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    for split_name in ("train", "val"):
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


def resolve_image_root() -> Path:
    candidates = [
        IMAGE_DATASET_ROOT / "images",
        IMAGE_DATASET_ROOT,
        Path("/kaggle/input/fathomnet-2026") / "images",
        Path("/kaggle/input/fathomnet-2026"),
    ]
    for candidate in candidates:
        if (candidate / "train").exists() and any((candidate / "train").iterdir()):
            return candidate
    for path in Path("/kaggle/input").rglob("train"):
        if path.is_dir() and any(path.iterdir()):
            return path.parent
    raise RuntimeError(
        "Image mirror not found under /kaggle/input; "
        f"top-level entries={sorted(path.name for path in Path('/kaggle/input').iterdir())}"
    )


EXPANDED_COCO = resolve_uploaded_input("t014-expanded-train.json")
EXPANDED_MANIFEST = resolve_uploaded_input("t014-expanded-manifest.json")
PSEUDO_LABEL_JSON = resolve_uploaded_input("t014-pseudo-labels.json")
BUNDLE_SUMMARY_JSON = resolve_uploaded_input("bundle-summary.json")
EXPANDED_INPUT = EXPANDED_COCO.parent
IMG_ROOT = resolve_image_root()

print(f"[cfg] run_id={RUN_ID}")
print("[cfg] machine_shape=NvidiaTeslaT4")
print(f"[data] image root: {IMG_ROOT}")
print(f"[data] expanded bundle root: {EXPANDED_INPUT}")

coco = json.loads(EXPANDED_COCO.read_text())
manifest = json.loads(EXPANDED_MANIFEST.read_text())
validate = {
    "image_count": len(coco["images"]),
    "annotation_count": len(coco["annotations"]),
    "category_count": len(coco["categories"]),
    "sha256": canonical_sha256(coco),
}
print(f"[data] expanded stats: {validate}")
image_ids = [int(image_id) for image_id in manifest["train_ids"]]
print(f"[data] train/val image count={len(image_ids)}")
bundle_summary = json.loads(BUNDLE_SUMMARY_JSON.read_text())

YOLO_DATA = WORK / "yolo_data"
TRAIN_IMG_INDEX = build_stem_index(IMG_ROOT / "train")
validate_image_index(coco, TRAIN_IMG_INDEX, image_ids)
class_names, idx_to_cat_id = coco_to_yolo(coco, YOLO_DATA, TRAIN_IMG_INDEX, image_ids)
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
        "training_run_id": RUN_ID,
        "config_source": "kaggle/yolo26s-t019-final/kernel_train.py",
        "expanded_coco": str(EXPANDED_COCO.name),
        "expanded_manifest": str(EXPANDED_MANIFEST.name),
        "pseudo_label_json": str(PSEUDO_LABEL_JSON.name),
        "bundle_summary": str(BUNDLE_SUMMARY_JSON.name),
        "bundle_file_sha256": {key: value["sha256"] for key, value in bundle_summary["files"].items()},
        "resolved_config": "resolved-config.yaml",
        "results_artifact": "results.csv",
        "selected_checkpoint": "best.pt",
        "last_checkpoint": "last.pt",
        "periodic_checkpoint_dir": "checkpoints",
        "stdout_log": "train.log",
    },
)
shutil.copy2(BUNDLE_SUMMARY_JSON, WORK / "bundle-summary.json")

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

for image_name in ["labels.jpg", "results.png", "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg"]:
    src = save_dir / image_name
    if src.exists():
        shutil.copy2(src, WORK / image_name)

resolved_cfg_src = save_dir / "args.yaml"
if resolved_cfg_src.exists():
    shutil.copy2(resolved_cfg_src, WORK / "resolved-config.yaml")
results_csv_src = save_dir / "results.csv"
if results_csv_src.exists():
    shutil.copy2(results_csv_src, WORK / "results.csv")

try:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
except Exception:
    pass

print("[done] training-only kernel complete")

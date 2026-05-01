#!/usr/bin/env python3
"""
YOLO11 training for FathomNet-CLEF — runs locally and on Kaggle.

Dataset path resolves via _paths.py. Trains with rectangular batches
(rect=True) at 640px short side (imgsz=1152 → height ~640 for 16:9 inputs).
Builds YOLO labels from COCO JSON into outputs/<name>/yolo_data/, with a
deterministic 90/10 train/val split (seed=42). WandB logs via env vars.

Usage:
  python train_yolo_kaggle.py --epochs 1 --batch 8 --name smoke-yolo
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _paths import DATASET_ROOT, TRAIN_JSON, TRAIN_IMGS  # noqa: E402
from run_tracking import (  # noqa: E402
    copy_if_exists,
    git_commit,
    relpath_str,
    resolve_artifact_dir,
    tee_output,
    tracked_run_name,
    write_json,
)


def load_wandb_key() -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    return None


def coco_to_yolo(coco_json: Path, dst_root: Path, train_imgs_dir: Path,
                 val_split: float = 0.1, seed: int = 42):
    """Build YOLO labels + symlinked image dirs split into train/val."""
    coco = json.loads(Path(coco_json).read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
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
        for iid in ids:
            im = img_by_id[iid]
            stem = Path(im["file_name"]).stem
            src = Path(train_imgs_dir) / im["file_name"]
            dst = images_dir / im["file_name"]
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

    return class_names, splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--width", type=int, default=1152, help="Long-side target")
    p.add_argument("--height", type=int, default=640,
                   help="Short-side target (informational; rect derives from imgsz)")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Override; defaults to max(width, height)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--name", default="smoke-yolo")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--run-id", default=None,
                   help="Optional tracked training ID (for example T-004)")
    p.add_argument("--artifact-dir", default=None,
                   help="Optional durable artifact directory override")
    p.add_argument("--no-rect", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    imgsz = args.imgsz if args.imgsz is not None else max(args.width, args.height)
    run_name = tracked_run_name(args.name, args.run_id)

    challenge_root = Path(__file__).resolve().parent.parent
    out_root = challenge_root / "outputs" / run_name
    artifact_dir = resolve_artifact_dir(challenge_root, "training", args.run_id, args.artifact_dir)
    log_path = artifact_dir / "stdout.log" if artifact_dir else None

    with tee_output(log_path):
        out_root.mkdir(parents=True, exist_ok=True)
        yolo_data = out_root / "yolo_data"

        print(f"Dataset:  {DATASET_ROOT}")
        print(f"Output:   {out_root}")
        print(f"Run:      {run_name}")
        print(f"imgsz={imgsz} (w={args.width} h={args.height}) rect={not args.no_rect} batch={args.batch}")

        class_names, splits = coco_to_yolo(TRAIN_JSON, yolo_data, TRAIN_IMGS)
        print(f"Built YOLO data: {len(splits['train'])} train / {len(splits['val'])} val")

        yaml_path = yolo_data / "fathomnet.yaml"
        yaml_path.write_text(
            f"path: {yolo_data}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names}\n"
        )

        if artifact_dir:
            write_json(
                artifact_dir / "resolved-config.json",
                {
                    "batch": args.batch,
                    "class_count": len(class_names),
                    "class_names": class_names,
                    "config_source": "scripts/train_yolo.py",
                    "dataset_root": str(DATASET_ROOT),
                    "epochs": args.epochs,
                    "height": args.height,
                    "imgsz": imgsz,
                    "model": args.model,
                    "name": run_name,
                    "rect": not args.no_rect,
                    "requested_name": args.name,
                    "seed": 42,
                    "train_json": str(TRAIN_JSON),
                    "val_split": 0.1,
                    "wandb_enabled": not args.no_wandb,
                    "width": args.width,
                    "workers": args.workers,
                    "yaml_path": relpath_str(yaml_path, challenge_root),
                },
            )
            write_json(
                artifact_dir / "metadata.json",
                {
                    "artifact_dir": relpath_str(artifact_dir, challenge_root),
                    "config_source": "scripts/train_yolo.py",
                    "git_commit": git_commit(challenge_root),
                    "kind": "training",
                    "metrics_artifact": relpath_str(artifact_dir / "results.csv", challenge_root),
                    "output_root": relpath_str(out_root, challenge_root),
                    "platform": "local",
                    "resolved_config": relpath_str(artifact_dir / "resolved-config.json", challenge_root),
                    "run_id": args.run_id,
                    "run_name": run_name,
                    "stdout_log": relpath_str(artifact_dir / "stdout.log", challenge_root),
                    "track": "YOLO",
                    "train_labels_yaml": relpath_str(yaml_path, challenge_root),
                },
            )

        if not args.no_wandb:
            key = load_wandb_key()
            if key:
                os.environ["WANDB_API_KEY"] = key
                import wandb as _wandb
                _orig_wandb_init = _wandb.init
                _run_name = run_name

                def _patched_wandb_init(*a, **kw):
                    kw["project"] = "fathomnet2026"
                    kw["entity"] = "little_farms"
                    kw["name"] = _run_name
                    kw.pop("id", None)
                    return _orig_wandb_init(*a, **kw)

                _wandb.init = _patched_wandb_init
            else:
                print("[warn] no WANDB_API_KEY; running without wandb")

        from ultralytics import YOLO
        model = YOLO(args.model)
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            rect=not args.no_rect,
            workers=args.workers,
            project=str(out_root),
            name="run",
            exist_ok=True,
            device=0,
            amp=True,
        )

        if artifact_dir:
            copy_if_exists(out_root / "run" / "results.csv", artifact_dir / "results.csv")
            copy_if_exists(out_root / "run" / "args.yaml", artifact_dir / "ultralytics-args.yaml")
            best_ckpt = out_root / "run" / "weights" / "best.pt"
            if best_ckpt.exists():
                write_json(
                    artifact_dir / "selected-checkpoint.json",
                    {
                        "relative_path": relpath_str(best_ckpt, challenge_root),
                        "selection": "training framework best checkpoint",
                    },
                )
        print("OK")


if __name__ == "__main__":
    main()

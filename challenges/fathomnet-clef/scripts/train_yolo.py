#!/usr/bin/env python3
"""
YOLO11 training for FathomNet-CLEF — runs locally and on Kaggle.

Dataset path resolves via _paths.py. Trains with rectangular batches
(rect=True) at 640px short side (imgsz=1152 → height ~640 for 16:9 inputs).
Builds YOLO labels from COCO JSON into outputs/<name>/yolo_data/, using a
shared explicit train/val split manifest. WandB logs via env vars.

Usage:
  python train_yolo_kaggle.py --epochs 1 --batch 8 --name smoke-yolo
"""
import argparse
import json
import os
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
from split_manifest import (  # noqa: E402
    default_split_manifest_path,
    ensure_manifest,
    load_manifest,
)


AUGMENT_PROFILES = {
    "ultralytics_default": {},
    "underwater_v1": {
        "hsv_h": 0.01,
        "hsv_s": 0.2,
        "hsv_v": 0.15,
        "translate": 0.05,
        "scale": 0.2,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "close_mosaic": 0,
        "auto_augment": None,
    },
}


def load_wandb_key() -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    return None


def setup_wandb(
    run_name: str,
    enabled: bool,
    *,
    settings=None,
    wandb_module=None,
) -> bool:
    if not enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return False

    key = load_wandb_key()
    if not key:
        print("[warn] no WANDB_API_KEY; running without wandb")
        os.environ["WANDB_DISABLED"] = "true"
        return False

    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_PROJECT"] = "fathomnet2026"
    os.environ["WANDB_ENTITY"] = "little_farms"
    os.environ["WANDB_RUN_NAME"] = run_name
    os.environ.pop("WANDB_DISABLED", None)

    if settings is None:
        try:
            from ultralytics.utils import SETTINGS as settings  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[warn] unable to load Ultralytics settings for wandb: {exc}")
            settings = None
    if settings is not None:
        settings["wandb"] = True

    if wandb_module is None:
        try:
            import wandb as wandb_module  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[warn] wandb package unavailable; running without wandb: {exc}")
            os.environ["WANDB_DISABLED"] = "true"
            return False

    try:
        wandb_module.login(key=key, relogin=True)
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[warn] wandb login failed; continuing with best effort: {exc}")

    _orig_wandb_init = wandb_module.init

    def _patched_wandb_init(*a, **kw):
        kw["project"] = "fathomnet2026"
        kw["entity"] = "little_farms"
        kw["name"] = run_name
        kw.pop("id", None)
        kw.setdefault("resume", "never")
        print(f"[wandb] init project={kw['project']} entity={kw['entity']} name={kw['name']}")
        return _orig_wandb_init(*a, **kw)

    wandb_module.init = _patched_wandb_init
    print(f"[wandb] enabled for run {run_name}")
    return True


def augmentation_overrides(profile: str) -> dict:
    if profile not in AUGMENT_PROFILES:
        raise ValueError(f"Unknown augmentation profile: {profile}")
    return dict(AUGMENT_PROFILES[profile])


def build_train_overrides(
    *,
    yaml_path: Path,
    out_root: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    rect: bool,
    seed: int,
    augment_profile: str,
) -> dict:
    overrides = {
        "data": str(yaml_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "rect": rect,
        "workers": workers,
        "project": str(out_root),
        "name": "run",
        "exist_ok": True,
        "device": 0,
        "amp": True,
        "plots": True,
        "seed": seed,
        "deterministic": True,
    }
    overrides.update(augmentation_overrides(augment_profile))
    return overrides


def build_resolved_config_payload(
    *,
    batch: int,
    class_names: list[str],
    dataset_root: str,
    epochs: int,
    height: int,
    imgsz: int,
    model: str,
    requested_name: str,
    run_name: str,
    rect: bool,
    seed: int,
    split_manifest: str,
    train_json: str,
    wandb_enabled: bool,
    width: int,
    workers: int,
    yaml_path: str,
    augment_profile: str,
    augment_overrides: dict,
) -> dict:
    return {
        "augment_overrides": augment_overrides,
        "augment_profile": augment_profile,
        "batch": batch,
        "class_count": len(class_names),
        "class_names": class_names,
        "config_source": "scripts/train_yolo.py",
        "dataset_root": dataset_root,
        "epochs": epochs,
        "height": height,
        "imgsz": imgsz,
        "model": model,
        "name": run_name,
        "rect": rect,
        "requested_name": requested_name,
        "seed": seed,
        "split_manifest": split_manifest,
        "train_json": train_json,
        "wandb_enabled": wandb_enabled,
        "width": width,
        "workers": workers,
        "yaml_path": yaml_path,
    }


def coco_to_yolo(coco_json: Path, dst_root: Path, train_imgs_dir: Path, split_manifest_path: Path):
    """Build YOLO labels + symlinked image dirs using an explicit split manifest."""
    coco = json.loads(Path(coco_json).read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    class_names = [c["name"] for c in cats]

    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    manifest = load_manifest(split_manifest_path)
    splits = {"train": manifest["train_ids"], "val": manifest["val_ids"]}

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
    p.add_argument("--split-manifest", default=str(default_split_manifest_path()),
                   help="Path to shared train/val split manifest JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--augment-profile",
        default="underwater_v1",
        choices=sorted(AUGMENT_PROFILES.keys()),
        help="Named augmentation profile for comparable YOLO experiments",
    )
    p.add_argument("--no-rect", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    imgsz = args.imgsz if args.imgsz is not None else max(args.width, args.height)
    run_name = tracked_run_name(args.name, args.run_id)
    wandb_enabled = setup_wandb(run_name, not args.no_wandb)

    challenge_root = Path(__file__).resolve().parent.parent
    out_root = challenge_root / "outputs" / run_name
    artifact_dir = resolve_artifact_dir(challenge_root, "training", args.run_id, args.artifact_dir)
    log_path = artifact_dir / "stdout.log" if artifact_dir else None

    with tee_output(log_path):
        out_root.mkdir(parents=True, exist_ok=True)
        yolo_data = out_root / "yolo_data"
        split_manifest_path = ensure_manifest(TRAIN_JSON, Path(args.split_manifest))

        print(f"Dataset:  {DATASET_ROOT}")
        print(f"Output:   {out_root}")
        print(f"Run:      {run_name}")
        print(f"Split:    {split_manifest_path}")
        print(f"imgsz={imgsz} (w={args.width} h={args.height}) rect={not args.no_rect} batch={args.batch}")

        class_names, splits = coco_to_yolo(TRAIN_JSON, yolo_data, TRAIN_IMGS, split_manifest_path)
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
                build_resolved_config_payload(
                    batch=args.batch,
                    class_names=class_names,
                    dataset_root=str(DATASET_ROOT),
                    epochs=args.epochs,
                    height=args.height,
                    imgsz=imgsz,
                    model=args.model,
                    requested_name=args.name,
                    run_name=run_name,
                    rect=not args.no_rect,
                    seed=args.seed,
                    split_manifest=relpath_str(split_manifest_path, challenge_root),
                    train_json=str(TRAIN_JSON),
                    wandb_enabled=wandb_enabled,
                    width=args.width,
                    workers=args.workers,
                    yaml_path=relpath_str(yaml_path, challenge_root),
                    augment_profile=args.augment_profile,
                    augment_overrides=augmentation_overrides(args.augment_profile),
                ),
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
                    "split_manifest": relpath_str(split_manifest_path, challenge_root),
                    "stdout_log": relpath_str(artifact_dir / "stdout.log", challenge_root),
                    "track": "YOLO",
                    "train_labels_yaml": relpath_str(yaml_path, challenge_root),
                },
            )

        from ultralytics import YOLO
        model = YOLO(args.model)
        model.train(**build_train_overrides(
            yaml_path=yaml_path,
            out_root=out_root,
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            workers=args.workers,
            rect=not args.no_rect,
            seed=args.seed,
            augment_profile=args.augment_profile,
        ))

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

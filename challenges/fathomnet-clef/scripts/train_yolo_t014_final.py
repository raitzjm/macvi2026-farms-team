#!/usr/bin/env python3
"""Final local YOLO26s retraining script for the T-014 k-fold pseudo-label family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _paths import DATASET_ROOT, TRAIN_IMGS, TRAIN_JSON  # noqa: E402
from run_tracking import (  # noqa: E402
    copy_if_exists,
    git_commit,
    relpath_str,
    resolve_artifact_dir,
    tee_output,
    tracked_run_name,
    write_json,
)
from train_yolo import build_train_overrides, build_resolved_config_payload, setup_wandb  # noqa: E402


FINAL_AUGMENT_PROFILE = "t014_final_locked"


def build_full_dataset_manifest(coco: dict) -> dict:
    image_ids = sorted(int(image["id"]) for image in coco["images"])
    return {
        "manifest_version": 1,
        "name": "t014-expanded-100pct-v1",
        "notes": [
            "T-014 final retrain manifest.",
            "train and val both point to the full expanded dataset by lead policy.",
        ],
        "train_ids": image_ids,
        "val_ids": image_ids,
    }


def merge_pseudo_annotations(coco: dict, pseudo: dict) -> dict:
    merged = {
        **coco,
        "annotations": [dict(ann) for ann in coco["annotations"]],
    }
    next_id = max((int(ann["id"]) for ann in merged["annotations"]), default=0) + 1
    for ann in pseudo.get("annotations", []):
        merged["annotations"].append(
            {
                **ann,
                "id": next_id,
            }
        )
        next_id += 1
    return merged


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_coco(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_full_yolo_split_map(coco: dict) -> tuple[list[str], dict[str, list[int]]]:
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    class_names = [c["name"] for c in cats]
    image_ids = sorted(int(image["id"]) for image in coco["images"])
    return class_names, {"train": image_ids, "val": image_ids}


def build_full_dataset_yolo_labels(coco: dict, dst_root: Path, train_imgs_dir: Path) -> tuple[list[str], dict[str, list[int]]]:
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    class_names, splits = build_full_yolo_split_map(coco)
    img_by_id = {int(im["id"]): im for im in coco["images"]}
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    for split_name, ids in splits.items():
        labels_dir = dst_root / "labels" / split_name
        images_dir = dst_root / "images" / split_name
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            image = img_by_id[image_id]
            src = Path(train_imgs_dir) / image["file_name"]
            dst = images_dir / image["file_name"]
            if not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    import shutil
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
            (labels_dir / f"{Path(image['file_name']).stem}.txt").write_text("\n".join(lines))

    return class_names, splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo-label-json", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="t014-yolo26s-final")
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    challenge_root = Path(__file__).resolve().parent.parent
    run_name = tracked_run_name(args.name, args.run_id)
    artifact_dir = resolve_artifact_dir(challenge_root, "training", args.run_id, args.artifact_dir)
    log_path = artifact_dir / "stdout.log" if artifact_dir else None
    wandb_enabled = setup_wandb(run_name, not args.no_wandb)

    with tee_output(log_path):
        out_root = challenge_root / "outputs" / run_name
        out_root.mkdir(parents=True, exist_ok=True)

        coco = json.loads(Path(TRAIN_JSON).read_text())
        pseudo = json.loads(Path(args.pseudo_label_json).read_text())
        merged_coco = merge_pseudo_annotations(coco, pseudo)

        manifest_path = out_root / "t014-expanded-manifest.json"
        write_manifest(manifest_path, build_full_dataset_manifest(merged_coco))

        merged_coco_path = out_root / "expanded-train.json"
        write_coco(merged_coco_path, merged_coco)

        yolo_data = out_root / "yolo_data"
        class_names, splits = build_full_dataset_yolo_labels(merged_coco, yolo_data, TRAIN_IMGS)
        print(f"Built expanded YOLO data: {len(splits['train'])} train / {len(splits['val'])} val")

        yaml_path = yolo_data / "fathomnet.yaml"
        yaml_path.write_text(
            f"path: {yolo_data}\n"
            "train: images/train\n"
            "val: images/val\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names}\n"
        )

        augment_overrides = {
            "hsv_h": 0.01,
            "hsv_s": 0.2,
            "hsv_v": 0.15,
            "translate": 0.05,
            "scale": 0.5,
            "fliplr": 0.5,
            "mosaic": 0.85,
            "close_mosaic": 10,
            "cos_lr": True,
        }

        if artifact_dir:
            write_json(
                artifact_dir / "resolved-config.json",
                build_resolved_config_payload(
                    batch=args.batch,
                    class_names=class_names,
                    dataset_root=str(DATASET_ROOT),
                    epochs=args.epochs,
                    height=args.imgsz,
                    imgsz=args.imgsz,
                    model=args.model,
                    requested_name=args.name,
                    run_name=run_name,
                    rect=False,
                    seed=args.seed,
                    split_manifest=relpath_str(manifest_path, challenge_root),
                    train_json=relpath_str(merged_coco_path, challenge_root),
                    wandb_enabled=wandb_enabled,
                    width=args.imgsz,
                    workers=args.workers,
                    yaml_path=relpath_str(yaml_path, challenge_root),
                    augment_profile=FINAL_AUGMENT_PROFILE,
                    augment_overrides=augment_overrides,
                ),
            )
            write_json(
                artifact_dir / "metadata.json",
                {
                    "artifact_dir": relpath_str(artifact_dir, challenge_root),
                    "config_source": "scripts/train_yolo_t014_final.py",
                    "git_commit": git_commit(challenge_root),
                    "kind": "training",
                    "metrics_artifact": relpath_str(artifact_dir / "results.csv", challenge_root),
                    "output_root": relpath_str(out_root, challenge_root),
                    "platform": "local RTX 3050 4GB",
                    "pseudo_label_json": relpath_str(Path(args.pseudo_label_json), challenge_root),
                    "resolved_config": relpath_str(artifact_dir / "resolved-config.json", challenge_root),
                    "run_id": args.run_id,
                    "run_name": run_name,
                    "split_manifest": relpath_str(manifest_path, challenge_root),
                    "stdout_log": relpath_str(artifact_dir / "stdout.log", challenge_root),
                    "track": "YOLO26s",
                    "train_labels_yaml": relpath_str(yaml_path, challenge_root),
                },
            )

        from ultralytics import YOLO

        model = YOLO(args.model)
        overrides = build_train_overrides(
            yaml_path=yaml_path,
            out_root=out_root,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            rect=False,
            seed=args.seed,
            augment_profile="ultralytics_default",
            save_period=args.save_period,
        )
        overrides.update(
            {
                "device": args.device,
                "cache": False,
                "patience": 0,
                "amp": True,
                "cos_lr": True,
            }
        )
        overrides.update(augment_overrides)
        model.train(**overrides)

        if artifact_dir:
            copy_if_exists(out_root / "run" / "results.csv", artifact_dir / "results.csv")
            copy_if_exists(out_root / "run" / "args.yaml", artifact_dir / "ultralytics-args.yaml")
            copy_if_exists(out_root / "run" / "weights" / "epoch45.pt", artifact_dir / "epoch45.pt")
            copy_if_exists(out_root / "run" / "weights" / "last.pt", artifact_dir / "last.pt")


if __name__ == "__main__":
    main()

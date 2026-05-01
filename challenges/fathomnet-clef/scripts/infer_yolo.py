#!/usr/bin/env python3
"""Local YOLO inference utility for FathomNet-CLEF test set."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _paths import TEST_IMGS, TEST_JSON, TRAIN_JSON  # noqa: E402
from run_tracking import (  # noqa: E402
    git_commit,
    relpath_str,
    resolve_artifact_dir,
    tee_output,
    tracked_submission_path,
    write_json,
)


CSV_COLUMNS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "score",
]


def _load_json(path: Path):
    return json.loads(path.read_text())


def load_test_images(test_json: Path) -> list[dict]:
    data = _load_json(test_json)
    return sorted(data["images"], key=lambda x: x["id"])


def derive_idx_to_cat_id(train_json: Path) -> dict[int, int]:
    data = _load_json(train_json)
    cats = sorted(data["categories"], key=lambda c: c["id"])
    return {idx: cat["id"] for idx, cat in enumerate(cats)}


def load_idx_to_cat_id(idx_to_cat_id_json: str | None, train_json: Path) -> dict[int, int]:
    if idx_to_cat_id_json is None:
        return derive_idx_to_cat_id(train_json)
    raw = _load_json(Path(idx_to_cat_id_json))
    return {int(k): int(v) for k, v in raw.items()}


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return x1, y1, x2 - x1, y2 - y1


def chunked(items: list, size: int):
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def chunked_pairs(items_a: list, items_b: list, size: int):
    if len(items_a) != len(items_b):
        raise ValueError("paired chunk inputs must have the same length")
    for start in range(0, len(items_a), size):
        end = start + size
        yield items_a[start:end], items_b[start:end]


def should_log_progress(processed: int, total: int, interval: int = 50) -> bool:
    return processed == total or processed % interval == 0


def has_positive_area(width: float, height: float) -> bool:
    return width > 0 and height > 0


def write_submission_csv(output_csv: Path, rows: list[dict]):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sahi_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized == "cpu" or normalized.startswith("cuda"):
        return normalized
    return f"cuda:{normalized}"


def assert_gpu_if_needed(device: str):
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")


def run_regular_yolo(
    checkpoint: str,
    images: list[dict],
    idx_to_cat_id: dict[int, int],
    output_csv: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    batch: int,
):
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    image_paths = [str(TEST_IMGS / im["file_name"]) for im in images]
    image_ids = [im["id"] for im in images]

    rows = []
    annotation_id = 1
    total = len(image_paths)
    processed = 0
    for image_chunk, image_id_chunk in chunked_pairs(image_paths, image_ids, batch):
        results = model.predict(
            source=image_chunk,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            batch=batch,
            verbose=False,
            stream=False,
        )
        for image_id, res in zip(image_id_chunk, results):
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.tolist()
                confs = res.boxes.conf.tolist()
                classes = res.boxes.cls.tolist()
                for box, score, cls_idx in zip(xyxy, confs, classes):
                    x, y, w, h = xyxy_to_xywh(*box)
                    if not has_positive_area(w, h):
                        continue
                    rows.append(
                        {
                            "annotation_id": annotation_id,
                            "image_id": image_id,
                            "category_id": idx_to_cat_id[int(cls_idx)],
                            "bbox_x": float(x),
                            "bbox_y": float(y),
                            "bbox_width": float(w),
                            "bbox_height": float(h),
                            "score": float(score),
                        }
                    )
                    annotation_id += 1
            processed += 1
            if should_log_progress(processed, total):
                print(f"[infer] processed {processed}/{total} images", flush=True)

    write_submission_csv(output_csv, rows)


def run_sahi_yolo(
    checkpoint: str,
    images: list[dict],
    idx_to_cat_id: dict[int, int],
    output_csv: Path,
    imgsz: int,
    conf: float,
    device: str,
    slice_size: int,
    overlap: float,
    postprocess_match_threshold: float,
):
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except Exception as exc:
        raise RuntimeError("SAHI not available; install sahi to use --sahi") from exc

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=checkpoint,
        confidence_threshold=conf,
        device=sahi_device(device),
        image_size=imgsz,
    )

    rows = []
    annotation_id = 1
    total = len(images)
    for processed, im in enumerate(images, start=1):
        image_path = TEST_IMGS / im["file_name"]
        pred = get_sliced_prediction(
            image=str(image_path),
            detection_model=detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_match_threshold=postprocess_match_threshold,
        )
        for obj in pred.object_prediction_list:
            x, y, w, h = obj.bbox.to_xywh()
            if not has_positive_area(w, h):
                continue
            cls_idx = int(obj.category.id)
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": im["id"],
                    "category_id": idx_to_cat_id[cls_idx],
                    "bbox_x": float(x),
                    "bbox_y": float(y),
                    "bbox_width": float(w),
                    "bbox_height": float(h),
                    "score": float(obj.score.value),
                }
            )
            annotation_id += 1
        if should_log_progress(processed, total, interval=25):
            print(f"[infer] processed {processed}/{total} images", flush=True)

    write_submission_csv(output_csv, rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--run-id", default=None,
                   help="Optional tracked inference ID (for example I-009)")
    p.add_argument("--artifact-dir", default=None,
                   help="Optional durable artifact directory override")
    p.add_argument("--source-train-id", default=None,
                   help="Training run ID that produced the primary checkpoint")
    p.add_argument("--imgsz", type=int, default=1152)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--idx-to-cat-id", default=None)
    p.add_argument("--sahi", action="store_true")
    p.add_argument("--slice-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--postprocess-match-threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    assert_gpu_if_needed(args.device)

    challenge_root = Path(__file__).resolve().parent.parent
    artifact_dir = resolve_artifact_dir(challenge_root, "inference", args.run_id, args.artifact_dir)
    log_path = artifact_dir / "stdout.log" if artifact_dir else None

    with tee_output(log_path):
        images = load_test_images(TEST_JSON)
        idx_to_cat_id = load_idx_to_cat_id(args.idx_to_cat_id, TRAIN_JSON)

        output_csv = tracked_submission_path(Path(args.output), args.run_id, args.source_train_id)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        if artifact_dir:
            write_json(
                artifact_dir / "resolved-config.json",
                {
                    "batch": args.batch,
                    "checkpoint": args.checkpoint,
                    "conf": args.conf,
                    "config_source": "scripts/infer_yolo.py",
                    "device": args.device,
                    "idx_to_cat_id": args.idx_to_cat_id,
                    "imgsz": args.imgsz,
                    "iou": args.iou,
                    "output_csv": relpath_str(output_csv, challenge_root),
                    "requested_output_csv": args.output,
                    "postprocess_match_threshold": args.postprocess_match_threshold,
                    "sahi": args.sahi,
                    "slice_size": args.slice_size,
                    "overlap": args.overlap,
                    "source_train_id": args.source_train_id,
                },
            )
            write_json(
                artifact_dir / "metadata.json",
                {
                    "artifact_dir": relpath_str(artifact_dir, challenge_root),
                    "checkpoint_path": args.checkpoint,
                    "config_source": "scripts/infer_yolo.py",
                    "git_commit": git_commit(challenge_root),
                    "kind": "inference",
                    "output_csv": relpath_str(output_csv, challenge_root),
                    "resolved_config": relpath_str(artifact_dir / "resolved-config.json", challenge_root),
                    "run_id": args.run_id,
                    "source_train_id": args.source_train_id,
                    "stdout_log": relpath_str(artifact_dir / "stdout.log", challenge_root),
                    "track": "YOLO",
                },
            )

        if args.sahi:
            run_sahi_yolo(
                checkpoint=args.checkpoint,
                images=images,
                idx_to_cat_id=idx_to_cat_id,
                output_csv=output_csv,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                slice_size=args.slice_size,
                overlap=args.overlap,
                postprocess_match_threshold=args.postprocess_match_threshold,
            )
        else:
            run_regular_yolo(
                checkpoint=args.checkpoint,
                images=images,
                idx_to_cat_id=idx_to_cat_id,
                output_csv=output_csv,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                batch=args.batch,
            )

        print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()

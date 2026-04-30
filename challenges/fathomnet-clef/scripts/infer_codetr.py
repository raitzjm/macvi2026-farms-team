#!/usr/bin/env python3
"""Local Co-DETR inference utility for FathomNet-CLEF test set."""

from __future__ import annotations

import argparse
import sys
import types
from contextlib import contextmanager
from pathlib import Path


HERE = Path(__file__).resolve().parent
CHALLENGE_ROOT = HERE.parent
MMDET_SRC = CHALLENGE_ROOT / "envs" / "mmdet-src"
CODETR_PROJ = MMDET_SRC / "projects" / "CO-DETR"

sys.path.insert(0, str(HERE))

from _paths import TEST_IMGS, TEST_JSON, TRAIN_JSON  # noqa: E402
from infer_yolo import (  # noqa: E402
    assert_gpu_if_needed,
    has_positive_area,
    load_idx_to_cat_id,
    load_test_images,
    sahi_device,
    should_log_progress,
    write_submission_csv,
    xyxy_to_xywh,
)


def rows_from_xywh(
    image_id: int,
    boxes_xywh: list[tuple[float, float, float, float]],
    scores: list[float],
    labels: list[int],
    idx_to_cat_id: dict[int, int],
    annotation_id_start: int,
) -> tuple[list[dict], int]:
    rows = []
    annotation_id = annotation_id_start
    for (x, y, w, h), score, label in zip(boxes_xywh, scores, labels):
        if not has_positive_area(w, h):
            continue
        rows.append(
            {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": idx_to_cat_id[int(label)],
                "bbox_x": float(x),
                "bbox_y": float(y),
                "bbox_width": float(w),
                "bbox_height": float(h),
                "score": float(score),
            }
        )
        annotation_id += 1
    return rows, annotation_id


def rows_from_xyxy(
    image_id: int,
    boxes_xyxy: list[tuple[float, float, float, float]],
    scores: list[float],
    labels: list[int],
    idx_to_cat_id: dict[int, int],
    annotation_id_start: int,
) -> tuple[list[dict], int]:
    boxes_xywh = [xyxy_to_xywh(*box) for box in boxes_xyxy]
    return rows_from_xywh(image_id, boxes_xywh, scores, labels, idx_to_cat_id, annotation_id_start)


def resolve_default_config_path(checkpoint: Path) -> Path:
    sibling_config = checkpoint.parent / "config_used.py"
    if sibling_config.exists():
        return sibling_config

    raise FileNotFoundError(
        "No config_used.py found beside the checkpoint. Pass --config explicitly or use an archived Co-DETR artifact directory."
    )


def prepare_codetr_imports():
    if CODETR_PROJ.exists():
        sys.path.insert(0, str(CODETR_PROJ))

    import codetr  # noqa: F401

    if "projects" not in sys.modules:
        sys.modules["projects"] = types.ModuleType("projects")
    if "projects.CO-DETR" not in sys.modules:
        sys.modules["projects.CO-DETR"] = types.ModuleType("projects.CO-DETR")
    sys.modules["projects.CO-DETR.codetr"] = codetr


@contextmanager
def torch_load_weights_only_false():
    import torch

    orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        return orig_torch_load(*args, **{**kwargs, "weights_only": False})

    torch.load = _patched_torch_load
    try:
        yield
    finally:
        torch.load = orig_torch_load


def run_regular_codetr(
    checkpoint: str,
    config_path: Path,
    images: list[dict],
    idx_to_cat_id: dict[int, int],
    output_csv: Path,
    score_thr: float,
    device: str,
):
    prepare_codetr_imports()

    from mmdet.apis import inference_detector, init_detector

    with torch_load_weights_only_false():
        model = init_detector(str(config_path), checkpoint, device=f"cuda:{device}" if device.isdigit() else device)

    rows = []
    annotation_id = 1
    total = len(images)
    for processed, im in enumerate(images, start=1):
        image_path = TEST_IMGS / im["file_name"]
        result = inference_detector(model, str(image_path))
        pred_instances = result.pred_instances
        scores = pred_instances.scores.tolist()
        labels = pred_instances.labels.tolist()
        bboxes = pred_instances.bboxes.tolist()

        kept_bboxes = []
        kept_scores = []
        kept_labels = []
        for bbox, score, label in zip(bboxes, scores, labels):
            if float(score) < score_thr:
                continue
            kept_bboxes.append(tuple(float(v) for v in bbox))
            kept_scores.append(float(score))
            kept_labels.append(int(label))

        image_rows, annotation_id = rows_from_xyxy(
            image_id=im["id"],
            boxes_xyxy=kept_bboxes,
            scores=kept_scores,
            labels=kept_labels,
            idx_to_cat_id=idx_to_cat_id,
            annotation_id_start=annotation_id,
        )
        rows.extend(image_rows)
        if should_log_progress(processed, total, interval=25):
            print(f"[infer] processed {processed}/{total} images", flush=True)

    write_submission_csv(output_csv, rows)


def run_sahi_codetr(
    checkpoint: str,
    config_path: Path,
    images: list[dict],
    idx_to_cat_id: dict[int, int],
    output_csv: Path,
    score_thr: float,
    device: str,
    slice_size: int,
    overlap: float,
    postprocess_match_threshold: float,
):
    prepare_codetr_imports()

    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except Exception as exc:
        raise RuntimeError("SAHI not available; install sahi to use --sahi") from exc

    with torch_load_weights_only_false():
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="mmdet",
            model_path=checkpoint,
            config_path=str(config_path),
            confidence_threshold=score_thr,
            device=sahi_device(device),
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
        boxes_xywh = []
        scores = []
        labels = []
        for obj in pred.object_prediction_list:
            boxes_xywh.append(tuple(float(v) for v in obj.bbox.to_xywh()))
            scores.append(float(obj.score.value))
            labels.append(int(obj.category.id))

        image_rows, annotation_id = rows_from_xywh(
            image_id=im["id"],
            boxes_xywh=boxes_xywh,
            scores=scores,
            labels=labels,
            idx_to_cat_id=idx_to_cat_id,
            annotation_id_start=annotation_id,
        )
        rows.extend(image_rows)
        if should_log_progress(processed, total, interval=25):
            print(f"[infer] processed {processed}/{total} images", flush=True)

    write_submission_csv(output_csv, rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional. Defaults to config_used.py beside the checkpoint.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--idx-to-cat-id", default=None)
    parser.add_argument("--sahi", action="store_true")
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    assert_gpu_if_needed(args.device)

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else resolve_default_config_path(checkpoint_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}. Pass --config explicitly.")

    images = load_test_images(TEST_JSON)
    idx_to_cat_id = load_idx_to_cat_id(args.idx_to_cat_id, TRAIN_JSON)
    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.sahi:
        run_sahi_codetr(
            checkpoint=str(checkpoint_path),
            config_path=config_path,
            images=images,
            idx_to_cat_id=idx_to_cat_id,
            output_csv=output_csv,
            score_thr=args.score_thr,
            device=args.device,
            slice_size=args.slice_size,
            overlap=args.overlap,
            postprocess_match_threshold=args.postprocess_match_threshold,
        )
    else:
        run_regular_codetr(
            checkpoint=str(checkpoint_path),
            config_path=config_path,
            images=images,
            idx_to_cat_id=idx_to_cat_id,
            output_csv=output_csv,
            score_thr=args.score_thr,
            device=args.device,
        )

    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()

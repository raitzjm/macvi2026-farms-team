#!/usr/bin/env python3
"""Local helpers and CLI for the T-014 YOLO26s k-fold pseudo-label workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _paths import TRAIN_IMGS, TRAIN_JSON  # noqa: E402
from split_manifest import load_manifest  # noqa: E402


PLATEAU_EPOCHS = (35, 40, 45)
RUN_IDS = ("T-014", "T-015", "T-016", "T-017", "T-018")
OUTPUT_TAGS = {
    0: "t014-fold0",
    1: "t015-fold1",
    2: "t016-fold2",
    3: "t017-fold3",
    4: "t018-fold4",
}


def plateau_checkpoint_names() -> list[str]:
    return [f"epoch{epoch}.pt" for epoch in PLATEAU_EPOCHS]


def plateau_checkpoint_paths(root: Path, fold_index: int) -> list[Path]:
    tag = OUTPUT_TAGS[fold_index]
    return [root / tag / "checkpoints" / name for name in plateau_checkpoint_names()]


def _xyxy(box: dict) -> tuple[float, float, float, float]:
    x1 = float(box["bbox_x"])
    y1 = float(box["bbox_y"])
    x2 = x1 + float(box["bbox_width"])
    y2 = y1 + float(box["bbox_height"])
    return x1, y1, x2, y2


def box_iou(box_a: dict, box_b: dict) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(box_a)
    bx1, by1, bx2, by2 = _xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _fuse_cluster(cluster: list[dict]) -> dict:
    total = sum(float(item["score"]) for item in cluster)
    if total <= 0:
        total = float(len(cluster))
    fused = {
        "category_id": int(cluster[0]["category_id"]),
        "bbox_x": sum(float(item["bbox_x"]) * float(item["score"]) for item in cluster) / total,
        "bbox_y": sum(float(item["bbox_y"]) * float(item["score"]) for item in cluster) / total,
        "bbox_width": sum(float(item["bbox_width"]) * float(item["score"]) for item in cluster) / total,
        "bbox_height": sum(float(item["bbox_height"]) * float(item["score"]) for item in cluster) / total,
        "score": sum(float(item["score"]) for item in cluster) / len(cluster),
        "votes": len(cluster),
        "checkpoints": sorted(str(item["checkpoint"]) for item in cluster),
    }
    return fused


def cluster_temporal_candidates(
    predictions_by_checkpoint: dict[str, list[dict]],
    *,
    match_iou: float,
    min_votes: int,
) -> list[dict]:
    flat: list[dict] = []
    for checkpoint, rows in predictions_by_checkpoint.items():
        for row in rows:
            flat.append({**row, "checkpoint": checkpoint})

    flat.sort(key=lambda row: float(row["score"]), reverse=True)
    clusters: list[list[dict]] = []
    for row in flat:
        matched = False
        for cluster in clusters:
            if int(cluster[0]["category_id"]) != int(row["category_id"]):
                continue
            if checkpoint_in_cluster(cluster, str(row["checkpoint"])):
                continue
            if box_iou(cluster[0], row) >= match_iou:
                cluster.append(row)
                matched = True
                break
        if not matched:
            clusters.append([row])

    fused = [_fuse_cluster(cluster) for cluster in clusters if len(cluster) >= min_votes]
    fused.sort(key=lambda row: float(row["score"]), reverse=True)
    return fused


def checkpoint_in_cluster(cluster: list[dict], checkpoint: str) -> bool:
    return any(str(item["checkpoint"]) == checkpoint for item in cluster)


def candidate_band(score: float, *, strong_thr: float, medium_thr: float) -> str | None:
    if score >= strong_thr:
        return "strong"
    if score >= medium_thr:
        return "medium"
    return None


def has_cross_fold_confirmation(candidate: dict, confirm_predictions: list[dict], *, confirm_iou: float, confirm_score: float) -> bool:
    for row in confirm_predictions:
        if int(row["category_id"]) != int(candidate["category_id"]):
            continue
        if float(row["score"]) < confirm_score:
            continue
        if box_iou(candidate, row) >= confirm_iou:
            return True
    return False


def clip_box(box: dict, image_width: int, image_height: int) -> dict | None:
    x1, y1, x2, y2 = _xyxy(box)
    x1 = min(max(0.0, x1), float(image_width))
    y1 = min(max(0.0, y1), float(image_height))
    x2 = min(max(0.0, x2), float(image_width))
    y2 = min(max(0.0, y2), float(image_height))
    w = x2 - x1
    h = y2 - y1
    if w <= 1e-6 or h <= 1e-6:
        return None
    return {
        **box,
        "bbox_x": x1,
        "bbox_y": y1,
        "bbox_width": w,
        "bbox_height": h,
    }


def class_aware_nms(boxes: list[dict], *, iou_thr: float) -> list[dict]:
    kept: list[dict] = []
    for row in sorted(boxes, key=lambda item: float(item["score"]), reverse=True):
        if any(int(prev["category_id"]) == int(row["category_id"]) and box_iou(prev, row) >= iou_thr for prev in kept):
            continue
        kept.append(row)
    return kept


def finalize_image_candidates(
    candidates: list[dict],
    gt_boxes: list[dict],
    *,
    duplicate_iou: float,
    final_iou: float,
    max_per_image: int,
    max_per_category: int,
    image_width: int,
    image_height: int,
) -> list[dict]:
    filtered: list[dict] = []
    for row in candidates:
        clipped = clip_box(row, image_width, image_height)
        if clipped is None:
            continue
        if any(int(gt["category_id"]) == int(clipped["category_id"]) and box_iou(gt, clipped) >= duplicate_iou for gt in gt_boxes):
            continue
        filtered.append(clipped)

    deduped = class_aware_nms(filtered, iou_thr=final_iou)
    kept: list[dict] = []
    per_category: defaultdict[int, int] = defaultdict(int)
    for row in deduped:
        category_id = int(row["category_id"])
        if per_category[category_id] >= max_per_category:
            continue
        if len(kept) >= max_per_image:
            break
        kept.append(row)
        per_category[category_id] += 1
    return kept


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_idx_to_cat_id(path: Path) -> dict[int, int]:
    raw = _load_json(path)
    return {int(key): int(value) for key, value in raw.items()}


def remap_prediction_categories(predictions: dict[int, list[dict]], idx_to_cat_id: dict[int, int]) -> dict[int, list[dict]]:
    remapped: dict[int, list[dict]] = {}
    for image_id, rows in predictions.items():
        remapped[image_id] = []
        for row in rows:
            cls_idx = int(row["category_id"])
            remapped[image_id].append({**row, "category_id": idx_to_cat_id[cls_idx]})
    return remapped


def load_train_context(train_json: Path, manifest_path: Path) -> tuple[dict, dict, dict[int, list[dict]]]:
    coco = _load_json(train_json)
    manifest = load_manifest(manifest_path)
    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_image[int(ann["image_id"])].append(
            {
                "category_id": int(ann["category_id"]),
                "bbox_x": float(ann["bbox"][0]),
                "bbox_y": float(ann["bbox"][1]),
                "bbox_width": float(ann["bbox"][2]),
                "bbox_height": float(ann["bbox"][3]),
                "score": 1.0,
                "checkpoint": "gt",
            }
        )
    return coco, manifest, gt_by_image


def _infer_checkpoint(checkpoint: Path, images: list[dict], *, conf: float, iou: float, imgsz: int, device: str, batch: int) -> dict[int, list[dict]]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    image_paths = [str(TRAIN_IMGS / image["file_name"]) for image in images]
    image_ids = [int(image["id"]) for image in images]
    rows_by_image: dict[int, list[dict]] = {image_id: [] for image_id in image_ids}

    for start in range(0, len(image_paths), batch):
        batch_paths = image_paths[start : start + batch]
        batch_ids = image_ids[start : start + batch]
        results = model.predict(
            source=batch_paths,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            batch=batch,
            verbose=False,
            stream=False,
        )
        for image_id, result in zip(batch_ids, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.tolist()
            scores = result.boxes.conf.tolist()
            classes = result.boxes.cls.tolist()
            for box, score, cls_idx in zip(xyxy, scores, classes):
                x1, y1, x2, y2 = box
                rows_by_image[image_id].append(
                    {
                        "category_id": int(cls_idx),
                        "bbox_x": float(x1),
                        "bbox_y": float(y1),
                        "bbox_width": float(x2 - x1),
                        "bbox_height": float(y2 - y1),
                        "score": float(score),
                        "checkpoint": checkpoint.name,
                    }
                )
    return rows_by_image


def build_oof_pseudo_labels(
    *,
    outputs_root: Path,
    manifest_path: Path,
    train_json: Path,
    output_json: Path,
    raw_dir: Path | None,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    batch: int,
    temporal_iou: float,
    temporal_votes: int,
    strong_thr: float,
    medium_thr: float,
    confirm_iou: float,
    confirm_score: float,
    duplicate_iou: float,
    final_iou: float,
    max_per_image: int,
    max_per_category: int,
) -> dict:
    coco, manifest, gt_by_image = load_train_context(train_json, manifest_path)
    images_by_id = {int(image["id"]): image for image in coco["images"]}
    annotations: list[dict] = []
    stats = {
        "folds": {},
        "accepted_total": 0,
        "image_total": 0,
    }
    next_ann_id = 1

    for fold in manifest["folds"]:
        fold_index = int(fold["fold_index"])
        val_images = [images_by_id[int(image_id)] for image_id in fold["val_ids"]]
        checkpoint_paths = plateau_checkpoint_paths(outputs_root, fold_index)
        idx_to_cat_id = load_idx_to_cat_id(outputs_root / OUTPUT_TAGS[fold_index] / "idx_to_cat_id.json")
        confirm_idx_to_cat_id = {
            other_fold: load_idx_to_cat_id(outputs_root / OUTPUT_TAGS[other_fold] / "idx_to_cat_id.json")
            for other_fold in range(len(RUN_IDS))
            if other_fold != fold_index
        }
        oof_predictions = {
            checkpoint_path.name: remap_prediction_categories(
                _infer_checkpoint(
                    checkpoint_path,
                    val_images,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    batch=batch,
                ),
                idx_to_cat_id,
            )
            for checkpoint_path in checkpoint_paths
        }

        raw_payload = {}
        fold_count = 0
        for image in val_images:
            image_id = int(image["id"])
            prediction_sets = {
                checkpoint_name: rows_by_image[image_id]
                for checkpoint_name, rows_by_image in oof_predictions.items()
            }
            raw_payload[str(image_id)] = prediction_sets
            temporal = cluster_temporal_candidates(
                prediction_sets,
                match_iou=temporal_iou,
                min_votes=temporal_votes,
            )

            strong = [row for row in temporal if candidate_band(float(row["score"]), strong_thr=strong_thr, medium_thr=medium_thr) == "strong"]
            medium = [row for row in temporal if candidate_band(float(row["score"]), strong_thr=strong_thr, medium_thr=medium_thr) == "medium"]

            confirm_rows: list[dict] = []
            if medium:
                confirm_paths = [
                    (
                        other_fold,
                        outputs_root / OUTPUT_TAGS[other_fold] / "checkpoints" / "epoch40.pt",
                    )
                    for other_fold in range(len(RUN_IDS))
                    if other_fold != fold_index
                ]
                confirm_images = [image]
                for other_fold, checkpoint_path in confirm_paths:
                    confirm_map = remap_prediction_categories(
                        _infer_checkpoint(
                            checkpoint_path,
                            confirm_images,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            device=device,
                            batch=1,
                        ),
                        confirm_idx_to_cat_id[other_fold],
                    )
                    confirm_rows.extend(confirm_map[image_id])

            accepted = strong + [
                row
                for row in medium
                if has_cross_fold_confirmation(row, confirm_rows, confirm_iou=confirm_iou, confirm_score=confirm_score)
            ]
            finalized = finalize_image_candidates(
                accepted,
                gt_by_image.get(image_id, []),
                duplicate_iou=duplicate_iou,
                final_iou=final_iou,
                max_per_image=max_per_image,
                max_per_category=max_per_category,
                image_width=int(image["width"]),
                image_height=int(image["height"]),
            )

            for row in finalized:
                annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": image_id,
                        "category_id": int(row["category_id"]),
                        "bbox": [
                            round(float(row["bbox_x"]), 4),
                            round(float(row["bbox_y"]), 4),
                            round(float(row["bbox_width"]), 4),
                            round(float(row["bbox_height"]), 4),
                        ],
                        "area": round(float(row["bbox_width"]) * float(row["bbox_height"]), 4),
                        "iscrowd": 0,
                        "score": round(float(row["score"]), 6),
                        "attributes": {
                            "fold_index": fold_index,
                            "fold_name": fold["fold_name"],
                            "temporal_votes": int(row["votes"]),
                            "temporal_checkpoints": list(row["checkpoints"]),
                        },
                    }
                )
                next_ann_id += 1
                fold_count += 1

        stats["folds"][str(fold_index)] = {
            "fold_name": fold["fold_name"],
            "image_count": len(val_images),
            "accepted_count": fold_count,
        }
        stats["accepted_total"] += fold_count
        stats["image_total"] += len(val_images)
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"fold{fold_index}-oof-raw.json").write_text(json.dumps(raw_payload, indent=2, sort_keys=True) + "\n")

    payload = {
        "images": coco["images"],
        "annotations": annotations,
        "categories": coco["categories"],
        "info": {
            "source": "T-014 YOLO26s k-fold pseudo labels",
            "manifest": str(manifest_path),
            "plateau_checkpoints": plateau_checkpoint_names(),
            "stats": stats,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if raw_dir is not None:
        (raw_dir / "summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="/home/farms/kaggle_outputs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-json", default=str(TRAIN_JSON))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--temporal-iou", type=float, default=0.55)
    parser.add_argument("--temporal-votes", type=int, default=2)
    parser.add_argument("--strong-thr", type=float, default=0.65)
    parser.add_argument("--medium-thr", type=float, default=0.50)
    parser.add_argument("--confirm-iou", type=float, default=0.50)
    parser.add_argument("--confirm-score", type=float, default=0.40)
    parser.add_argument("--duplicate-iou", type=float, default=0.30)
    parser.add_argument("--final-iou", type=float, default=0.55)
    parser.add_argument("--max-per-image", type=int, default=3)
    parser.add_argument("--max-per-category", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_oof_pseudo_labels(
        outputs_root=Path(args.outputs_root),
        manifest_path=Path(args.manifest),
        train_json=Path(args.train_json),
        output_json=Path(args.output_json),
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        temporal_iou=args.temporal_iou,
        temporal_votes=args.temporal_votes,
        strong_thr=args.strong_thr,
        medium_thr=args.medium_thr,
        confirm_iou=args.confirm_iou,
        confirm_score=args.confirm_score,
        duplicate_iou=args.duplicate_iou,
        final_iou=args.final_iou,
        max_per_image=args.max_per_image,
        max_per_category=args.max_per_category,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

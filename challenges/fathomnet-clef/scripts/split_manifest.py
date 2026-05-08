#!/usr/bin/env python3
"""Shared train/val split manifest helpers for FathomNet-CLEF."""

from __future__ import annotations

import json
import random
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_VAL_SPLIT = 0.1
DEFAULT_SEED = 42
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "split-manifests" / "shared-train-val-v1.json"
)


def default_split_manifest_path() -> Path:
    return DEFAULT_MANIFEST_PATH


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text())


def _source_stats(coco: dict) -> dict:
    payload = json.dumps(coco, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "image_count": len(coco.get("images", [])),
        "annotation_count": len(coco.get("annotations", [])),
        "category_count": len(coco.get("categories", [])),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _count_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    return "5+"


def _image_targets(coco: dict) -> list[dict]:
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_img[int(ann["image_id"])].append(ann)

    class_freq = Counter()
    targets = []
    for image in sorted(coco["images"], key=lambda im: int(im["id"])):
        image_id = int(image["id"])
        classes = sorted({int(ann["category_id"]) for ann in anns_by_img.get(image_id, [])})
        ann_count = len(anns_by_img.get(image_id, []))
        for category_id in classes:
            class_freq[category_id] += 1
        targets.append(
            {
                "image_id": image_id,
                "classes": classes,
                "ann_count": ann_count,
                "count_bin": _count_bin(ann_count),
            }
        )
    return targets, class_freq


def build_default_manifest(coco: dict, val_split: float = DEFAULT_VAL_SPLIT, seed: int = DEFAULT_SEED) -> dict:
    img_ids = sorted(int(im["id"]) for im in coco["images"])
    rng = random.Random(seed)
    rng.shuffle(img_ids)
    cut = int(len(img_ids) * (1 - val_split))
    train_ids = sorted(img_ids[:cut])
    val_ids = sorted(img_ids[cut:])
    return {
        "manifest_version": 1,
        "name": "shared-train-val-v1",
        "seed": seed,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "notes": [
            "Shared deterministic split used across frameworks.",
            "This is a canonical shared 90/10 split, not a hand-audited micro-val set.",
        ],
    }


def build_kfold_manifest(coco: dict, k: int = 5, seed: int = DEFAULT_SEED) -> dict:
    if k < 2:
        raise ValueError("k must be at least 2")

    targets, class_freq = _image_targets(coco)
    if len(targets) < k:
        raise ValueError("k cannot exceed the number of images")

    rng = random.Random(seed)
    tie_breakers = {target["image_id"]: rng.random() for target in targets}
    ordered_targets = sorted(
        targets,
        key=lambda target: (
            -len(target["classes"]),
            min((class_freq[category_id] for category_id in target["classes"]), default=0),
            -target["ann_count"],
            tie_breakers[target["image_id"]],
            target["image_id"],
        ),
    )

    folds = [
        {
            "fold_index": fold_index,
            "val_ids": [],
            "class_counts": Counter(),
            "count_bins": Counter(),
        }
        for fold_index in range(k)
    ]

    for target in ordered_targets:
        def fold_score(fold: dict) -> tuple:
            class_pressure = sum(fold["class_counts"][category_id] for category_id in target["classes"])
            return (
                class_pressure,
                fold["count_bins"][target["count_bin"]],
                len(fold["val_ids"]),
                fold["fold_index"],
            )

        chosen = min(folds, key=fold_score)
        chosen["val_ids"].append(target["image_id"])
        chosen["count_bins"][target["count_bin"]] += 1
        for category_id in target["classes"]:
            chosen["class_counts"][category_id] += 1

    all_ids = sorted(target["image_id"] for target in targets)
    manifest_folds = []
    for fold in folds:
        val_ids = sorted(fold["val_ids"])
        val_id_set = set(val_ids)
        manifest_folds.append(
            {
                "fold_index": fold["fold_index"],
                "fold_name": f"fold-{fold['fold_index']}",
                "train_ids": [image_id for image_id in all_ids if image_id not in val_id_set],
                "val_ids": val_ids,
                "val_image_count": len(val_ids),
                "count_bin_summary": dict(sorted(fold["count_bins"].items())),
                "class_presence_summary": dict(sorted(fold["class_counts"].items())),
            }
        )

    return {
        "manifest_version": 1,
        "name": "t014-yolo26s-kfold-v1",
        "k": k,
        "seed": seed,
        "generator": {
            "script": "challenges/fathomnet-clef/scripts/split_manifest.py",
            "algorithm_version": 1,
        },
        "stratification": {
            "class_presence": True,
            "object_count_bins": ["1", "2", "3-4", "5+"],
        },
        "source_stats": _source_stats(coco),
        "image_count": len(all_ids),
        "folds": manifest_folds,
        "notes": [
            "Deterministic k-fold split for T-014.",
            "Validation folds partition the full labeled training image set exactly once.",
        ],
    }


def ensure_manifest(train_json: Path, manifest_path: Path | None = None) -> Path:
    path = manifest_path or default_split_manifest_path()
    if path.exists():
        return path
    coco = load_coco(train_json)
    manifest = build_default_manifest(coco)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text())
    if "folds" in data:
        normalized_folds = []
        seen_val_ids = set()
        for raw_fold in data["folds"]:
            train_ids = [int(i) for i in raw_fold["train_ids"]]
            val_ids = [int(i) for i in raw_fold["val_ids"]]
            overlap = set(train_ids).intersection(val_ids)
            if overlap:
                raise ValueError(f"K-fold manifest has overlapping train/val ids: {sorted(list(overlap))[:5]}")
            val_overlap = seen_val_ids.intersection(val_ids)
            if val_overlap:
                raise ValueError(f"K-fold manifest reuses validation ids: {sorted(list(val_overlap))[:5]}")
            seen_val_ids.update(val_ids)
            fold = dict(raw_fold)
            fold["fold_index"] = int(raw_fold["fold_index"])
            fold["train_ids"] = train_ids
            fold["val_ids"] = val_ids
            normalized_folds.append(fold)
        data["folds"] = sorted(normalized_folds, key=lambda fold: fold["fold_index"])
        data["k"] = int(data.get("k", len(data["folds"])))
        return data

    required = {"train_ids", "val_ids"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Split manifest missing keys: {sorted(missing)}")
    train_ids = [int(i) for i in data["train_ids"]]
    val_ids = [int(i) for i in data["val_ids"]]
    overlap = set(train_ids).intersection(val_ids)
    if overlap:
        raise ValueError(f"Split manifest has overlapping train/val ids: {sorted(list(overlap))[:5]}")
    data["train_ids"] = train_ids
    data["val_ids"] = val_ids
    return data

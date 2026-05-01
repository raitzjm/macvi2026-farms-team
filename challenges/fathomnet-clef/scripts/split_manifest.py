#!/usr/bin/env python3
"""Shared train/val split manifest helpers for FathomNet-CLEF."""

from __future__ import annotations

import json
import random
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

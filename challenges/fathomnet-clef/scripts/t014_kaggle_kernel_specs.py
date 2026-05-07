#!/usr/bin/env python3
"""Fold-specific Kaggle kernel metadata for the T-014 family."""

from __future__ import annotations


OWNER = "raitzjm"
MACHINE_SHAPE = "NvidiaTeslaT4"
RUN_IDS = ["T-014", "T-015", "T-016", "T-017", "T-018"]
SLUG_OVERRIDES = {
    2: "fathomnet-yolo26s-t-016-fold-2a",
}
TITLE_OVERRIDES = {
    2: "FathomNet YOLO26s T-016 Fold 2A",
}


def build_fold_spec(fold_index: int) -> dict:
    if fold_index < 0 or fold_index >= len(RUN_IDS):
        raise ValueError(f"fold_index out of range: {fold_index}")

    run_id = RUN_IDS[fold_index]
    slug_tail = SLUG_OVERRIDES.get(
        fold_index,
        f"fathomnet-yolo26s-{run_id.lower()}-fold-{fold_index}",
    )
    slug = f"{OWNER}/{slug_tail}"
    return {
        "fold_index": fold_index,
        "fold_name": f"fold-{fold_index}",
        "run_id": run_id,
        "slug": slug,
        "title": TITLE_OVERRIDES.get(fold_index, f"FathomNet YOLO26s {run_id} Fold {fold_index}"),
        "machine_shape": MACHINE_SHAPE,
    }


def all_fold_specs() -> list[dict]:
    return [build_fold_spec(fold_index) for fold_index in range(len(RUN_IDS))]

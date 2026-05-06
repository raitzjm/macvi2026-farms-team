#!/usr/bin/env python3
"""Render fold-specific Kaggle kernel directories for the T-014 family."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from t014_kaggle_kernel_specs import all_fold_specs


CHALLENGE_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_ROOT = CHALLENGE_ROOT / "kaggle"
TEMPLATE_DIR = KAGGLE_ROOT / "yolo26s-t014-template"
MANIFEST_PATH = CHALLENGE_ROOT / "artifacts" / "split-manifests" / "t014-yolo26s-kfold-v1.json"


def main():
    manifest_name = MANIFEST_PATH.name
    for spec in all_fold_specs():
        kernel_dir_name = f"yolo26s-t014-fold{spec['fold_index']}"
        kernel_dir = KAGGLE_ROOT / kernel_dir_name
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        shutil.copytree(TEMPLATE_DIR, kernel_dir)
        shutil.copy2(MANIFEST_PATH, kernel_dir / manifest_name)

        fold_config = {
            "run_id": spec["run_id"],
            "fold_index": spec["fold_index"],
            "fold_name": spec["fold_name"],
            "run_name": f"fathomnet-yolo26s-{spec['run_id']}-{spec['fold_name']}",
            "manifest_file": manifest_name,
            "kernel_dir_name": kernel_dir_name,
            "machine_shape": spec["machine_shape"],
        }
        (kernel_dir / "fold-config.json").write_text(json.dumps(fold_config, indent=2, sort_keys=True) + "\n")

        metadata = {
            "id": spec["slug"],
            "title": spec["title"],
            "code_file": "kernel_train.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": spec["machine_shape"],
            "enable_internet": True,
            "dataset_sources": ["cosheimil/fathomnetclef2026"],
            "competition_sources": ["fathomnet-2026"],
            "kernel_sources": [],
        }
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(kernel_dir)


if __name__ == "__main__":
    main()

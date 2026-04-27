"""
Multi-environment data path resolution for FathomNet-CLEF 2026.

Priority:
  1. FATHOMNET_DATA_DIR env var (explicit override, works everywhere)
  2. Kaggle kernel detection (KAGGLE_KERNEL_RUN_TYPE)
  3. TensorDock detection (TENSORDOCK_INSTANCE)
  4. Default local/container path
"""

import os
from pathlib import Path

def _resolve() -> Path:
    if override := os.environ.get("FATHOMNET_DATA_DIR"):
        return Path(override)
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Path("/kaggle/input/fathomnet-2026")
    if os.environ.get("TENSORDOCK_INSTANCE"):
        raise EnvironmentError(
            "TensorDock detected but FATHOMNET_DATA_DIR is not set. "
            "Export FATHOMNET_DATA_DIR before running."
        )
    return Path("/home/farms/data/datasets/fathomnet-2026")

DATASET_ROOT = _resolve()

TRAIN_JSON  = DATASET_ROOT / "train_dataset.json"
TEST_JSON   = DATASET_ROOT / "test_dataset.json"
TRAIN_IMGS  = DATASET_ROOT / "train_set"
TEST_IMGS   = DATASET_ROOT / "test_set"

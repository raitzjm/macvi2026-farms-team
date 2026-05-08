import json
import unittest
from pathlib import Path


KERNEL_DIR = Path(__file__).resolve().parents[1] / "kaggle" / "yolo26s-t019-final"


class TestT019KaggleKernelSpec(unittest.TestCase):
    def test_kernel_metadata_matches_locked_kaggle_policy(self):
        metadata = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())

        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["id"], "raitzjm/fathomnet-yolo26s-t-019-final")
        self.assertIn("cosheimil/fathomnetclef2026", metadata["dataset_sources"])
        self.assertIn("raitzjm/fathomnet-t019-final-retrain-inputs", metadata["dataset_sources"])
        self.assertEqual(metadata["competition_sources"], ["fathomnet-2026"])

    def test_kernel_script_keeps_locked_final_training_policy(self):
        text = (KERNEL_DIR / "kernel_train.py").read_text()

        self.assertIn('RUN_ID = "T-019"', text)
        self.assertIn('batch=16', text)
        self.assertIn('epochs=50', text)
        self.assertIn('imgsz=1024', text)
        self.assertIn('rect=False', text)
        self.assertIn('save_period=5', text)
        self.assertIn('mosaic=0.85', text)
        self.assertIn('close_mosaic=10', text)
        self.assertIn('if WANDB_API_KEY.startswith("wandb_")', text)
        self.assertIn('resolve_uploaded_input("t014-expanded-train.json")', text)
        self.assertIn('resolve_uploaded_input("bundle-summary.json")', text)
        self.assertIn('IMAGE_DATASET_ROOT = Path("/kaggle/input/fathomnetclef2026")', text)
        self.assertIn("def resolve_image_root()", text)
        self.assertIn('validate_image_index', text)
        self.assertIn('bundle-summary.json', text)
        self.assertIn('"kind": "training"', text)
        self.assertNotIn('project=str(WORK / "runs")', text)


if __name__ == "__main__":
    unittest.main()

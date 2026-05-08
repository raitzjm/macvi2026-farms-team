import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "split_manifest.py"
SPEC = importlib.util.spec_from_file_location("split_manifest", MODULE_PATH)
split_manifest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(split_manifest)


class TestSplitManifest(unittest.TestCase):
    def test_build_default_manifest_produces_disjoint_split(self):
        coco = {"images": [{"id": i} for i in range(10)]}
        manifest = split_manifest.build_default_manifest(coco, val_split=0.2, seed=7)
        self.assertEqual(len(manifest["train_ids"]), 8)
        self.assertEqual(len(manifest["val_ids"]), 2)
        self.assertFalse(set(manifest["train_ids"]).intersection(manifest["val_ids"]))

    def test_ensure_manifest_writes_default_file(self):
        with tempfile.TemporaryDirectory() as td:
            train_json = Path(td) / "train.json"
            manifest_path = Path(td) / "split.json"
            train_json.write_text(json.dumps({"images": [{"id": i} for i in range(6)]}))
            path = split_manifest.ensure_manifest(train_json, manifest_path)
            self.assertEqual(path, manifest_path)
            data = json.loads(manifest_path.read_text())
            self.assertIn("train_ids", data)
            self.assertIn("val_ids", data)

    def test_build_kfold_manifest_assigns_each_image_once(self):
        coco = {
            "images": [{"id": i} for i in range(10)],
            "annotations": [
                {"image_id": 0, "category_id": 1},
                {"image_id": 1, "category_id": 1},
                {"image_id": 2, "category_id": 2},
                {"image_id": 3, "category_id": 2},
                {"image_id": 4, "category_id": 3},
                {"image_id": 5, "category_id": 3},
                {"image_id": 6, "category_id": 1},
                {"image_id": 7, "category_id": 2},
                {"image_id": 8, "category_id": 3},
                {"image_id": 9, "category_id": 1},
            ],
        }
        manifest = split_manifest.build_kfold_manifest(coco, k=5, seed=42)
        self.assertEqual(manifest["k"], 5)
        self.assertEqual(len(manifest["folds"]), 5)
        val_ids = []
        for fold in manifest["folds"]:
            self.assertIn("train_ids", fold)
            self.assertIn("val_ids", fold)
            self.assertFalse(set(fold["train_ids"]).intersection(fold["val_ids"]))
            val_ids.extend(fold["val_ids"])
        self.assertEqual(sorted(val_ids), list(range(10)))

    def test_build_kfold_manifest_exposes_fold_train_and_val_ids(self):
        coco = {
            "images": [{"id": i} for i in range(6)],
            "annotations": [
                {"image_id": 0, "category_id": 1},
                {"image_id": 0, "category_id": 2},
                {"image_id": 1, "category_id": 1},
                {"image_id": 2, "category_id": 2},
                {"image_id": 3, "category_id": 3},
                {"image_id": 4, "category_id": 3},
                {"image_id": 5, "category_id": 1},
            ],
        }
        manifest = split_manifest.build_kfold_manifest(coco, k=3, seed=9)
        all_ids = {0, 1, 2, 3, 4, 5}
        for index, fold in enumerate(manifest["folds"]):
            self.assertEqual(fold["fold_index"], index)
            self.assertEqual(set(fold["train_ids"]) | set(fold["val_ids"]), all_ids)
        self.assertEqual(manifest["source_stats"]["image_count"], 6)
        self.assertEqual(manifest["source_stats"]["annotation_count"], 7)

    def test_load_manifest_normalizes_kfold_ids(self):
        payload = {
            "name": "kfold-demo",
            "k": 2,
            "folds": [
                {"fold_index": 0, "train_ids": ["1", "2"], "val_ids": ["3"]},
                {"fold_index": 1, "train_ids": ["3"], "val_ids": ["1", "2"]},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kfold.json"
            path.write_text(json.dumps(payload))
            data = split_manifest.load_manifest(path)
        self.assertEqual(data["folds"][0]["train_ids"], [1, 2])
        self.assertEqual(data["folds"][1]["val_ids"], [1, 2])

    def test_build_kfold_manifest_is_deterministic_for_same_seed(self):
        coco = {
            "images": [{"id": i} for i in range(8)],
            "annotations": [
                {"image_id": 0, "category_id": 1},
                {"image_id": 1, "category_id": 1},
                {"image_id": 2, "category_id": 2},
                {"image_id": 3, "category_id": 2},
                {"image_id": 4, "category_id": 3},
                {"image_id": 5, "category_id": 3},
                {"image_id": 6, "category_id": 1},
                {"image_id": 7, "category_id": 2},
            ],
        }
        first = split_manifest.build_kfold_manifest(coco, k=4, seed=42)
        second = split_manifest.build_kfold_manifest(coco, k=4, seed=42)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

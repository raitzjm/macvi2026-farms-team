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


if __name__ == "__main__":
    unittest.main()

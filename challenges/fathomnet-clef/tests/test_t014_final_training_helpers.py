import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_yolo_t014_final.py"
SPEC = importlib.util.spec_from_file_location("train_yolo_t014_final", MODULE_PATH)
train_yolo_t014_final = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_yolo_t014_final)


class TestT014FinalTrainingHelpers(unittest.TestCase):
    def test_build_full_dataset_manifest_uses_all_images_for_train_and_val(self):
        coco = {
            "images": [{"id": 9}, {"id": 3}, {"id": 7}],
            "annotations": [],
            "categories": [],
        }

        manifest = train_yolo_t014_final.build_full_dataset_manifest(coco)

        self.assertEqual(manifest["train_ids"], [3, 7, 9])
        self.assertEqual(manifest["val_ids"], [3, 7, 9])

    def test_merge_pseudo_annotations_assigns_fresh_ids(self):
        coco = {
            "images": [{"id": 1}],
            "annotations": [
                {"id": 4, "image_id": 1, "category_id": 41, "bbox": [1, 2, 3, 4]},
            ],
            "categories": [{"id": 41, "name": "urchin"}],
        }
        pseudo = {
            "annotations": [
                {"id": 99, "image_id": 1, "category_id": 41, "bbox": [9, 8, 7, 6], "score": 0.73},
            ]
        }

        merged = train_yolo_t014_final.merge_pseudo_annotations(coco, pseudo)

        self.assertEqual(len(merged["annotations"]), 2)
        self.assertEqual(merged["annotations"][0]["id"], 4)
        self.assertEqual(merged["annotations"][1]["id"], 5)
        self.assertEqual(merged["annotations"][1]["score"], 0.73)

    def test_write_manifest_creates_json_file(self):
        manifest = {"train_ids": [1], "val_ids": [1], "name": "t014-expanded"}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            train_yolo_t014_final.write_manifest(path, manifest)
            self.assertIn('"name": "t014-expanded"', path.read_text())


if __name__ == "__main__":
    unittest.main()

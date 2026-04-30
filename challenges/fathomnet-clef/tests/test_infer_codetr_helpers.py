import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "infer_codetr.py"
SPEC = importlib.util.spec_from_file_location("infer_codetr", MODULE_PATH)
infer_codetr = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(infer_codetr)


class TestInferCodetrHelpers(unittest.TestCase):
    def test_rows_from_xywh_maps_category_ids_and_annotation_ids(self):
        rows, next_id = infer_codetr.rows_from_xywh(
            image_id=42,
            boxes_xywh=[(10.0, 11.0, 20.0, 21.0), (1.0, 2.0, 3.0, 4.0)],
            scores=[0.9, 0.7],
            labels=[0, 2],
            idx_to_cat_id={0: 101, 1: 202, 2: 303},
            annotation_id_start=7,
        )

        self.assertEqual(next_id, 9)
        self.assertEqual(
            rows,
            [
                {
                    "annotation_id": 7,
                    "image_id": 42,
                    "category_id": 101,
                    "bbox_x": 10.0,
                    "bbox_y": 11.0,
                    "bbox_width": 20.0,
                    "bbox_height": 21.0,
                    "score": 0.9,
                },
                {
                    "annotation_id": 8,
                    "image_id": 42,
                    "category_id": 303,
                    "bbox_x": 1.0,
                    "bbox_y": 2.0,
                    "bbox_width": 3.0,
                    "bbox_height": 4.0,
                    "score": 0.7,
                },
            ],
        )

    def test_rows_from_xywh_skips_non_positive_area(self):
        rows, next_id = infer_codetr.rows_from_xywh(
            image_id=5,
            boxes_xywh=[(1.0, 2.0, 0.0, 3.0), (4.0, 5.0, 6.0, 0.0), (7.0, 8.0, 9.0, 10.0)],
            scores=[0.1, 0.2, 0.3],
            labels=[0, 0, 0],
            idx_to_cat_id={0: 88},
            annotation_id_start=1,
        )

        self.assertEqual(next_id, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["annotation_id"], 1)
        self.assertEqual(rows[0]["bbox_width"], 9.0)
        self.assertEqual(rows[0]["bbox_height"], 10.0)

    def test_rows_from_xyxy_converts_before_building_rows(self):
        rows, next_id = infer_codetr.rows_from_xyxy(
            image_id=9,
            boxes_xyxy=[(2.0, 3.0, 8.0, 13.0)],
            scores=[0.55],
            labels=[1],
            idx_to_cat_id={1: 909},
            annotation_id_start=4,
        )

        self.assertEqual(next_id, 5)
        self.assertEqual(rows[0]["bbox_x"], 2.0)
        self.assertEqual(rows[0]["bbox_y"], 3.0)
        self.assertEqual(rows[0]["bbox_width"], 6.0)
        self.assertEqual(rows[0]["bbox_height"], 10.0)
        self.assertEqual(rows[0]["category_id"], 909)


if __name__ == "__main__":
    unittest.main()

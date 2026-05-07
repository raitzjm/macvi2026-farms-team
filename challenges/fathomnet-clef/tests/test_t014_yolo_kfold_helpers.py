import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "t014_yolo_kfold.py"
SPEC = importlib.util.spec_from_file_location("t014_yolo_kfold", MODULE_PATH)
t014_yolo_kfold = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(t014_yolo_kfold)


def _pred(category_id: int, x: float, y: float, w: float, h: float, score: float, checkpoint: str):
    return {
        "category_id": category_id,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_width": w,
        "bbox_height": h,
        "score": score,
        "checkpoint": checkpoint,
    }


class TestT014YoloKfoldHelpers(unittest.TestCase):
    def test_plateau_checkpoint_names_are_fixed_late_epochs(self):
        self.assertEqual(
            t014_yolo_kfold.plateau_checkpoint_names(),
            ["epoch35.pt", "epoch40.pt", "epoch45.pt"],
        )

    def test_cluster_temporal_candidates_requires_two_of_three_votes(self):
        predictions = {
            "epoch35.pt": [
                _pred(7, 10, 10, 40, 20, 0.70, "epoch35.pt"),
                _pred(11, 150, 150, 30, 30, 0.51, "epoch35.pt"),
            ],
            "epoch40.pt": [
                _pred(7, 12, 9, 39, 21, 0.66, "epoch40.pt"),
            ],
            "epoch45.pt": [
                _pred(7, 11, 11, 40, 19, 0.61, "epoch45.pt"),
            ],
        }

        candidates = t014_yolo_kfold.cluster_temporal_candidates(predictions, match_iou=0.55, min_votes=2)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["category_id"], 7)
        self.assertEqual(candidate["votes"], 3)
        self.assertAlmostEqual(candidate["score"], (0.70 + 0.66 + 0.61) / 3, places=6)
        self.assertEqual(candidate["checkpoints"], ["epoch35.pt", "epoch40.pt", "epoch45.pt"])

    def test_candidate_band_distinguishes_strong_medium_and_reject(self):
        self.assertEqual(t014_yolo_kfold.candidate_band(0.66, strong_thr=0.65, medium_thr=0.50), "strong")
        self.assertEqual(t014_yolo_kfold.candidate_band(0.50, strong_thr=0.65, medium_thr=0.50), "medium")
        self.assertIsNone(t014_yolo_kfold.candidate_band(0.49, strong_thr=0.65, medium_thr=0.50))

    def test_confirmation_requires_matching_class_score_and_iou(self):
        candidate = _pred(7, 10, 10, 40, 20, 0.57, "fused")
        confirm_predictions = [
            _pred(7, 11, 11, 40, 20, 0.44, "epoch40.pt"),
            _pred(9, 10, 10, 40, 20, 0.91, "epoch40.pt"),
        ]

        self.assertTrue(
            t014_yolo_kfold.has_cross_fold_confirmation(
                candidate,
                confirm_predictions,
                confirm_iou=0.50,
                confirm_score=0.40,
            )
        )
        self.assertFalse(
            t014_yolo_kfold.has_cross_fold_confirmation(
                candidate,
                [_pred(7, 80, 80, 40, 20, 0.44, "epoch40.pt")],
                confirm_iou=0.50,
                confirm_score=0.40,
            )
        )

    def test_finalize_image_candidates_rejects_gt_duplicates_and_caps_per_class(self):
        candidates = [
            _pred(7, 10, 10, 40, 20, 0.72, "fused"),
            _pred(7, 60, 10, 40, 20, 0.68, "fused"),
            _pred(7, 110, 10, 40, 20, 0.66, "fused"),
            _pred(11, 160, 10, 40, 20, 0.64, "fused"),
        ]
        gt_boxes = [
            _pred(7, 8, 8, 44, 24, 1.0, "gt"),
        ]

        finalized = t014_yolo_kfold.finalize_image_candidates(
            candidates,
            gt_boxes,
            duplicate_iou=0.30,
            final_iou=0.55,
            max_per_image=3,
            max_per_category=2,
            image_width=400,
            image_height=300,
        )

        self.assertEqual(len(finalized), 3)
        self.assertEqual([row["category_id"] for row in finalized], [7, 7, 11])
        self.assertGreater(finalized[0]["score"], finalized[1]["score"])


if __name__ == "__main__":
    unittest.main()

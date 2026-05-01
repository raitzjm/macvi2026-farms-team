import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "infer_yolo.py"
SPEC = importlib.util.spec_from_file_location("infer_yolo", MODULE_PATH)
infer_yolo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(infer_yolo)

RUN_TRACKING_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking.py"
RUN_TRACKING_SPEC = importlib.util.spec_from_file_location("run_tracking", RUN_TRACKING_PATH)
run_tracking = importlib.util.module_from_spec(RUN_TRACKING_SPEC)
assert RUN_TRACKING_SPEC.loader is not None
RUN_TRACKING_SPEC.loader.exec_module(run_tracking)


class TestInferYoloHelpers(unittest.TestCase):
    def test_derive_idx_to_cat_id_sorts_by_category_id(self):
        with tempfile.TemporaryDirectory() as td:
            train_json = Path(td) / "train.json"
            train_json.write_text(
                '{"categories": [{"id": 12, "name": "c"}, {"id": 2, "name": "a"}, {"id": 9, "name": "b"}]}'
            )

            mapping = infer_yolo.derive_idx_to_cat_id(train_json)
            self.assertEqual(mapping, {0: 2, 1: 9, 2: 12})

    def test_load_idx_to_cat_id_coerces_string_keys_and_values(self):
        with tempfile.TemporaryDirectory() as td:
            map_json = Path(td) / "map.json"
            map_json.write_text('{"0": "101", "1": "302"}')

            mapping = infer_yolo.load_idx_to_cat_id(str(map_json), Path(td) / "unused.json")
            self.assertEqual(mapping, {0: 101, 1: 302})

    def test_xyxy_to_xywh_converts_coordinates(self):
        out = infer_yolo.xyxy_to_xywh(10.0, 12.0, 50.0, 44.0)
        self.assertEqual(out, (10.0, 12.0, 40.0, 32.0))

    def test_sahi_device_uses_cuda_prefix_for_numeric_gpu(self):
        self.assertEqual(infer_yolo.sahi_device("0"), "cuda:0")
        self.assertEqual(infer_yolo.sahi_device("cuda:1"), "cuda:1")
        self.assertEqual(infer_yolo.sahi_device("cpu"), "cpu")

    def test_write_submission_csv_preserves_header_when_empty(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "submission.csv"

            infer_yolo.write_submission_csv(output, [])

            self.assertEqual(
                output.read_text(),
                "annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_width,bbox_height,score\n",
            )

    def test_chunked_splits_items_by_batch_size(self):
        chunks = list(infer_yolo.chunked([1, 2, 3, 4, 5], 2))
        self.assertEqual(chunks, [[1, 2], [3, 4], [5]])

    def test_should_log_progress_on_interval_and_end(self):
        self.assertFalse(infer_yolo.should_log_progress(1, 10, interval=3))
        self.assertTrue(infer_yolo.should_log_progress(3, 10, interval=3))
        self.assertTrue(infer_yolo.should_log_progress(10, 10, interval=3))

    def test_chunked_pairs_preserves_path_id_alignment(self):
        pairs = list(infer_yolo.chunked_pairs(["a", "b", "c"], [11, 22, 33], 2))
        self.assertEqual(pairs, [(["a", "b"], [11, 22]), (["c"], [33])])

    def test_has_positive_area_rejects_zero_height(self):
        self.assertTrue(infer_yolo.has_positive_area(1.0, 2.0))
        self.assertFalse(infer_yolo.has_positive_area(1.0, 0.0))
        self.assertFalse(infer_yolo.has_positive_area(0.0, 2.0))

    def test_tracked_run_name_appends_run_id_once(self):
        self.assertEqual(
            run_tracking.tracked_run_name("codetr-r50-baseline", "T-004"),
            "codetr-r50-baseline-T-004",
        )
        self.assertEqual(
            run_tracking.tracked_run_name("codetr-r50-baseline-T-004", "T-004"),
            "codetr-r50-baseline-T-004",
        )

    def test_tracked_submission_path_appends_inference_and_source_ids(self):
        output = Path("/tmp/submission.csv")
        self.assertEqual(
            run_tracking.tracked_submission_path(output, "I-009", "T-004").name,
            "submission-I-009-from-T-004.csv",
        )


if __name__ == "__main__":
    unittest.main()

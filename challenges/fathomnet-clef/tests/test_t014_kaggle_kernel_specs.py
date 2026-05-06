import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "t014_kaggle_kernel_specs.py"
SPEC = importlib.util.spec_from_file_location("t014_kaggle_kernel_specs", MODULE_PATH)
t014_specs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(t014_specs)


class TestT014KaggleKernelSpecs(unittest.TestCase):
    def test_build_fold_spec_uses_reserved_ids(self):
        spec = t014_specs.build_fold_spec(0)
        self.assertEqual(spec["run_id"], "T-014")
        self.assertEqual(spec["fold_index"], 0)
        self.assertEqual(spec["slug"], "raitzjm/fathomnet-yolo26s-t-014-fold-0")
        self.assertEqual(spec["machine_shape"], "NvidiaTeslaT4")

    def test_all_fold_specs_cover_five_reserved_runs(self):
        specs = t014_specs.all_fold_specs()
        self.assertEqual(len(specs), 5)
        self.assertEqual([spec["run_id"] for spec in specs], ["T-014", "T-015", "T-016", "T-017", "T-018"])
        self.assertEqual([spec["fold_index"] for spec in specs], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()

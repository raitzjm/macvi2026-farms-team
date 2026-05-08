import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "t019_prepare_kaggle_dataset.py"
SPEC = importlib.util.spec_from_file_location("t019_prepare_kaggle_dataset", MODULE_PATH)
t019_prepare_kaggle_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(t019_prepare_kaggle_dataset)


class TestT019PrepareKaggleDataset(unittest.TestCase):
    def test_build_bundle_copies_expected_files_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pseudo = root / "pseudo.json"
            expanded = root / "expanded-train.json"
            manifest = root / "t014-expanded-manifest.json"
            pseudo.write_text('{"annotations": []}\n')
            expanded.write_text('{"images": []}\n')
            manifest.write_text('{"train_ids": []}\n')

            bundle_dir = root / "bundle"
            summary = t019_prepare_kaggle_dataset.build_bundle(
                bundle_dir=bundle_dir,
                pseudo_label_json=pseudo,
                expanded_coco_json=expanded,
                expanded_manifest_json=manifest,
            )

            self.assertTrue((bundle_dir / "t014-pseudo-labels.json").exists())
            self.assertTrue((bundle_dir / "t014-expanded-train.json").exists())
            self.assertTrue((bundle_dir / "t014-expanded-manifest.json").exists())
            self.assertTrue((bundle_dir / "dataset-metadata.json").exists())
            self.assertTrue((bundle_dir / "README.md").exists())

            metadata = json.loads((bundle_dir / "dataset-metadata.json").read_text())
            self.assertEqual(metadata["id"], "raitzjm/fathomnet-t019-final-retrain-inputs")
            self.assertEqual(metadata["licenses"][0]["name"], "CC0-1.0")

            self.assertEqual(summary["file_count"], 3)
            self.assertEqual(summary["files"]["pseudo_label_json"]["dst"], "t014-pseudo-labels.json")
            self.assertEqual(summary["files"]["expanded_coco_json"]["dst"], "t014-expanded-train.json")
            self.assertEqual(summary["files"]["expanded_manifest_json"]["dst"], "t014-expanded-manifest.json")

    def test_build_bundle_rejects_unexpected_existing_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pseudo = root / "pseudo.json"
            expanded = root / "expanded-train.json"
            manifest = root / "t014-expanded-manifest.json"
            pseudo.write_text('{"annotations": []}\n')
            expanded.write_text('{"images": []}\n')
            manifest.write_text('{"train_ids": []}\n')

            bundle_dir = root / "bundle"
            bundle_dir.mkdir()
            (bundle_dir / "unexpected.bin").write_text("x")

            with self.assertRaisesRegex(RuntimeError, "unexpected files"):
                t019_prepare_kaggle_dataset.build_bundle(
                    bundle_dir=bundle_dir,
                    pseudo_label_json=pseudo,
                    expanded_coco_json=expanded,
                    expanded_manifest_json=manifest,
                )


if __name__ == "__main__":
    unittest.main()

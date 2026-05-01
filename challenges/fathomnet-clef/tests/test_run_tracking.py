import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking.py"
SPEC = importlib.util.spec_from_file_location("run_tracking", MODULE_PATH)
run_tracking = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_tracking)


class TestRunTracking(unittest.TestCase):
    def test_resolve_artifact_dir_uses_default_tree_for_run_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = run_tracking.resolve_artifact_dir(root, "training", "T-009", None)
            self.assertEqual(out, root / "artifacts" / "training-runs" / "T-009")

    def test_resolve_artifact_dir_prefers_explicit_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = run_tracking.resolve_artifact_dir(root, "inference", "I-002", "/tmp/custom-dir")
            self.assertEqual(out, Path("/tmp/custom-dir"))

    def test_write_json_writes_sorted_pretty_payload(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "meta.json"
            run_tracking.write_json(target, {"b": 2, "a": 1})
            self.assertEqual(target.read_text(), '{\n  "a": 1,\n  "b": 2\n}\n')

    def test_tee_output_writes_to_console_and_file(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "stdout.log"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with run_tracking.tee_output(target):
                    print("hello")
                    print("warn", file=__import__("sys").stderr)

            self.assertIn("hello", stdout.getvalue())
            self.assertIn("warn", stderr.getvalue())
            content = target.read_text()
            self.assertIn("hello", content)
            self.assertIn("warn", content)

    def test_git_commit_returns_none_outside_repo(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(run_tracking.git_commit(Path(td)))


if __name__ == "__main__":
    unittest.main()

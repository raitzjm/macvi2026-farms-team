import importlib.util
import os
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train_yolo = load_module("train_yolo", "train_yolo.py")
train_mmdet = load_module("train_mmdet", "train_mmdet.py")


class TestWandbEnvHelpers(unittest.TestCase):
    def test_load_wandb_key_returns_env_value(self):
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "from-env"}, clear=False):
            self.assertEqual(train_yolo.load_wandb_key(), "from-env")

    def test_load_wandb_key_returns_none_when_env_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(train_yolo.load_wandb_key())

    def test_load_wandb_key_ignores_legacy_key_file(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(train_yolo.Path, "exists", return_value=True):
                with mock.patch.object(train_yolo.Path, "read_text", return_value="from-file"):
                    self.assertIsNone(train_yolo.load_wandb_key())

    def test_setup_wandb_enables_ultralytics_settings_and_patches_init(self):
        login_calls = []
        init_calls = []
        fake_settings = {"wandb": False}

        fake_wandb = SimpleNamespace(
            login=lambda **kw: login_calls.append(kw),
            init=lambda *a, **kw: init_calls.append((a, kw)) or "run",
        )

        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "from-env"}, clear=True):
            self.assertTrue(
                train_yolo.setup_wandb(
                    "fathomnet-yolo26l-T-007",
                    True,
                    settings=fake_settings,
                    wandb_module=fake_wandb,
                )
            )
            self.assertEqual(fake_settings["wandb"], True)
            self.assertEqual(os.environ["WANDB_PROJECT"], "fathomnet2026")
            self.assertEqual(os.environ["WANDB_ENTITY"], "little_farms")

        self.assertEqual(login_calls, [{"key": "from-env", "relogin": True}])
        self.assertEqual(len(init_calls), 0)
        self.assertEqual(fake_wandb.init.__name__, "_patched_wandb_init")

    def test_setup_wandb_uses_env_value(self):
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "from-env"}, clear=True):
            self.assertTrue(train_mmdet.setup_wandb(False))
            self.assertNotIn("WANDB_DISABLED", os.environ)

    def test_setup_wandb_disables_when_env_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(train_mmdet.setup_wandb(False))
            self.assertEqual(os.environ["WANDB_DISABLED"], "true")

    def test_setup_wandb_ignores_legacy_key_file(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(train_mmdet.Path, "exists", return_value=True):
                with mock.patch.object(train_mmdet.Path, "read_text", return_value="from-file"):
                    self.assertFalse(train_mmdet.setup_wandb(False))
                    self.assertEqual(os.environ["WANDB_DISABLED"], "true")

    def test_setup_wandb_disables_when_flagged_off(self):
        with mock.patch.dict(os.environ, {"WANDB_API_KEY": "from-env"}, clear=True):
            self.assertFalse(train_mmdet.setup_wandb(True))
            self.assertEqual(os.environ["WANDB_DISABLED"], "true")


if __name__ == "__main__":
    unittest.main()

"""Verify the simple YAML configuration entry point."""

import os
from pathlib import Path
import runpy
import tempfile
import unittest

import yaml


SETTINGS_FILE = Path(__file__).resolve().parents[1] / "app/config/settings.py"
CONFIG_FILE = SETTINGS_FILE.with_name("config.yaml")


class SettingsTests(unittest.TestCase):
    def test_loads_values_from_project_working_directory(self):
        data = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
        data["detection"]["vehicle_model"] = "models/custom.pt"
        data["detection"]["vehicle_confidence"] = 0.42
        data["video"]["source"] = 0
        data["video"]["save_output"] = False
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "app/config"
            config_dir.mkdir(parents=True)
            (config_dir / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
            try:
                os.chdir(root)
                values = runpy.run_path(str(SETTINGS_FILE))
            finally:
                os.chdir(previous)
            self.assertEqual(values["VEHICLE_MODEL_PATH"], "models/custom.pt")
            self.assertEqual(values["VEHICLE_CONFIDENCE"], 0.42)
            self.assertEqual(values["VIDEO_SOURCE"], 0)
            self.assertIs(values["SAVE_OUTPUT"], False)
            self.assertFalse((root / "models").exists())
            self.assertFalse((root / "logs").exists())

    def test_missing_config_raises_file_error(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with self.assertRaises(FileNotFoundError):
                    runpy.run_path(str(SETTINGS_FILE))
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

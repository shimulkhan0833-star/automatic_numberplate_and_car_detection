"""Behavioral tests for configuration validation and environment handling."""

import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

from app.config.settings import PROJECT_ROOT, load_settings
from app.core.exceptions import ConfigurationException


class SettingsTests(unittest.TestCase):
    """Exercise the loader with isolated project roots and real YAML files."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = self.root / "config.yaml"
        self.template = yaml.safe_load(
            (PROJECT_ROOT / "app/config/config.yaml").read_text(encoding="utf-8")
        )
        self.write(self.template)

    def write(self, data: dict) -> None:
        """Write a temporary YAML configuration."""
        self.config.write_text(yaml.safe_dump(data), encoding="utf-8")

    def load(self):
        """Load against the isolated project root."""
        return load_settings("config.yaml", project_root=self.root)

    def test_paths_use_project_root_without_creating_outputs(self) -> None:
        settings = self.load()
        self.assertEqual(settings.logging.log_file, self.root / "logs/anpr.log")
        self.assertEqual(settings.video.source, self.root / "data/videos/input/sample.mp4")
        self.assertFalse(settings.logging.log_file.parent.exists())
        self.assertFalse(settings.detection.plate_model.exists())

    def test_absolute_paths_are_preserved(self) -> None:
        destination = self.root / "elsewhere/output"
        self.template["video"]["output_dir"] = str(destination)
        self.write(self.template)
        self.assertEqual(self.load().video.output_dir, destination)

    def test_missing_and_malformed_files(self) -> None:
        with self.assertRaises(ConfigurationException):
            load_settings("missing.yaml", project_root=self.root)
        for content in ("[broken", "", "[]", "logging: {}"):
            with self.subTest(content=content):
                self.config.write_text(content, encoding="utf-8")
                with self.assertRaises(ConfigurationException):
                    self.load()

    def test_invalid_field_values(self) -> None:
        cases = [
            ("detection", "vehicle_confidence", 1.1),
            ("detection", "iou_threshold", float("nan")),
            ("ocr", "confidence", True),
            ("ocr", "language", ""),
            ("logging", "max_bytes", 0),
            ("logging", "backup_count", True),
            ("logging", "console", "false"),
            ("logging", "level", "INVALID"),
            ("tracking", "tracker", "unknown.yaml"),
            ("matching", "method", "unknown"),
            ("video", "source", -1),
            ("video", "source", False),
            ("database", "path", ""),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value):
                data = deepcopy(self.template)
                data[section][key] = value
                self.write(data)
                with self.assertRaises(ConfigurationException):
                    self.load()

    def test_unknown_keys_are_rejected(self) -> None:
        self.template["logging"]["max_byte"] = 100
        self.write(self.template)
        with self.assertRaises(ConfigurationException):
            self.load()

    def test_camera_and_stream_sources(self) -> None:
        for source, expected in ((0, 0), ("2", 2), ("rtsp://camera/live", "rtsp://camera/live")):
            with self.subTest(source=source):
                self.template["video"]["source"] = source
                self.write(self.template)
                self.assertEqual(self.load().video.source, expected)

    def test_environment_precedence_and_no_global_mutation(self) -> None:
        self.template["video"]["source"] = "${ANPR_TEST_SOURCE}"
        self.write(self.template)
        (self.root / ".env").write_text(
            "ANPR_TEST_SOURCE=rtsp://user:secret@camera/live\n", encoding="utf-8"
        )
        with patch.dict(os.environ, {}, clear=True):
            settings = self.load()
            self.assertEqual(settings.video.source, "rtsp://user:secret@camera/live")
            self.assertNotIn("secret", repr(settings))
            self.assertNotIn("ANPR_TEST_SOURCE", os.environ)
        with patch.dict(os.environ, {"ANPR_TEST_SOURCE": "0"}):
            self.assertEqual(self.load().video.source, 0)

    def test_missing_environment_variable(self) -> None:
        self.template["video"]["source"] = "${ANPR_UNSET_SOURCE}"
        self.write(self.template)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ConfigurationException, "ANPR_UNSET_SOURCE"):
                self.load()


if __name__ == "__main__":
    unittest.main()

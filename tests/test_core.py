"""Checks for logging behavior and shared exception handling."""

import logging
import tempfile
import unittest
from pathlib import Path

from app.core.exceptions import ANPRException, OCRException
from app.core.logger import get_logger, setup_logging


class LoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        logger = get_logger()
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def test_reconfiguration_utf8_and_exception_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs" / "anpr.log"
            root_handlers = list(logging.getLogger().handlers)
            try:
                setup_logging(path, console=False)
                setup_logging(path, console=False)
                logger = get_logger("app.recognition.ocr")
                logger.debug("hidden debug message")
                logger.info("Plate: ঢাকা-1234")
                try:
                    raise OCRException("recognition failed")
                except OCRException:
                    logger.exception("OCR failure")
                content = path.read_text(encoding="utf-8")
                self.assertEqual(content.count("Plate: ঢাকা-1234"), 1)
                self.assertNotIn("hidden debug message", content)
                self.assertIn("Traceback", content)
                self.assertIn("OCRException: recognition failed", content)
                self.assertEqual(logging.getLogger().handlers, root_handlers)
            finally:
                self.tearDown()

    def test_rotation_keeps_bounded_backups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anpr.log"
            try:
                logger = setup_logging(
                    path, max_bytes=150, backup_count=2, console=False
                )
                for index in range(20):
                    logger.info("Frame %s: %s", index, "x" * 80)
                self.assertTrue(Path(f"{path}.1").exists())
                self.assertTrue(Path(f"{path}.2").exists())
                self.assertFalse(Path(f"{path}.3").exists())
                self.assertIn("Frame 19", path.read_text(encoding="utf-8"))
            finally:
                self.tearDown()


class ExceptionTests(unittest.TestCase):
    def test_domain_exception_retains_original_cause(self) -> None:
        original = RuntimeError("engine failed")
        try:
            raise OCRException("Cannot read plate") from original
        except ANPRException as error:
            self.assertIs(error.__cause__, original)
            self.assertEqual(str(error), "Cannot read plate")


if __name__ == "__main__":
    unittest.main()

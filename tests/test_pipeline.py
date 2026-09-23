"""Pipeline orchestration tests without models, cameras, or GUI windows."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.core.exceptions import StorageException, VideoInputException
from app.pipeline.anpr_pipeline import ANPRPipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cv = self.start_patch("app.pipeline.anpr_pipeline.cv2")
        self.start_patch("app.pipeline.anpr_pipeline.setup_logging")
        self.start_patch("app.pipeline.anpr_pipeline.logger")
        self.start_patch("app.pipeline.anpr_pipeline.settings.OUTPUT_DIR", self.temp.name)
        self.start_patch("app.pipeline.anpr_pipeline.settings.DATABASE_PATH",
                         str(Path(self.temp.name) / "anpr.db"))
        self.start_patch("app.pipeline.anpr_pipeline.settings.PLATE_DIR",
                         str(Path(self.temp.name) / "plates"))
        self.capture = self.cv.VideoCapture.return_value
        self.capture.isOpened.return_value = True
        self.capture.get.return_value = 25.0
        self.frame = np.zeros((20, 30, 3), dtype=np.uint8)
        self.capture.read.side_effect = [(True, self.frame), (False, None)]
        self.writer = self.cv.VideoWriter.return_value
        self.writer.isOpened.return_value = True
        self.tracker = Mock()
        tracker_module = ModuleType("app.tracking.vehicle_tracker")
        tracker_module.VehicleTracker = Mock(return_value=self.tracker)
        self.start_patch("sys.modules", {"app.tracking.vehicle_tracker": tracker_module}, dictionary=True)
        self.detector = self.start_patch("app.pipeline.anpr_pipeline.PlateDetector").return_value
        self.matcher = self.start_patch("app.pipeline.anpr_pipeline.PlateMatcher").return_value
        self.start_patch("app.pipeline.anpr_pipeline.OCREngine")
        self.reader = self.start_patch("app.pipeline.anpr_pipeline.PlateReader").return_value
        self.tracker.track.return_value = []
        self.detector.detect.return_value = []
        self.matcher.match.return_value = []
        self.reader.read.return_value = []

    def start_patch(self, target, *args, dictionary=False):
        patcher = patch.dict(target, *args) if dictionary else patch(target, *args)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def test_headless_processing_order_and_cleanup(self):
        calls = Mock()
        calls.attach_mock(self.tracker.track, "track")
        calls.attach_mock(self.detector.detect, "detect")
        calls.attach_mock(self.matcher.match, "match")
        calls.attach_mock(self.reader.read, "read")
        pipeline = ANPRPipeline("0", display=False, save_output=False)
        self.assertEqual(pipeline.run(), 1)
        self.assertEqual([call[0] for call in calls.mock_calls], ["track", "detect", "match", "read"])
        self.cv.VideoCapture.assert_called_once_with(0)
        self.capture.release.assert_called_once()
        self.cv.VideoWriter.assert_not_called()
        self.cv.namedWindow.assert_not_called()
        self.cv.destroyWindow.assert_not_called()

    def test_empty_video_releases_capture(self):
        self.capture.read.side_effect = [(False, None)]
        with self.assertRaises(VideoInputException):
            ANPRPipeline(0, display=False, save_output=False).run()
        self.capture.release.assert_called_once()
        self.detector.detect.assert_not_called()

    def test_unopenable_source_releases_capture(self):
        self.capture.isOpened.return_value = False
        with self.assertRaises(VideoInputException):
            ANPRPipeline(0, display=False, save_output=False).run()
        self.capture.release.assert_called_once()

    def test_missing_local_file(self):
        with self.assertRaises(VideoInputException):
            ANPRPipeline(Path(self.temp.name) / "missing.mp4").run()
        self.cv.VideoCapture.assert_not_called()

    def test_inference_failure_releases_video_and_window(self):
        self.detector.detect.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            ANPRPipeline(0, save_output=True).run()
        self.capture.release.assert_called_once()
        self.writer.release.assert_called_once()
        self.cv.destroyWindow.assert_called_once()

    def test_writer_failure_releases_resources(self):
        self.writer.isOpened.return_value = False
        with self.assertRaises(StorageException):
            ANPRPipeline(0, display=False, save_output=True).run()
        self.capture.release.assert_called_once()
        self.writer.release.assert_called_once()

    def test_saves_after_ocr_and_uses_fps_fallback(self):
        self.capture.get.return_value = float("nan")
        self.tracker.track.return_value = [SimpleNamespace(
            vehicle_id=1, class_id=2, class_name="car", confidence=.9, bbox=[0, 0, 10, 10])]
        self.cv.rectangle.side_effect = lambda frame, *args: frame.fill(255)
        self.reader.read.side_effect = lambda frame, matches: self.assertFalse(frame.any()) or []
        pipeline = ANPRPipeline(0, display=False, save_output=True)
        self.assertEqual(pipeline.run(), 1)
        self.assertTrue(self.frame.any())
        self.writer.write.assert_called_once()
        self.writer.release.assert_called_once()
        self.assertEqual(self.cv.VideoWriter.call_args.args[2:], (30.0, (30, 20)))
        self.assertEqual(pipeline.output_path.parent, Path(self.temp.name))

    def test_recognitions_persist_in_separate_run_sessions(self):
        self.reader.read.return_value = [SimpleNamespace(
            match=SimpleNamespace(vehicle_id=7),
            ocr=SimpleNamespace(text="ABC123", confidence=.9))]
        pipeline = ANPRPipeline(0, display=False, save_output=False)
        sessions = []
        for _ in range(2):
            self.capture.read.side_effect = [(True, self.frame), (True, self.frame), (False, None)]
            self.assertEqual(pipeline.run(), 2)
            sessions.append(pipeline.session_id)
        self.assertNotEqual(*sessions)
        with sqlite3.connect(Path(self.temp.name) / "anpr.db") as connection:
            rows = connection.execute(
                "SELECT session_id, vehicle_id, plate_text FROM plate_recognitions").fetchall()
        connection.close()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row[0] for row in rows}, set(sessions))
        self.assertTrue(all(row[1:] == (7, "ABC123") for row in rows))

    def test_vehicle_without_readable_plate_is_saved(self):
        self.tracker.track.return_value = [SimpleNamespace(
            vehicle_id=9, class_id=2, class_name="car")]
        ANPRPipeline(0, display=False, save_output=False).run()
        connection = sqlite3.connect(Path(self.temp.name) / "anpr.db")
        try:
            row = connection.execute(
                "SELECT vehicle_id, class_name, plate_text, confidence FROM plate_recognitions"
            ).fetchone()
            self.assertEqual(row, (9, "car", None, None))
        finally:
            connection.close()

    def test_q_stops_preview(self):
        self.cv.waitKey.return_value = ord("q")
        self.assertEqual(ANPRPipeline(0, save_output=False).run(), 1)
        self.capture.read.assert_called_once()
        self.capture.release.assert_called_once()
        self.cv.destroyWindow.assert_called_once()


if __name__ == "__main__":
    unittest.main()

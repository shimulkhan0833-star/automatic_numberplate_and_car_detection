"""Run tracking, plate detection, matching, and OCR for one video source."""
from math import isfinite
from pathlib import Path
from uuid import uuid4
from time import monotonic

import cv2

from app.config import settings
from app.core.exceptions import StorageException, VideoInputException
from app.core.logger import get_logger, setup_logging
from app.database.database import connect_database
from app.database.repository import PlateRepository
from app.detection.plate_detector import PlateDetector
from app.matching.matcher import PlateMatcher
from app.recognition.ocr_engine import OCREngine
from app.recognition.plate_reader import PlateReader

logger = get_logger(__name__)


class ANPRPipeline:
    """Process a video or camera; create fresh tracking/OCR state for each run.

    Preview is optional. Annotated video saving defaults to SAVE_OUTPUT.
    Tracked road vehicles and recognized plates are saved to SQLite; fresh best-reading crops are saved alongside their records.
    """

    def __init__(self, source: str | int | Path | None = None, *,
                 display: bool = True, save_output: bool | None = None) -> None:
        self.source = settings.VIDEO_SOURCE if source is None else source
        self.display = display
        self.save_output = settings.SAVE_OUTPUT if save_output is None else save_output
        self.output_path: Path | None = None
        self.session_id: str | None = None

    def _source(self) -> str | int:
        """Convert numeric strings to camera indices and check local files."""
        source = self.source
        if isinstance(source, str) and source.isdecimal():
            source = int(source)
        if type(source) is int and source >= 0:
            return source
        if isinstance(source, str) and source.lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        ):
            return source
        if not isinstance(source, (str, Path)) or not str(source).strip():
            raise VideoInputException("Expected a video path, URL, or nonnegative camera index")
        path = Path(source).expanduser()
        if not path.is_file():
            raise VideoInputException(f"Video file not found: {path}")
        return str(path)

    def run(self) -> int:
        """Return processed frame count; release video resources on every exit."""
        setup_logging(settings.LOG_FILE, level=settings.LOG_LEVEL,
                      max_bytes=settings.LOG_MAX_BYTES,
                      backup_count=settings.LOG_BACKUP_COUNT,
                      console=settings.LOG_CONSOLE)
        capture = None
        writer = None
        database = None
        window_open = False
        window = "ANPR - Q to quit"
        frame_count = 0
        self.output_path = None
        self.session_id = uuid4().hex
        try:
            source = self._source()
            live_source = isinstance(source, int) or str(source).lower().startswith(
                ("rtsp://", "rtsps://", "http://", "https://"))
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise VideoInputException("Cannot open configured video")
            success, frame = capture.read()
            if not success or frame is None:
                raise VideoInputException("Video did not provide a readable frame")

            # Import and initialize inference dependencies only when running.
            from app.tracking.vehicle_tracker import VehicleTracker

            tracker = VehicleTracker()
            detector = PlateDetector()
            matcher = PlateMatcher()
            reader = PlateReader(OCREngine(),
                                 retry_frames=settings.OCR_RETRY_FRAMES,
                                 refresh_frames=settings.OCR_REFRESH_FRAMES,
                                 cache_ttl_frames=settings.OCR_CACHE_TTL_FRAMES)
            database = connect_database(settings.DATABASE_PATH)
            repository = PlateRepository(database, settings.PLATE_DIR)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not isfinite(fps) or fps <= 0:
                fps = 30.0
            if self.save_output:
                output_dir = Path(settings.OUTPUT_DIR)
                output_dir.mkdir(parents=True, exist_ok=True)
                self.output_path = output_dir / f"anpr_{uuid4().hex}.mp4"
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(self.output_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise StorageException("Cannot open annotated video output")
            if self.display:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                window_open = True
            logger.info("ANPR processing started")
            started = monotonic()
            while success and frame is not None:
                timestamp = monotonic() - started if live_source else frame_count / fps
                # Complete inference on clean pixels before drawing labels.
                vehicles = tracker.track(frame)
                plates = detector.detect(frame)
                matches = matcher.match(vehicles, plates)
                readings = reader.read(frame, matches)
                repository.save_readings(self.session_id, readings, timestamp, vehicles=vehicles)
                if self.display or writer is not None:
                    self._annotate(frame, vehicles, readings)
                if writer is not None:
                    if frame.shape[:2] != (height, width):
                        raise StorageException("Video frame dimensions changed during recording")
                    writer.write(frame)
                frame_count += 1
                if self.display:
                    cv2.imshow(window, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        break
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                success, frame = capture.read()
            logger.info("Processed %d frames", frame_count)
            if self.output_path is not None:
                logger.info("Annotated video: %s", self.output_path)
            return frame_count
        except Exception:
            logger.exception("ANPR pipeline failed")
            raise
        finally:
            if database is not None:
                database.close()
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            if window_open:
                cv2.destroyWindow(window)

    @staticmethod
    def _annotate(frame, vehicles, readings) -> None:
        """Draw vehicle IDs and associated plate readings onto the frame."""
        for vehicle in vehicles:
            x1, y1, x2, y2 = map(int, vehicle.bbox)
            label = f"ID {vehicle.vehicle_id} | {vehicle.class_name} {vehicle.confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for reading in readings:
            match = reading.match
            plate = match.plate
            x1, y1, x2, y2 = map(int, plate.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            owner = f"ID {match.vehicle_id}" if match.vehicle_id is not None else "unmatched"
            text = (f"{reading.ocr.text} ({reading.ocr.confidence:.2f})"
                    if reading.ocr is not None else "unreadable")
            cv2.putText(frame, f"Plate {owner} | {text}",
                        (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

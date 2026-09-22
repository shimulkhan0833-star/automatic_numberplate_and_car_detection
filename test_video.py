"""Preview configured person and vehicle detections; press Q to exit."""

from dataclasses import asdict
from pathlib import Path

import cv2

from app.config.settings import load_settings
from app.core.exceptions import VideoInputException
from app.core.logger import get_logger, setup_logging
from app.detection.vehicle_detector import VehicleDetector


def main() -> None:
    """Display detections on video frames without saving or tracking them."""
    settings = load_settings()
    setup_logging(**asdict(settings.logging))
    logger = get_logger(__name__)
    capture = None

    try:
        source = settings.video.source
        if isinstance(source, Path) and not source.is_file():
            raise VideoInputException(f"Video file not found: {source}")

        detector = VehicleDetector(settings.detection)
        capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not capture.isOpened():
            raise VideoInputException("Cannot open the configured video source")

        frame_count = 0
        logger.info("Detection preview started. Press Q in the preview window to quit.")
        while True:
            success, frame = capture.read()
            if not success:
                if frame_count == 0:
                    raise VideoInputException("Video source did not provide a readable frame")
                logger.info("Video source ended or stopped providing frames")
                break

            for detection in detector.detect(frame):
                x1, y1, x2, y2 = map(int, detection.bbox)
                label = f"{detection.class_name} {detection.confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            frame_count += 1
            cv2.imshow("Detection preview - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        logger.info("Preview finished after %d frames", frame_count)
    except Exception:
        logger.exception("Video preview failed")
        raise
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

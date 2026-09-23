"""License plate detection using the configured custom YOLO weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.exceptions import DetectionException, ModelException, ModelNotFoundException
from app.core.logger import get_logger

from app.config import settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class PlateDetection:
    """A plate box in input-frame xyxy pixels and its detection confidence."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class PlateDetector:
    """Detect plates without cropping, vehicle association, OCR, or storage.

    Construct with ``PlateDetector()``. The configured
    model must be a single-class license plate detection model, like best.pt.
    """

    def __init__(self) -> None:
        """Load local plate weights once and check the model's class layout."""
        path = Path(settings.PLATE_MODEL_PATH)
        if not path.is_file():
            raise ModelNotFoundException(f"Plate model file not found: {path}")
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(path), task="detect")
            if self._model.task != "detect" or len(self._model.names) != 1:
                raise ValueError("Expected a single-class license plate detection model")
            self._class_name = self._model.names[0]
        except Exception as exc:
            raise ModelException(f"Cannot initialize plate detector from {path}") from exc
        logger.info("Plate detector loaded: %s (device=%s)", path, settings.DEVICE)

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        """Return plate detections for a nonempty HWC BGR uint8 frame.

        Coordinates refer to the supplied frame. No detections returns an empty
        list; invalid input or inference failures raise DetectionException.
        """
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
            or frame.dtype != np.uint8
        ):
            raise DetectionException("Expected a nonempty H x W x 3 BGR uint8 frame")

        try:
            results = self._model.predict(
                source=frame,
                conf=settings.PLATE_CONFIDENCE,
                iou=settings.IOU_THRESHOLD,
                device=settings.DEVICE,
                classes=[0],
                verbose=False,
                save=False,
                stream=False,
            )
            if len(results) != 1 or results[0].boxes is None:
                raise ValueError("Expected one plate detection result")
            boxes = results[0].boxes
            detections = []
            for bbox, confidence in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), strict=True
            ):
                x1, y1, x2, y2 = bbox
                detections.append(PlateDetection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(confidence),
                    class_id=0,
                    class_name=self._class_name,
                ))
        except Exception as exc:
            raise DetectionException("Plate detection failed for the frame") from exc

        logger.debug("Detected %d license plates", len(detections))
        return detections

"""Frame-based YOLO detection for people and road vehicles.

Usage::

    detector = VehicleDetector()
    detections = detector.detect(frame)  # OpenCV BGR uint8 frame

This module does not read videos, track objects, run OCR, or save results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.exceptions import (
    DetectionException,
    ModelException,
    ModelNotFoundException,
)
from app.core.logger import get_logger

from app.config import settings


_CLASS_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
logger = get_logger(__name__)


@dataclass(frozen=True)
class Detection:
    """One detection with original-frame pixel coordinates in xyxy order."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """Load a COCO detection model once and detect the five selected classes.

    Model weights configured in settings.py must
    already exist locally. Use a separate instance for each inference worker.
    People are returned with their own class ID so later plate association can
    exclude them. Failures retain their cause for logging by the caller.
    """

    def __init__(self) -> None:
        """Load local weights and verify the expected COCO class mapping."""
        model_path = Path(settings.VEHICLE_MODEL_PATH)
        if not model_path.is_file():
            raise ModelNotFoundException(f"Vehicle model file not found: {model_path}")

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(model_path), task="detect")
            if self._model.task != "detect":
                raise ValueError("Expected an object detection model")
            names = self._model.names
            if any(names[class_id] != name for class_id, name in _CLASS_NAMES.items()):
                raise ValueError("Model must use the expected COCO person and vehicle class IDs")
        except Exception as exc:
            raise ModelException(f"Cannot initialize vehicle detector from {model_path}") from exc

        logger.info("Vehicle detector loaded: %s (device=%s)", model_path, settings.DEVICE)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in one nonempty HWC BGR uint8 frame.

        Return an empty list when no selected objects are found. Coordinates
        remain floating-point pixels; no tracking IDs are assigned here.
        Invalid frames or inference failures raise DetectionException.
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
                conf=settings.VEHICLE_CONFIDENCE,
                iou=settings.IOU_THRESHOLD,
                device=settings.DEVICE,
                classes=list(_CLASS_NAMES),
                verbose=False,
                save=False,
                stream=False,
            )
            if len(results) != 1 or results[0].boxes is None:
                raise ValueError("Expected one object detection result for the frame")

            boxes = results[0].boxes
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            detections = []
            for bbox, confidence, raw_class_id in zip(
                coordinates, confidences, class_ids, strict=True
            ):
                class_id = int(raw_class_id)
                if class_id not in _CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = bbox
                detections.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=_CLASS_NAMES[class_id],
                ))
        except Exception as exc:
            raise DetectionException("Vehicle detection failed for the frame") from exc

        logger.debug("Detected %d people and vehicles", len(detections))
        return detections

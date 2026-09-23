"""Stateful YOLO tracking using the application's shared configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ultralytics import YOLO

from app.config import settings
from app.core.exceptions import ModelException, TrackingException
from app.core.logger import get_logger

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

logger = get_logger(__name__)


class ByteTracker:
    """Track sequential frames from one video with the configured tracker.

    Create a separate instance for each video to keep tracking state isolated.
    Configure application logging in the entry point before constructing it.
    """

    def __init__(self) -> None:
        """Initialize the vehicle model configured in settings.py."""
        try:
            self.model = YOLO(str(settings.VEHICLE_MODEL_PATH))
        except Exception as exc:
            raise ModelException("Could not load vehicle tracking model") from exc
        logger.info("Vehicle tracking model loaded successfully")

    def update(self, frame: np.ndarray) -> Results:
        """Return an Ultralytics result with boxes and available track IDs."""
        try:
            results = self.model.track(
                source=frame,
                tracker=settings.TRACKER,
                conf=settings.VEHICLE_CONFIDENCE,
                iou=settings.IOU_THRESHOLD,
                device=settings.DEVICE,
                persist=True,
                classes=[0, 2, 3, 5, 7],
                save=False,
                verbose=False,
            )
            return results[0]
        except Exception as exc:
            raise TrackingException("Vehicle tracking failed for the frame") from exc

"""Crop matched plates from clean frames and preserve vehicle association."""
from __future__ import annotations
from dataclasses import dataclass, field
from math import ceil, floor, isfinite
from typing import TYPE_CHECKING
import numpy as np
from app.core.exceptions import OCRException
from app.core.logger import get_logger
from app.recognition.ocr_engine import OCREngine, OCRResult
if TYPE_CHECKING:
    from app.matching.matcher import PlateMatch
logger = get_logger(__name__)

@dataclass(frozen=True)
class PlateReading:
    """A current-frame association and optional recognized plate text."""
    match: PlateMatch
    ocr: OCRResult | None
    crop: np.ndarray | None = field(default=None, repr=False, compare=False)


def crop_plate(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Clip xyxy coordinates to the frame and return an independent crop."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8:
        return None
    if len(bbox) != 4 or not all(isfinite(value) for value in bbox):
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    height, width = frame.shape[:2]
    left, top = max(0, floor(x1)), max(0, floor(y1))
    right, bottom = min(width, ceil(x2)), min(height, ceil(y2))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right].copy()


class PlateReader:
    """Read matched plates; a per-crop OCR failure does not stop the video."""
    def __init__(self, engine: OCREngine, *, retry_frames: int = 15,
                 refresh_frames: int = 150, cache_ttl_frames: int = 300) -> None:
        if any(type(v) is not int or v < 1 for v in
               (retry_frames, refresh_frames, cache_ttl_frames)):
            raise ValueError('OCR frame intervals must be positive integers')
        self.engine = engine
        self.retry_frames = retry_frames
        self.refresh_frames = refresh_frames
        self.cache_ttl_frames = cache_ttl_frames
        self._frame = 0
        self._cache: dict[int, tuple[int, int, OCRResult | None]] = {}

    def read(self, frame: np.ndarray, matches: list[PlateMatch]) -> list[PlateReading]:
        """Keep input order, with None OCR for unmatched or unreadable plates."""
        # Use a new reader per video. Expire IDs not matched recently.
        self._frame += 1
        self._cache = {key: entry for key, entry in self._cache.items()
                       if self._frame - entry[1] < self.cache_ttl_frames}
        readings = []
        for match in matches:
            result = None
            recognized_crop = None
            if match.vehicle_id is not None:
                last_attempt, _, result = self._cache.get(
                    match.vehicle_id, (-self.refresh_frames - self.retry_frames, self._frame, None))
                interval = self.refresh_frames if result is not None else self.retry_frames
                if self._frame - last_attempt < interval:
                    self._cache[match.vehicle_id] = (last_attempt, self._frame, result)
                    readings.append(PlateReading(match, result))
                    continue
                crop = crop_plate(frame, match.plate.bbox)
                if crop is not None:
                    try:
                        # Refresh rather than permanently locking in an early misread.
                        result = self.engine.recognize(crop)
                        if result is not None:
                            recognized_crop = crop
                    except OCRException:
                        logger.exception('OCR failed for vehicle ID %s', match.vehicle_id)
                self._cache[match.vehicle_id] = (self._frame, self._frame, result)
            readings.append(PlateReading(match, result, recognized_crop))
        return readings

"""Associate plates with tracked vehicles using geometry within one frame."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Sequence

from app.config import settings
from app.core.exceptions import MatchingException

if TYPE_CHECKING:
    from app.detection.plate_detector import PlateDetection
    from app.tracking.vehicle_tracker import TrackedVehicle


@dataclass(frozen=True)
class PlateMatch:
    """One input plate and its optional associated vehicle ID."""
    plate: PlateDetection
    vehicle_id: int | None


def _box(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate finite, positive-area xyxy coordinates."""
    try:
        x1, y1, x2, y2 = map(float, values)
        if not all(isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid box')
        return x1, y1, x2, y2
    except (TypeError, ValueError) as exc:
        raise MatchingException('Expected a finite positive-area xyxy box') from exc


class PlateMatcher:
    """Match every frame independently, including plates that appear later.

    Center matching selects the smallest vehicle containing the plate center.
    IoU matching selects the largest IoU above the configured threshold. Ties
    use the lowest vehicle ID. These are geometric heuristics: overlapping
    vehicles can still cause incorrect associations. Multiple plate detections
    may belong to the same vehicle; no historical association is retained.
    """
    def __init__(self, *, method: str = settings.MATCHING_METHOD,
                 iou_threshold: float = settings.MATCHING_IOU_THRESHOLD) -> None:
        if method not in ('center', 'iou'):
            raise MatchingException('Matching method must be center or iou')
        if not 0 <= iou_threshold <= 1:
            raise MatchingException('IoU threshold must be between 0 and 1')
        self.method = method
        self.iou_threshold = iou_threshold

    def match(self, vehicles: Sequence[TrackedVehicle], plates: Sequence[PlateDetection]) -> list[PlateMatch]:
        """Return one match per plate, preserving order and unmatched plates."""
        candidates = [(vehicle, _box(vehicle.bbox)) for vehicle in vehicles
                      if vehicle.class_id in (2, 3, 5, 7)]
        matches = []
        for plate in plates:
            px1, py1, px2, py2 = _box(plate.bbox)
            cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
            ranked = []
            for vehicle, (x1, y1, x2, y2) in candidates:
                area = (x2 - x1) * (y2 - y1)
                if self.method == 'center':
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        ranked.append((area, vehicle.vehicle_id))
                else:
                    intersection = max(0., min(px2, x2) - max(px1, x1)) * max(0., min(py2, y2) - max(py1, y1))
                    union = area + (px2 - px1) * (py2 - py1) - intersection
                    iou = intersection / union
                    if iou > 0 and iou >= self.iou_threshold:
                        ranked.append((-iou, vehicle.vehicle_id))
            matches.append(PlateMatch(plate, min(ranked)[1] if ranked else None))
        return matches

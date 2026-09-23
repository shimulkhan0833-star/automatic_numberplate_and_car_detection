"""Convert tracking results to records usable by the ANPR pipeline."""
from dataclasses import dataclass
import numpy as np
from app.tracking.tracker import ByteTracker
from app.core.logger import get_logger

logger = get_logger(__name__)

@dataclass(frozen=True)
class TrackedVehicle:
    """An identified person or vehicle in original-frame pixel coordinates."""
    vehicle_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]

class VehicleTracker:
    """Maintain one tracker per video; omit boxes without assigned IDs."""
    def __init__(self) -> None:
        self.tracker = ByteTracker()
        logger.info("Vehicle tracker initialized")

    def track(self, frame: np.ndarray) -> list[TrackedVehicle]:
        """Return identified people and vehicles for one sequential frame."""
        result = self.tracker.update(frame)
        if result.boxes is None or result.boxes.id is None:
            return []
        boxes = result.boxes.cpu()
        return [
            TrackedVehicle(int(track_id), int(class_id), result.names[int(class_id)],
                           float(confidence), [float(value) for value in bbox])
            for track_id, class_id, confidence, bbox in zip(
                boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist(),
                boxes.xyxy.tolist(), strict=True
            )
        ]

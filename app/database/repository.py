"""Persist the best plate reading for each tracked vehicle in a session."""
import sqlite3
import re
from pathlib import Path
from uuid import uuid4

import cv2
from datetime import datetime, timezone
from math import isfinite

from app.core.exceptions import DatabaseException, StorageException


class PlateRepository:
    """The caller owns the connection; each frame is saved atomically."""

    def __init__(self, connection: sqlite3.Connection, plate_dir: str | Path | None = None) -> None:
        self.connection = connection
        self.plate_dir = Path(plate_dir) if plate_dir is not None else None

    def save_readings(self, session_id: str, readings, video_timestamp: float, *, vehicles=()) -> None:
        """Save all road vehicles and retain their highest-confidence plate text.

        first_seen/last_seen are UTC processing times of vehicle observations.
        video_timestamp locates the best reading, not the latest observation.
        Equal confidence retains the earlier text and timestamp.
        """
        if not session_id or not isfinite(video_timestamp) or video_timestamp < 0:
            raise DatabaseException("Invalid recognition session or video timestamp")
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for reading in readings:
            if reading.match.vehicle_id is None or reading.ocr is None:
                continue
            text = reading.ocr.text.strip()
            confidence = reading.ocr.confidence
            if not text or not isfinite(confidence) or not 0 <= confidence <= 1:
                continue
            rows.append((session_id, reading.match.vehicle_id, text, confidence,
                         video_timestamp, now, now, getattr(reading, "crop", None)))
        vehicle_rows = [
            (session_id, vehicle.vehicle_id, vehicle.class_id, vehicle.class_name, now, now)
            for vehicle in vehicles if vehicle.class_id in (2, 3, 5, 7)
        ]
        if not rows and not vehicle_rows:
            return
        created = []
        try:
            with self.connection:
                self.connection.executemany("""
                    INSERT INTO plate_recognitions
                        (session_id, vehicle_id, class_id, class_name, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, vehicle_id) DO UPDATE SET
                        class_id = excluded.class_id,
                        class_name = excluded.class_name,
                        last_seen = excluded.last_seen
                """, vehicle_rows)
                for row in rows:
                    session, vehicle_id, text, score, timestamp, first, last, crop = row
                    previous = self.connection.execute(
                        "SELECT confidence FROM plate_recognitions WHERE session_id = ? AND vehicle_id = ?",
                        (session, vehicle_id)).fetchone()
                    better = previous is None or previous[0] is None or score > previous[0]
                    image_path = None
                    if better and crop is not None and self.plate_dir is not None:
                        image_path = self._save_crop(session, vehicle_id, crop)
                        created.append(image_path)
                    self.connection.execute("""
                        INSERT INTO plate_recognitions
                            (session_id, vehicle_id, plate_text, confidence,
                             video_timestamp, first_seen, last_seen, image_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id, vehicle_id) DO UPDATE SET
                            plate_text = CASE WHEN (confidence IS NULL OR excluded.confidence > confidence)
                                THEN excluded.plate_text ELSE plate_text END,
                            video_timestamp = CASE WHEN (confidence IS NULL OR excluded.confidence > confidence)
                                THEN excluded.video_timestamp ELSE video_timestamp END,
                            image_path = CASE WHEN (confidence IS NULL OR excluded.confidence > confidence)
                                THEN excluded.image_path ELSE image_path END,
                            confidence = MAX(COALESCE(confidence, 0), excluded.confidence),
                            last_seen = excluded.last_seen
                    """, (session, vehicle_id, text, score, timestamp, first, last, image_path))
        except Exception as exc:
            for path in created:
                Path(path).unlink(missing_ok=True)
            if isinstance(exc, sqlite3.Error):
                raise DatabaseException("Cannot save plate recognition results") from exc
            raise

    def _save_crop(self, session_id, vehicle_id, crop) -> str:
        """Write a unique PNG before committing its database reference."""
        path = None
        try:
            self.plate_dir.mkdir(parents=True, exist_ok=True)
            session = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)
            path = self.plate_dir / f"{session}_{int(vehicle_id)}_{uuid4().hex}.png"
            success, encoded = cv2.imencode(".png", crop)
            if not success:
                raise StorageException("Cannot encode plate crop")
            path.write_bytes(encoded.tobytes())
            return str(path)
        except (OSError, cv2.error) as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            raise StorageException("Cannot save plate crop") from exc

"""SQLite connection and recognition schema."""
import sqlite3
from pathlib import Path

from app.core.exceptions import DatabaseException


def connect_database(path: str | Path) -> sqlite3.Connection:
    """Create parent directories and initialize a database connection."""
    connection = None
    try:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        with connection:
            connection.execute("BEGIN")
            columns = connection.execute("PRAGMA table_info(plate_recognitions)").fetchall()
            legacy = bool(columns) and "class_id" not in {row["name"] for row in columns}
            if legacy:
                connection.execute("ALTER TABLE plate_recognitions RENAME TO plate_recognitions_legacy")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS plate_recognitions (
                    session_id TEXT NOT NULL,
                    vehicle_id INTEGER NOT NULL,
                    class_id INTEGER,
                    class_name TEXT,
                    plate_text TEXT,
                    image_path TEXT,
                    confidence REAL CHECK(confidence BETWEEN 0 AND 1),
                    video_timestamp REAL CHECK(video_timestamp >= 0),
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    PRIMARY KEY (session_id, vehicle_id)
                )
            """)
            if legacy:
                connection.execute("""
                    INSERT INTO plate_recognitions
                        (session_id, vehicle_id, plate_text, confidence,
                         video_timestamp, first_seen, last_seen)
                    SELECT session_id, vehicle_id, plate_text, confidence,
                           video_timestamp, first_seen, last_seen
                    FROM plate_recognitions_legacy
                """)
                connection.execute("DROP TABLE plate_recognitions_legacy")
            current = connection.execute("PRAGMA table_info(plate_recognitions)").fetchall()
            if "image_path" not in {row["name"] for row in current}:
                connection.execute("ALTER TABLE plate_recognitions ADD COLUMN image_path TEXT")
        return connection
    except (OSError, sqlite3.Error) as exc:
        if connection is not None:
            connection.close()
        raise DatabaseException("Cannot initialize recognition database") from exc

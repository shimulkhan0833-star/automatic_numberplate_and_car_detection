"""Read-only ANPR API. Run from the project root with:

    python -m uvicorn app.api.app:app --reload

All routes live here for now. This module does not load models or run videos.
"""
from contextlib import asynccontextmanager, contextmanager
from datetime import date
from pathlib import Path
import sqlite3

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import settings
from app.core.logger import get_logger
from app.database.database import connect_database

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare/migrate the SQLite schema once when the server starts."""
    connection = connect_database(settings.DATABASE_PATH)
    connection.close()
    yield


app = FastAPI(
    title="ANPR API",
    description="Browse saved vehicles, processing sessions, and plate images.",
    version="0.1.0",
    lifespan=lifespan,
)


@contextmanager
def database():
    """Use a separate read-only connection per request, always closing it.

    Opening inside each synchronous route avoids sharing SQLite connections
    between request threads. SQL parameters keep user input separate from SQL.
    """
    connection = None
    try:
        uri = Path(settings.DATABASE_PATH).resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        yield connection
    except sqlite3.Error as exc:
        logger.exception("API database query failed")
        raise HTTPException(503, "Database is temporarily unavailable") from exc
    finally:
        if connection is not None:
            connection.close()


@app.get("/health", tags=["Health"])
def health() -> dict:
    """Check that the server can read the application database."""
    with database() as connection:
        connection.execute("SELECT session_id FROM plate_recognitions LIMIT 1").fetchone()
    return {"status": "ok", "database": "ok"}


@app.get("/vehicles", tags=["Vehicles"])
def list_vehicles(
    session_id: str | None = None,
    plate_text: str | None = Query(default=None, min_length=1, max_length=100),
    vehicle_type: str | None = None,
    has_plate: bool | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """List vehicles; date filters match their first-seen UTC calendar date.

    Plate search is a literal, case-insensitive substring search. Pagination
    returns total matching records as well as the requested slice.
    """
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must not be after date_to")
    conditions, values = [], []
    for column, value in (("session_id", session_id), ("class_name", vehicle_type)):
        if value is not None:
            conditions.append(f"{column} = ?")
            values.append(value)
    if plate_text is not None:
        conditions.append("instr(lower(plate_text), lower(?)) > 0")
        values.append(plate_text)
    if has_plate is not None:
        conditions.append("plate_text IS NOT NULL" if has_plate else "plate_text IS NULL")
    if date_from:
        conditions.append("date(first_seen) >= ?")
        values.append(date_from.isoformat())
    if date_to:
        conditions.append("date(first_seen) <= ?")
        values.append(date_to.isoformat())
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    with database() as connection:
        # Keep count and page consistent if the video pipeline writes meanwhile.
        connection.execute("BEGIN")
        total = connection.execute(
            "SELECT COUNT(*) FROM plate_recognitions" + where, values).fetchone()[0]
        rows = connection.execute(
            "SELECT * FROM plate_recognitions" + where +
            " ORDER BY first_seen DESC, session_id, vehicle_id LIMIT ? OFFSET ?",
            [*values, limit, offset]).fetchall()
    return {"items": [dict(row) for row in rows], "total": total,
            "limit": limit, "offset": offset}


@app.get("/vehicles/{session_id}/{vehicle_id}", tags=["Vehicles"])
def get_vehicle(session_id: str, vehicle_id: int) -> dict:
    """Tracking IDs repeat between videos, so both identifiers are required."""
    with database() as connection:
        row = connection.execute(
            "SELECT * FROM plate_recognitions WHERE session_id = ? AND vehicle_id = ?",
            (session_id, vehicle_id)).fetchone()
    if row is None:
        raise HTTPException(404, "Vehicle not found")
    return dict(row)


@app.get("/vehicles/{session_id}/{vehicle_id}/plate-image", tags=["Vehicles"])
def get_plate_image(session_id: str, vehicle_id: int):
    """Serve the best reading's PNG, restricted to the configured crop folder."""
    vehicle = get_vehicle(session_id, vehicle_id)
    if not vehicle["image_path"]:
        raise HTTPException(404, "Plate image not available")
    # Resolve symlinks and reject database paths pointing outside plate storage.
    root = Path(settings.PLATE_DIR).resolve()
    path = Path(vehicle["image_path"]).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".png" or not path.is_file():
        raise HTTPException(404, "Plate image not available")
    return FileResponse(path, media_type="image/png")


# These summaries come from vehicle records, not a separate sessions table.
# Runs without any saved vehicles cannot appear; start/end below are observation
# times, not job execution times. Job status and video links are future work.
SESSION_SUMMARY = """
    SELECT session_id, COUNT(*) AS vehicle_count,
           COUNT(plate_text) AS recognized_count,
           MIN(first_seen) AS first_seen, MAX(last_seen) AS last_seen
    FROM plate_recognitions
"""


@app.get("/sessions", tags=["Sessions"])
def list_sessions(limit: int = Query(default=50, ge=1, le=200),
                  offset: int = Query(default=0, ge=0)) -> dict:
    """List sessions represented in the saved vehicle records."""
    with database() as connection:
        connection.execute("BEGIN")
        total = connection.execute(
            "SELECT COUNT(DISTINCT session_id) FROM plate_recognitions").fetchone()[0]
        rows = connection.execute(
            SESSION_SUMMARY + " GROUP BY session_id ORDER BY first_seen DESC, session_id LIMIT ? OFFSET ?",
            (limit, offset)).fetchall()
    return {"items": [dict(row) for row in rows], "total": total,
            "limit": limit, "offset": offset}


@app.get("/sessions/{session_id}", tags=["Sessions"])
def get_session(session_id: str) -> dict:
    """Return vehicle counts and observation times for one session."""
    with database() as connection:
        row = connection.execute(
            SESSION_SUMMARY + " WHERE session_id = ? GROUP BY session_id",
            (session_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Session not found")
    return dict(row)

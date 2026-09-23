"""Exercise API routes against isolated SQLite records and crop files."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.api.app import app
from app.database.database import connect_database


class APITests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / "anpr.db"
        self.plates = self.root / "plates"
        self.plates.mkdir()
        for name, value in (("DATABASE_PATH", str(self.db)), ("PLATE_DIR", str(self.plates))):
            patcher = patch("app.api.app.settings." + name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        connection = connect_database(self.db)
        with connection:
            connection.executemany("""INSERT INTO plate_recognitions
                (session_id, vehicle_id, class_name, plate_text, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)""", [
                ("one", 1, "car", "ABC123", "2026-09-20T10:00:00+00:00", "2026-09-20T10:01:00+00:00"),
                ("one", 2, "bus", None, "2026-09-20T10:00:00+00:00", "2026-09-20T10:02:00+00:00"),
                ("two", 1, "car", "XYZ789", "2026-09-21T10:00:00+00:00", "2026-09-21T10:01:00+00:00"),
            ])
        connection.close()

    def test_health_and_documentation(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertIn("/vehicles", self.client.get("/openapi.json").json()["paths"])

    def test_vehicle_filters_pagination_and_literal_search(self):
        result = self.client.get("/vehicles", params={"limit": 1, "offset": 1}).json()
        self.assertEqual(result["total"], 3)
        self.assertEqual(len(result["items"]), 1)
        result = self.client.get("/vehicles", params={"plate_text": "abc", "vehicle_type": "car", "session_id": "one"}).json()
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["plate_text"], "ABC123")
        self.assertEqual(self.client.get("/vehicles?has_plate=false").json()["total"], 1)
        self.assertEqual(self.client.get("/vehicles?date_from=2026-09-21&date_to=2026-09-21").json()["total"], 1)
        for value in ("%", "' OR 1=1 --"):
            self.assertEqual(self.client.get("/vehicles", params={"plate_text": value}).json()["total"], 0)

    def test_validation_and_missing_records(self):
        for query in ("limit=0", "offset=-1", "date_from=bad", "date_from=2026-09-22&date_to=2026-09-20"):
            self.assertEqual(self.client.get("/vehicles?" + query).status_code, 422)
        self.assertEqual(self.client.get("/vehicles/one/1").json()["plate_text"], "ABC123")
        for route in ("/vehicles/absent/1", "/sessions/absent", "/vehicles/one/2/plate-image"):
            self.assertEqual(self.client.get(route).status_code, 404)

    def test_session_counts(self):
        self.assertEqual(self.client.get("/sessions?limit=1").json()["total"], 2)
        result = self.client.get("/sessions/one").json()
        self.assertEqual(result["vehicle_count"], 2)
        self.assertEqual(result["recognized_count"], 1)

    def test_image_serving_and_folder_boundary(self):
        for path, status in ((self.plates / "plate.png", 200), (self.root / "outside.png", 404)):
            path.write_bytes(b"test PNG bytes")
            connection = connect_database(self.db)
            with connection:
                connection.execute("UPDATE plate_recognitions SET image_path=? WHERE session_id='one' AND vehicle_id=1", (str(path),))
            connection.close()
            response = self.client.get("/vehicles/one/1/plate-image")
            self.assertEqual(response.status_code, status)
            if status == 200:
                self.assertEqual(response.content, b"test PNG bytes")
                self.assertEqual(response.headers["content-type"], "image/png")
            path.unlink()
            self.assertEqual(self.client.get("/vehicles/one/1/plate-image").status_code, 404)

    def test_database_failure_returns_service_unavailable(self):
        with patch("app.api.app.settings.DATABASE_PATH", str(self.root / "missing.db")):
            with self.assertLogs("anpr", level="ERROR"):
                self.assertEqual(self.client.get("/vehicles").status_code, 503)
        self.assertFalse((self.root / "missing.db").exists())


if __name__ == "__main__":
    unittest.main()

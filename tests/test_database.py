"""Exercise persistence and duplicate handling with real temporary SQLite files."""
import sqlite3
import cv2
import numpy as np
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as R

from app.core.exceptions import DatabaseException
from app.database.database import connect_database
from app.database.repository import PlateRepository


def reading(text="ABC123", confidence=.8, vehicle_id=1):
    return R(match=R(vehicle_id=vehicle_id), ocr=R(text=text, confidence=confidence))


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "nested/anpr.db"
        self.db = connect_database(self.path)
        self.addCleanup(self.db.close)
        self.repo = PlateRepository(self.db)

    def rows(self):
        return self.db.execute("SELECT * FROM plate_recognitions").fetchall()

    def test_best_reading_and_duplicate_handling(self):
        self.repo.save_readings("session", [reading()], 1.0)
        first = self.rows()[0]
        self.repo.save_readings("session", [reading("WRONG", .6)], 2.0)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]["plate_text"], "ABC123")
        self.repo.save_readings("session", [reading("BETTER", .95)], 3.0)
        row = self.rows()[0]
        self.assertEqual(row["plate_text"], "BETTER")
        self.assertEqual(row["confidence"], .95)
        self.assertEqual(row["video_timestamp"], 3.0)
        self.assertEqual(row["first_seen"], first["first_seen"])
        self.assertGreaterEqual(row["last_seen"], first["last_seen"])
        self.repo.save_readings("session", [reading("TIED", .95)], 4.0)
        self.assertEqual(self.rows()[0]["plate_text"], "BETTER")
        self.assertEqual(self.rows()[0]["video_timestamp"], 3.0)

    def test_sessions_are_independent_and_data_persists(self):
        for session in ("one", "two"):
            self.repo.save_readings(session, [reading("O'NEIL")], 0.0)
        other = connect_database(self.path)
        try:
            rows = other.execute("SELECT * FROM plate_recognitions").fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["plate_text"], "O'NEIL")
        finally:
            other.close()

    def test_unreadable_unmatched_and_invalid_readings_are_skipped(self):
        self.repo.save_readings("one", [reading(vehicle_id=None), reading("  "),
            reading(confidence=float("nan")), reading(confidence=1.1),
            R(match=R(vehicle_id=1), ocr=None)], 0)
        self.assertEqual(self.rows(), [])

    def test_failed_frame_is_rolled_back(self):
        self.db.execute("""CREATE TRIGGER reject_second BEFORE INSERT ON plate_recognitions
            WHEN NEW.vehicle_id = 2 BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
        with self.assertRaises(DatabaseException):
            self.repo.save_readings("one", [reading(), reading(vehicle_id=2)], 0)
        self.assertEqual(self.rows(), [])

    def test_vehicle_without_plate_later_gets_reading(self):
        car = R(vehicle_id=1, class_id=2, class_name="car")
        person = R(vehicle_id=2, class_id=0, class_name="person")
        self.repo.save_readings("one", [], 0, vehicles=[car, person])
        first = self.rows()[0]
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(first["class_name"], "car")
        self.assertIsNone(first["plate_text"])
        self.assertIsNone(first["confidence"])
        self.assertIsNone(first["video_timestamp"])
        self.repo.save_readings("one", [reading()], 2, vehicles=[car])
        self.repo.save_readings("one", [], 3, vehicles=[car])
        row = self.rows()[0]
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(row["plate_text"], "ABC123")
        self.assertEqual(row["video_timestamp"], 2)
        self.assertEqual(row["first_seen"], first["first_seen"])
        self.assertGreaterEqual(row["last_seen"], first["last_seen"])

    def test_legacy_migration_preserves_records(self):
        path = Path(self.temp.name) / "legacy.db"
        legacy = sqlite3.connect(path)
        legacy.execute("""CREATE TABLE plate_recognitions (
            session_id TEXT NOT NULL, vehicle_id INTEGER NOT NULL,
            plate_text TEXT NOT NULL, confidence REAL NOT NULL,
            video_timestamp REAL NOT NULL, first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL, PRIMARY KEY(session_id, vehicle_id))""")
        legacy.execute("INSERT INTO plate_recognitions VALUES (?, ?, ?, ?, ?, ?, ?)",
                       ("old", 1, "ABC", .9, 2, "first", "last"))
        legacy.commit()
        legacy.close()
        db = connect_database(path)
        try:
            original = db.execute("SELECT * FROM plate_recognitions").fetchone()
            self.assertEqual(original["plate_text"], "ABC")
            self.assertEqual(original["first_seen"], "first")
            PlateRepository(db).save_readings("new", [], 0,
                vehicles=[R(vehicle_id=1, class_id=2, class_name="car")])
            self.assertEqual(db.execute("SELECT COUNT(*) FROM plate_recognitions").fetchone()[0], 2)
        finally:
            db.close()
        reopened = connect_database(path)
        reopened.close()

    def test_crop_tracks_best_fresh_reading(self):
        directory = Path(self.temp.name) / "plates"
        repo = PlateRepository(self.db, directory)
        first = reading()
        first.crop = np.full((5, 10, 3), 40, dtype=np.uint8)
        repo.save_readings("session", [first], 0)
        initial = self.rows()[0]["image_path"]
        np.testing.assert_array_equal(cv2.imread(initial), first.crop)
        repo.save_readings("session", [reading()], 1)  # Cached, no crop.
        worse = reading("WRONG", .6)
        worse.crop = np.zeros((5, 10, 3), dtype=np.uint8)
        repo.save_readings("session", [worse], 2)
        self.assertEqual(self.rows()[0]["image_path"], initial)
        self.assertEqual(len(list(directory.glob("*.png"))), 1)
        better = reading("BETTER", .95)
        better.crop = np.full((5, 10, 3), 200, dtype=np.uint8)
        repo.save_readings("session", [better], 3)
        updated = self.rows()[0]
        self.assertNotEqual(updated["image_path"], initial)
        self.assertEqual(updated["plate_text"], "BETTER")
        np.testing.assert_array_equal(cv2.imread(updated["image_path"]), better.crop)

    def test_database_failure_removes_new_crop(self):
        directory = Path(self.temp.name) / "plates"
        repo = PlateRepository(self.db, directory)
        self.db.execute("""CREATE TRIGGER reject_crop BEFORE INSERT ON plate_recognitions
            BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
        item = reading()
        item.crop = np.zeros((5, 10, 3), dtype=np.uint8)
        with self.assertRaises(DatabaseException):
            repo.save_readings("session", [item], 0)
        self.assertEqual(list(directory.glob("*.png")), [])
        self.assertEqual(self.rows(), [])

    def test_crop_column_added_to_previous_schema(self):
        path = Path(self.temp.name) / "previous.db"
        db = sqlite3.connect(path)
        db.execute("""CREATE TABLE plate_recognitions (
            session_id TEXT, vehicle_id INTEGER, class_id INTEGER, class_name TEXT,
            plate_text TEXT, confidence REAL, video_timestamp REAL,
            first_seen TEXT, last_seen TEXT, PRIMARY KEY(session_id, vehicle_id))""")
        db.execute("INSERT INTO plate_recognitions VALUES ('s',1,2,'car',NULL,NULL,NULL,'a','b')")
        db.commit()
        db.close()
        db = connect_database(path)
        try:
            row = db.execute("SELECT * FROM plate_recognitions").fetchone()
            self.assertEqual(row["class_name"], "car")
            self.assertIsNone(row["image_path"])
        finally:
            db.close()

    def test_connection_errors_are_wrapped(self):
        with self.assertRaises(DatabaseException):
            connect_database(Path(self.temp.name))
        self.db.close()
        with self.assertRaises(DatabaseException):
            self.repo.save_readings("one", [reading()], 0)


if __name__ == "__main__":
    unittest.main()

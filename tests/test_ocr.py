"""Crop boundaries, OCR filtering, and association tests."""
import unittest
from types import SimpleNamespace as R
from unittest.mock import Mock, patch
import numpy as np
from app.recognition.ocr_engine import OCREngine, OCRResult
from app.recognition.plate_reader import PlateReader, crop_plate
from app.core.exceptions import OCRException

class OCRTests(unittest.TestCase):
    def test_only_fresh_ocr_carries_its_crop(self):
        engine = Mock()
        engine.recognize.return_value = OCRResult('ABC', .9)
        reader = PlateReader(engine)
        frame = np.full((10, 10, 3), 42, dtype=np.uint8)
        matches = [R(vehicle_id=1, plate=R(bbox=(1, 2, 5, 6)))]
        fresh = reader.read(frame, matches)[0]
        np.testing.assert_array_equal(fresh.crop, frame[2:6, 1:5])
        frame[:] = 0
        self.assertTrue((fresh.crop == 42).all())
        cached = reader.read(frame, matches)[0]
        self.assertIsNone(cached.crop)
        self.assertEqual(cached.ocr.text, 'ABC')
        engine.recognize.assert_called_once()

    def test_cache_refresh_and_expiry(self):
        engine = Mock()
        engine.recognize.return_value = OCRResult('ABC123', .9)
        reader = PlateReader(engine, retry_frames=2, refresh_frames=3, cache_ttl_frames=2)
        frame = np.zeros((10,10,3), dtype=np.uint8)
        matches = [R(vehicle_id=5, plate=R(bbox=(0,0,5,5)))]
        for _ in range(3):
            self.assertEqual(reader.read(frame, matches)[0].ocr.text, 'ABC123')
        self.assertEqual(engine.recognize.call_count, 1)
        reader.read(frame, matches)
        self.assertEqual(engine.recognize.call_count, 2)
        reader.read(frame, [])
        reader.read(frame, [])
        reader.read(frame, matches)
        self.assertEqual(engine.recognize.call_count, 3)

    def test_unreadable_retries_are_throttled(self):
        engine = Mock()
        engine.recognize.side_effect = [None, OCRResult('ABC', .9)]
        reader = PlateReader(engine, retry_frames=2)
        frame = np.zeros((10,10,3), dtype=np.uint8)
        matches = [R(vehicle_id=5, plate=R(bbox=(0,0,5,5)))]
        reader.read(frame, matches)
        reader.read(frame, matches)
        self.assertEqual(engine.recognize.call_count, 1)
        self.assertEqual(reader.read(frame, matches)[0].ocr.text, 'ABC')

    def test_crop_boundaries_and_copy(self):
        frame = np.zeros((10,20,3),dtype=np.uint8)
        crop = crop_plate(frame,(-5,-2,6.5,8))
        self.assertEqual(crop.shape,(8,7,3))
        crop[:] = 255
        self.assertFalse(frame.any())
        for box in [(30,0,40,5),(0,0,0,5),(0,0,float('nan'),5)]:
            self.assertIsNone(crop_plate(frame,box))

    @patch("app.recognition.ocr_engine.settings.OCR_CONFIDENCE", .5)
    def test_confidence_filter_and_empty_output(self):
        engine = OCREngine.__new__(OCREngine)
        engine._engine = Mock()
        engine._engine.predict.return_value = [{'rec_texts':['ABC','123','bad'], 'rec_scores':[.9,.8,.2]}]
        frame = np.zeros((10,20,3),dtype=np.uint8)
        result = engine.recognize(frame)
        self.assertEqual(result.text,'ABC 123')
        self.assertAlmostEqual(result.confidence,.85)
        engine._engine.predict.return_value = [{'rec_texts':[], 'rec_scores':[]}]
        self.assertIsNone(engine.recognize(frame))
        self.assertIsNone(engine.recognize(np.array([])))

    def test_failure_does_not_stop_later_plates(self):
        engine = Mock()
        engine.recognize.side_effect = [OCRException('failed'), OCRResult('ABC123',.9)]
        matches = [R(vehicle_id=i,plate=R(bbox=(0,0,5,5))) for i in (None,5,6)]
        with self.assertLogs('anpr',level='ERROR'):
            readings = PlateReader(engine).read(np.zeros((10,10,3),dtype=np.uint8),matches)
        self.assertIsNone(readings[0].ocr)
        self.assertIsNone(readings[1].ocr)
        self.assertEqual(readings[2].match.vehicle_id,6)
        self.assertEqual(readings[2].ocr.text,'ABC123')
        self.assertEqual(engine.recognize.call_count,2)

if __name__ == '__main__':
    unittest.main()

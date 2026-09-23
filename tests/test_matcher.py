"""Geometry and late-appearance tests without loading inference models."""
import unittest
from types import SimpleNamespace as Record
from app.matching.matcher import PlateMatcher
from app.core.exceptions import MatchingException

class MatcherTests(unittest.TestCase):
    def matcher(self, method='center', threshold=.1):
        return PlateMatcher(method=method, iou_threshold=threshold)

    def test_late_plate_and_person_exclusion(self):
        car = Record(vehicle_id=5, class_id=2, bbox=[0,0,100,100])
        person = Record(vehicle_id=1, class_id=0, bbox=[40,40,60,60])
        plate = Record(bbox=[45,45,55,55])
        matcher = self.matcher()
        self.assertEqual(matcher.match([car], []), [])
        self.assertEqual(matcher.match([person,car], [plate])[0].vehicle_id, 5)
        self.assertIsNone(matcher.match([person], [plate])[0].vehicle_id)

    def test_overlap_and_unmatched(self):
        large = Record(vehicle_id=5, class_id=2, bbox=[0,0,100,100])
        small = Record(vehicle_id=8, class_id=7, bbox=[20,20,80,80])
        plates = [Record(bbox=[30,30,40,40]), Record(bbox=[200,200,210,210])]
        self.assertEqual([m.vehicle_id for m in self.matcher().match([large,small],plates)], [8,None])

    def test_iou_threshold_and_disjoint(self):
        car = Record(vehicle_id=5,class_id=2,bbox=[0,0,10,10])
        plate = Record(bbox=[0,0,5,10])
        self.assertEqual(self.matcher('iou',.5).match([car],[plate])[0].vehicle_id,5)
        self.assertIsNone(self.matcher('iou',.6).match([car],[plate])[0].vehicle_id)
        self.assertIsNone(self.matcher('iou',0).match([car],[Record(bbox=[20,20,30,30])])[0].vehicle_id)

    def test_invalid_box(self):
        with self.assertRaises(MatchingException):
            self.matcher().match([], [Record(bbox=[0,0,0,10])])

if __name__ == '__main__':
    unittest.main()

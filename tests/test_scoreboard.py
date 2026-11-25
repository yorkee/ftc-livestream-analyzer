"""Unit tests for top/bottom panels and four-team validation."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from ftc_analyzer.scoreboard import (
    detect_alliance_panels, read_team_slots, stabilize_team_numbers, live_boxes,
)


class ScoreboardTests(unittest.TestCase):
    def make_frame(self, bottom=False):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        y = 540 if bottom else 0
        red, blue = (40, 0, 210), (200, 110, 15)
        cv2.rectangle(frame, (0, y + 50), (639, y + 170), red, -1)
        cv2.rectangle(frame, (640, y + 50), (1279, y + 170), blue, -1)
        for x, values in ((48, ("30403", "24378")), (686, ("18664", "7842"))):
            for row, number in enumerate(values):
                cv2.putText(frame, number, (x, y + 95 + row * 53),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return frame

    def test_top_and_bottom_banner_detection(self):
        for bottom in (False, True):
            with self.subTest(bottom=bottom):
                panels = detect_alliance_panels(self.make_frame(bottom))
                self.assertEqual(set(panels), {"red", "blue"})
                self.assertEqual(panels["red"][1] >= 500, bottom)

    def test_intro_layout_rejected(self):
        for bottom in (False, True):
            self.assertEqual(live_boxes(self.make_frame(bottom)), {})

    def test_south_australia_fixtures(self):
        root = Path(__file__).resolve().parent
        positive = root / "frame_00000017.jpg"
        negative = root / "frame_00000034.jpg"
        if not positive.exists() or not negative.exists():
            self.skipTest("Copy both South Australia frames into tests/")
        image = cv2.imread(str(positive))
        expected = {"red1":38987,"red2":22407,"blue1":38884,"blue2":18611}
        for frame in (image, np.roll(image, 540, axis=0)):
            slots, _, _ = read_team_slots(frame)
            self.assertEqual(slots, expected)
        self.assertEqual(live_boxes(cv2.imread(str(negative))), {})

    def test_temporal_voting_and_schedule_confirmation(self):
        schedule = [{"matchNumber": 3, "teams": [
            {"teamNumber": 30403, "station": "Red1"},
            {"teamNumber": 24378, "station": "Red2"},
            {"teamNumber": 18664, "station": "Blue1"},
            {"teamNumber": 7842, "station": "Blue2"},
        ]}]
        rows = [
            {"match_number_candidate": 3, "team_slots_raw": {
                "red1": 30403, "red2": 24378, "blue1": n, "blue2": 7842}}
            for n in (1664, 18664, 18664)
        ]
        result = stabilize_team_numbers(rows, schedule)
        self.assertTrue(all(row["teams_complete"] for row in result))
        self.assertTrue(all(row["team_number_source"] == "first_schedule_verified"
                            for row in result))
        self.assertTrue(all(row["blue_team_numbers"] == [18664, 7842]
                            for row in result))

    def test_schedule_does_not_fabricate_without_ocr_agreement(self):
        schedule = [{"matchNumber": 1, "teams": [
            {"teamNumber": 1111, "station": "Red1"},
            {"teamNumber": 2222, "station": "Red2"},
            {"teamNumber": 3333, "station": "Blue1"},
            {"teamNumber": 4444, "station": "Blue2"},
        ]}]
        rows = [{"match_number_candidate": 1, "team_slots_raw": {
            "red1": 9999, "red2": None, "blue1": None, "blue2": None}}]
        result = stabilize_team_numbers(rows, schedule)
        self.assertEqual(result[0]["team_number_source"], "temporal_ocr")
        self.assertFalse(result[0]["teams_complete"])

    def test_temporal_vote_without_schedule(self):
        rows = [{"match_number_candidate": 3, "team_slots_raw": {
            "red1": 30403, "red2": 24378, "blue1": n, "blue2": 7842}}
            for n in (18664, 1664, 18664)]
        result = stabilize_team_numbers(rows)
        self.assertTrue(all(row["team_slots"]["blue1"] == 18664
                            for row in result))


if __name__ == "__main__":
    unittest.main()

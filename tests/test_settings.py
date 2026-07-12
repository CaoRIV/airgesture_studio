from __future__ import annotations

import unittest

from airgesture.config import SETTINGS, SettingsError, load_settings
from airgesture.puzzle.capture_gesture import (
    CaptureGestureConfig,
    TwoHandSpreadCaptureGesture,
)


class SettingsTests(unittest.TestCase):
    def test_loads_default_json_settings(self) -> None:
        settings = load_settings()

        self.assertEqual(settings.camera.fps, 30)
        self.assertEqual(settings.air_drawing.drawing.thin_brush_size, 7)
        self.assertAlmostEqual(settings.air_drawing.adaptive_smoothing.slow_alpha, 0.18)
        self.assertEqual(settings.air_drawing.drawing.max_undo_steps, 20)
        self.assertEqual(settings.puzzle.capture.stable_frames_required, 5)
        self.assertAlmostEqual(settings.puzzle.capture.spread_ratio_required, 0.34)

    def test_calibration_hand_count_does_not_mutate_base_settings(self) -> None:
        tracker = SETTINGS.calibration_tracker(required_hands=2)

        self.assertEqual(tracker.max_num_hands, 2)
        self.assertEqual(SETTINGS.calibration.tracker.max_num_hands, 1)

    def test_missing_settings_file_has_clear_error(self) -> None:
        with self.assertRaisesRegex(SettingsError, "Settings file not found"):
            load_settings("does-not-exist.json")

    def test_capture_gesture_uses_provided_config(self) -> None:
        config = CaptureGestureConfig(
            stable_frames_required=9,
            centered_radius_ratio=0.25,
            cluster_radius_ratio=0.20,
            spread_ratio_required=0.42,
            stable_motion_threshold=30.0,
            dynamic_motion_ratio=0.05,
        )

        gesture = TwoHandSpreadCaptureGesture(config=config)

        self.assertEqual(gesture.stable_frames_required, 9)
        self.assertEqual(gesture.spread_ratio_required, 0.42)
        self.assertEqual(gesture.dynamic_motion_ratio, 0.05)


if __name__ == "__main__":
    unittest.main()

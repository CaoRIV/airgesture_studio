from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from airgesture.config import SettingsError, load_settings, resolve_settings_path
from airgesture.config.settings import BUNDLED_SETTINGS_PATH
from airgesture.puzzle.capture_gesture import (
    CaptureGestureConfig,
    TwoHandSpreadCaptureGesture,
)


class SettingsTests(unittest.TestCase):
    def test_loads_bundled_json_settings(self) -> None:
        settings = load_settings(BUNDLED_SETTINGS_PATH)

        self.assertEqual(settings.camera.fps, 30)
        self.assertEqual(settings.camera.discovery_max_devices, 5)
        self.assertEqual(settings.camera.read_failure_tolerance, 2)
        self.assertEqual(settings.camera.reconnect_attempts, 3)
        self.assertEqual(settings.air_drawing.drawing.thin_brush_size, 7)
        self.assertAlmostEqual(settings.air_drawing.adaptive_smoothing.slow_alpha, 0.18)
        self.assertEqual(settings.air_drawing.drawing.max_undo_steps, 20)
        self.assertAlmostEqual(
            settings.air_drawing.recognition.snap_confidence_threshold,
            0.74,
        )
        self.assertEqual(settings.air_drawing.recognition.suggestion_count, 3)
        self.assertEqual(settings.puzzle.capture.stable_frames_required, 5)
        self.assertAlmostEqual(settings.puzzle.capture.spread_ratio_required, 0.34)
        self.assertEqual(settings.calibration.min_brightness, 55.0)
        self.assertEqual(settings.calibration.max_brightness, 220.0)

    def test_calibration_hand_count_does_not_mutate_base_settings(self) -> None:
        settings = load_settings(BUNDLED_SETTINGS_PATH)
        tracker = settings.calibration_tracker(required_hands=2)

        self.assertEqual(tracker.max_num_hands, 2)
        self.assertEqual(settings.calibration.tracker.max_num_hands, 1)

    def test_missing_settings_file_has_clear_error(self) -> None:
        with self.assertRaisesRegex(SettingsError, "Settings file not found"):
            load_settings("does-not-exist.json")

    def test_default_settings_are_copied_to_user_config_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with patch.dict(
                "os.environ",
                {"AIRGESTURE_DATA_DIR": temporary_directory},
                clear=True,
            ):
                settings_path = resolve_settings_path()
                settings = load_settings()

                self.assertEqual(
                    settings_path,
                    Path(temporary_directory) / "config" / "settings.json",
                )
                self.assertTrue(settings_path.is_file())

        self.assertEqual(settings.camera.fps, 30)

    def test_explicit_settings_path_is_not_replaced_by_defaults(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "custom-settings.json"
            with patch.dict(
                "os.environ",
                {"AIRGESTURE_SETTINGS_PATH": str(missing_path)},
                clear=True,
            ):
                self.assertEqual(resolve_settings_path(), missing_path)
                with self.assertRaisesRegex(SettingsError, "Settings file not found"):
                    load_settings()

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

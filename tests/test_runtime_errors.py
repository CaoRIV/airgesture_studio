from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from airgesture.config import SettingsError, require_valid_settings
from airgesture.core.camera import Camera
from airgesture.errors import CameraError
from airgesture.ui.runtime_errors import run_with_error_dialog


class RuntimeErrorHandlingTests(unittest.TestCase):
    def test_expected_error_is_shown_without_crashing(self) -> None:
        def fail() -> int:
            raise CameraError("Camera is unavailable")

        with (
            patch("airgesture.ui.runtime_errors.show_error_dialog") as dialog,
            patch("airgesture.ui.runtime_errors.get_runtime_logger") as logger,
        ):
            exit_code = run_with_error_dialog("Camera Check", fail)

        self.assertEqual(exit_code, 1)
        dialog.assert_called_once_with("Camera Check", "Camera is unavailable")
        logger.return_value.error.assert_called_once()

    def test_invalid_runtime_settings_are_rejected_after_import(self) -> None:
        error = SettingsError("Invalid user settings")
        with patch("airgesture.config.SETTINGS_ERROR", error):
            with self.assertRaisesRegex(SettingsError, "Invalid user settings"):
                require_valid_settings()


class CameraFailureTests(unittest.TestCase):
    def test_open_or_raise_reports_unavailable_camera(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = False
        with patch("airgesture.core.camera.cv2.VideoCapture", return_value=capture):
            camera = Camera()
            with self.assertRaisesRegex(CameraError, "Could not open webcam 0"):
                camera.open_or_raise()

        capture.release.assert_called_once_with()

    def test_read_or_raise_reports_disconnected_camera(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        camera = Camera()
        camera._capture = capture

        with self.assertRaisesRegex(CameraError, "stopped returning frames"):
            camera.read_or_raise()


if __name__ == "__main__":
    unittest.main()

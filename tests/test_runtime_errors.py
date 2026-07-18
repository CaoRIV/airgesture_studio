from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from airgesture.config import SettingsError, require_valid_settings
from airgesture.core.camera import Camera, CameraConfig
from airgesture.core.hand_tracker import _validate_model
from airgesture.errors import (
    CameraAccessError,
    CameraDisconnectedError,
    CameraError,
    CameraNotFoundError,
    HandTrackingError,
)
from airgesture.ui.runtime_errors import run_with_error_dialog


class RuntimeErrorHandlingTests(unittest.TestCase):
    def test_missing_hand_model_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(HandTrackingError, "model is missing"):
            _validate_model(Path("missing-hand-model.task"))

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
    @staticmethod
    def capture(
        opened: bool,
        reads: list[tuple[bool, object]] | None = None,
        width: float = 1280.0,
        height: float = 720.0,
        fps: float = 30.0,
    ) -> Mock:
        capture = Mock()
        capture.isOpened.return_value = opened
        capture.set.return_value = True
        capture.read.side_effect = reads or []
        properties = {
            3: width,
            4: height,
            5: fps,
        }
        capture.get.side_effect = lambda property_id: properties.get(
            property_id,
            0.0,
        )
        return capture

    def test_open_or_raise_reports_unavailable_camera(self) -> None:
        capture = self.capture(opened=False)
        with patch("airgesture.core.camera.cv2.VideoCapture", return_value=capture):
            camera = Camera()
            with self.assertRaisesRegex(CameraNotFoundError, "was not detected"):
                camera.open_or_raise()

        self.assertEqual(
            capture.release.call_count,
            len(Camera.backend_candidates()),
        )

    def test_read_or_raise_reports_disconnected_camera(self) -> None:
        capture = self.capture(opened=True, reads=[(False, None)])
        camera = Camera(
            CameraConfig(
                mirror=False,
                read_failure_tolerance=0,
                reconnect_attempts=0,
            )
        )
        camera._capture = capture

        with self.assertRaisesRegex(CameraDisconnectedError, "was disconnected"):
            camera.read_or_raise()

    def test_discovered_but_unavailable_camera_reports_access_error(self) -> None:
        capture = self.capture(opened=False)
        config = CameraConfig(camera_index=2)
        config.available_indices = (2,)
        with patch("airgesture.core.camera.cv2.VideoCapture", return_value=capture):
            camera = Camera(config)
            with self.assertRaisesRegex(
                CameraAccessError,
                "detected but could not be opened",
            ):
                camera.open_or_raise()

    def test_falls_back_from_directshow_to_media_foundation(self) -> None:
        directshow = self.capture(opened=False)
        media_foundation = self.capture(opened=True)
        with patch(
            "airgesture.core.camera.cv2.VideoCapture",
            side_effect=[directshow, media_foundation],
        ):
            camera = Camera(CameraConfig(mirror=False))
            camera.open_or_raise()

        self.assertIsNotNone(camera.info)
        assert camera.info is not None
        self.assertEqual(camera.info.backend, "Media Foundation")
        self.assertEqual(camera.info.width, 1280)
        self.assertEqual(camera.info.height, 720)
        self.assertEqual(camera.info.fps, 30.0)
        directshow.release.assert_called_once_with()

    def test_reconnects_after_camera_stops_returning_frames(self) -> None:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        initial = self.capture(opened=True, reads=[(False, None)])
        recovered = self.capture(opened=True, reads=[(True, frame)])
        config = CameraConfig(
            mirror=False,
            read_failure_tolerance=0,
            reconnect_attempts=1,
            reconnect_delay_seconds=0.0,
        )
        with patch(
            "airgesture.core.camera.cv2.VideoCapture",
            side_effect=[initial, recovered],
        ):
            camera = Camera(config)
            camera.open_or_raise()
            recovered_frame = camera.read_or_raise()

        np.testing.assert_array_equal(recovered_frame, frame)
        self.assertEqual(camera.reconnect_count, 1)

    def test_discovers_only_indices_that_can_be_opened(self) -> None:
        with patch.object(
            Camera,
            "_probe_index",
            side_effect=[True, False, True, False],
        ):
            indices = Camera.discover_indices(max_devices=4)

        self.assertEqual(indices, [0, 2])


if __name__ == "__main__":
    unittest.main()

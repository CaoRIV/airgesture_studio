from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from airgesture.ui.window import (
    ResponsiveWindow,
    ScreenMetrics,
    Viewport,
    fit_frame_to_viewport,
    is_f11_key,
)


class FrameFittingTests(unittest.TestCase):
    def test_frame_keeps_aspect_ratio_in_tall_viewport(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        output, bounds = fit_frame_to_viewport(frame, Viewport(900, 900))

        self.assertEqual(output.shape, (900, 900, 3))
        self.assertEqual(bounds, (0, 197, 900, 506))

    def test_f11_key_variants_are_recognized(self) -> None:
        self.assertTrue(is_f11_key(0x7A))
        self.assertTrue(is_f11_key(0x7A0000))
        self.assertFalse(is_f11_key(ord("q")))


class FullscreenInputTests(unittest.TestCase):
    def window(self) -> ResponsiveWindow:
        with patch(
            "airgesture.ui.window.screen_metrics",
            return_value=ScreenMetrics(1920, 1080, 1.5),
        ):
            return ResponsiveWindow("Test")

    def test_f11_toggles_and_is_consumed(self) -> None:
        window = self.window()
        with patch.object(window, "toggle_fullscreen") as toggle:
            consumed = window.handle_window_key(0x7A0000)

        self.assertTrue(consumed)
        toggle.assert_called_once_with()

    def test_escape_leaves_fullscreen_without_quitting(self) -> None:
        window = self.window()
        window.is_fullscreen = True
        with patch.object(window, "leave_fullscreen") as leave:
            consumed = window.handle_window_key(27)

        self.assertTrue(consumed)
        leave.assert_called_once_with()

    def test_escape_is_not_consumed_in_windowed_mode(self) -> None:
        window = self.window()

        self.assertFalse(window.handle_window_key(27))

    def test_native_close_is_detected(self) -> None:
        window = self.window()
        with patch(
            "airgesture.ui.window.cv2.getWindowProperty",
            return_value=0.0,
        ):
            self.assertFalse(window.is_open())

    def test_visible_window_stays_open(self) -> None:
        window = self.window()
        with patch(
            "airgesture.ui.window.cv2.getWindowProperty",
            return_value=1.0,
        ):
            self.assertTrue(window.is_open())


if __name__ == "__main__":
    unittest.main()

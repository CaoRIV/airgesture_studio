from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from airgesture import app
from airgesture.calibration import (
    CameraCheckConfig,
    draw_camera_check_hud,
)


class MenuNavigationTests(unittest.TestCase):
    def test_drawing_opens_directly_without_camera_check(self) -> None:
        with (
            patch.object(app.cv2, "destroyWindow"),
            patch.object(app.drawing_main, "main") as drawing_main,
            patch.object(app, "run_camera_check") as camera_check,
        ):
            app.run_action(app.MenuAction.DRAWING)

        drawing_main.assert_called_once_with()
        camera_check.assert_not_called()

    def test_puzzle_opens_directly_without_camera_check(self) -> None:
        with (
            patch.object(app.cv2, "destroyWindow"),
            patch.object(app.puzzle_main, "main") as puzzle_main,
            patch.object(app, "run_camera_check") as camera_check,
        ):
            app.run_action(app.MenuAction.PUZZLE)

        puzzle_main.assert_called_once_with()
        camera_check.assert_not_called()

    def test_camera_check_is_a_separate_optional_action(self) -> None:
        with (
            patch.object(app.cv2, "destroyWindow"),
            patch.object(app, "run_camera_check") as camera_check,
        ):
            app.run_action(app.MenuAction.CAMERA_CHECK)

        camera_check.assert_called_once()
        config = camera_check.call_args.args[0]
        self.assertIsInstance(config, CameraCheckConfig)
        self.assertEqual(config.required_hands, 1)

    def test_menu_renders_all_four_selected_states(self) -> None:
        for selected_index in range(len(app.MENU_ITEMS)):
            frame = app.render_menu(selected_index)
            self.assertEqual(frame.shape, (720, 1280, 3))
            self.assertGreater(int(np.count_nonzero(frame)), 0)


class CameraCheckLayoutTests(unittest.TestCase):
    def test_hud_renders_without_clipping_coordinates_at_hd(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_camera_check_hud(
            frame,
            (160, 0, 960, 720),
            CameraCheckConfig(),
            hand_count=1,
            brightness=120.0,
            ready=True,
        )

        self.assertGreater(int(np.count_nonzero(frame)), 0)

    def test_hud_scales_metrics_for_smaller_display(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_camera_check_hud(
            frame,
            (0, 0, 640, 480),
            CameraCheckConfig(),
            hand_count=0,
            brightness=35.0,
            ready=False,
        )

        self.assertGreater(int(np.count_nonzero(frame)), 0)


if __name__ == "__main__":
    unittest.main()

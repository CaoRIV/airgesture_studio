from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from airgesture import app
from airgesture.calibration import (
    CAMERA_CHECK_PAPER,
    CameraCheckConfig,
    draw_camera_check_hud,
)
from airgesture.config import SettingsError
from airgesture.core.camera import CameraConfig


class MenuNavigationTests(unittest.TestCase):
    def test_invalid_settings_show_dialog_before_menu_opens(self) -> None:
        error = SettingsError("Invalid settings file")
        with (
            patch.object(app, "require_valid_settings", side_effect=error),
            patch("airgesture.ui.runtime_errors.show_error_dialog") as dialog,
            patch("airgesture.ui.runtime_errors.get_runtime_logger"),
            patch.object(app.cv2, "namedWindow") as named_window,
        ):
            exit_code = app.main()

        self.assertEqual(exit_code, 1)
        dialog.assert_called_once_with(app.WINDOW_NAME, "Invalid settings file")
        named_window.assert_not_called()

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

    def test_menu_renders_at_resized_viewports(self) -> None:
        for width, height in ((960, 540), (1600, 900), (900, 700)):
            frame = app.render_menu(0, width=width, height=height)

            self.assertEqual(frame.shape, (height, width, 3))
            self.assertGreater(int(np.count_nonzero(frame)), 0)
            for x, y, item_width, item_height in app.menu_item_rects(width, height):
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + item_width, width)
                self.assertLessEqual(y + item_height, height)

    def test_mouse_hover_and_click_activate_menu_item(self) -> None:
        state = app.MenuPointerState()
        x, y, width, height = app.menu_item_rects(state.width, state.height)[1]
        center_x = x + width // 2
        center_y = y + height // 2

        app.handle_menu_mouse(
            app.cv2.EVENT_MOUSEMOVE,
            center_x,
            center_y,
            0,
            state,
        )
        self.assertEqual(state.hovered_index, 1)

        app.handle_menu_mouse(
            app.cv2.EVENT_LBUTTONUP,
            center_x,
            center_y,
            0,
            state,
        )
        self.assertEqual(state.activated_action, app.MenuAction.PUZZLE)

    def test_camera_selection_cycles_through_discovered_indices(self) -> None:
        self.assertEqual(app.cycle_camera_index([0, 2], 0, 1), 2)
        self.assertEqual(app.cycle_camera_index([0, 2], 2, 1), 0)
        self.assertEqual(app.cycle_camera_index([0, 2], 0, -1), 2)

    def test_rescan_selects_first_available_camera(self) -> None:
        config = CameraConfig(camera_index=4, discovery_max_devices=6)
        with patch.object(app.Camera, "discover_indices", return_value=[1, 3]) as discover:
            indices = app.refresh_camera_indices(config)

        self.assertEqual(indices, [1, 3])
        self.assertEqual(config.camera_index, 1)
        self.assertEqual(config.available_indices, (1, 3))
        discover.assert_called_once_with(max_devices=6)

    def test_menu_renders_no_camera_state(self) -> None:
        frame = app.render_menu(0, camera_index=0, camera_indices=[])

        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertGreater(int(np.count_nonzero(frame)), 0)


class CameraCheckLayoutTests(unittest.TestCase):
    def test_hud_renders_without_clipping_coordinates_at_hd(self) -> None:
        camera_color = (24, 28, 34)
        frame = np.full((720, 1280, 3), camera_color, dtype=np.uint8)
        draw_camera_check_hud(
            frame,
            (160, 0, 960, 720),
            CameraCheckConfig(),
            hand_count=1,
            brightness=120.0,
            ready=True,
        )

        self.assertGreater(int(np.count_nonzero(frame)), 0)
        self.assertTupleEqual(
            tuple(int(value) for value in frame[5, 640]),
            CAMERA_CHECK_PAPER,
        )
        self.assertTupleEqual(
            tuple(int(value) for value in frame[300, 640]),
            camera_color,
        )

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

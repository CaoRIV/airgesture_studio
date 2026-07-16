from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from airgesture.core.smoothing import AdaptivePointSmoother, AdaptiveSmoothingConfig
from airgesture.drawing.canvas import (
    CanvasConfig,
    DrawingCanvas,
    open_output_directory,
)
from airgesture.drawing.stroke_state import StrokeEndDebouncer
from airgesture.drawing.toolbar import GestureToolbar, ToolbarAction
from airgesture.errors import DrawingSaveError


class AdaptivePointSmootherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.smoother = AdaptivePointSmoother(
            AdaptiveSmoothingConfig(
                slow_alpha=0.10,
                fast_alpha=0.90,
                slow_speed=100.0,
                fast_speed=1000.0,
                speed_smoothing_alpha=0.25,
                missing_frame_tolerance=2,
            )
        )

    def test_uses_strong_smoothing_for_slow_motion(self) -> None:
        self.smoother.update((100, 100), timestamp_seconds=0.0)

        point = self.smoother.update((105, 100), timestamp_seconds=0.1)

        self.assertAlmostEqual(self.smoother.alpha, 0.10)
        self.assertEqual(point, (100, 100))

    def test_reduces_smoothing_for_fast_motion(self) -> None:
        self.smoother.update((100, 100), timestamp_seconds=0.0)

        point = self.smoother.update((300, 100), timestamp_seconds=0.1)

        self.assertAlmostEqual(self.smoother.alpha, 0.90)
        self.assertEqual(point, (280, 100))

    def test_tolerates_brief_missing_detection(self) -> None:
        self.smoother.update((100, 100), timestamp_seconds=0.0)

        self.assertEqual(self.smoother.update(None, 0.1), (100, 100))
        self.assertEqual(self.smoother.update(None, 0.2), (100, 100))
        self.assertIsNone(self.smoother.update(None, 0.3))


class StrokeEndDebouncerTests(unittest.TestCase):
    def test_waits_for_delay_and_grace_frames(self) -> None:
        debouncer = StrokeEndDebouncer(delay_seconds=0.18, grace_frames=2)

        self.assertFalse(debouncer.should_finalize(1.00, True))
        self.assertFalse(debouncer.should_finalize(1.10, True))
        self.assertTrue(debouncer.should_finalize(1.20, True))

    def test_repinch_cancels_pending_finalization(self) -> None:
        debouncer = StrokeEndDebouncer(delay_seconds=0.18, grace_frames=1)
        debouncer.should_finalize(1.00, True)
        debouncer.mark_active()

        self.assertFalse(debouncer.should_finalize(1.10, True))
        self.assertFalse(debouncer.should_finalize(1.20, True))


class CanvasUndoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = DrawingCanvas(
            CanvasConfig(brush_thickness=4, eraser_thickness=8, max_history_steps=3)
        )
        self.frame = np.zeros((80, 80, 3), dtype=np.uint8)
        self.canvas.ensure_size(self.frame.shape)

    def test_undo_restores_previous_committed_stroke(self) -> None:
        self.canvas.commit_clean_stroke([(10, 10), (60, 10)])
        first_stroke = self.canvas.compose(self.frame)
        self.canvas.commit_clean_stroke([(10, 30), (60, 30)])

        self.assertTrue(self.canvas.undo())
        np.testing.assert_array_equal(self.canvas.compose(self.frame), first_stroke)

        self.assertTrue(self.canvas.undo())
        self.assertFalse(np.any(self.canvas.compose(self.frame)))
        self.assertFalse(self.canvas.undo())

    def test_undo_restores_erased_pixels(self) -> None:
        self.canvas.commit_clean_stroke([(10, 40), (70, 40)])
        before_erase = self.canvas.compose(self.frame)
        self.canvas.begin_history_action()
        self.canvas.erase_line((25, 40), (55, 40))

        self.assertTrue(self.canvas.undo())
        np.testing.assert_array_equal(self.canvas.compose(self.frame), before_erase)

    def test_clear_removes_undo_history(self) -> None:
        self.canvas.commit_clean_stroke([(10, 10), (60, 10)])

        self.canvas.clear()

        self.assertFalse(self.canvas.can_undo)
        self.assertFalse(self.canvas.undo())


class CanvasSaveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = DrawingCanvas(CanvasConfig())
        self.canvas.ensure_size((80, 80, 3))
        self.canvas.commit_clean_stroke([(10, 10), (70, 70)])

    def test_save_writes_an_image_atomically(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            saved_path = self.canvas.save(temporary_directory, "drawing.png")

            self.assertEqual(saved_path, Path(temporary_directory) / "drawing.png")
            self.assertTrue(saved_path.is_file())
            self.assertFalse(any(saved_path.parent.glob("*.tmp.png")))

    def test_save_does_not_overwrite_an_existing_file(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            first_path = self.canvas.save(temporary_directory, "drawing.png")
            second_path = self.canvas.save(temporary_directory, "drawing.png")

            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.is_file())
            self.assertTrue(second_path.is_file())

    def test_save_reports_encoder_failure(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with patch(
                "airgesture.drawing.canvas.cv2.imwrite",
                return_value=False,
            ):
                with self.assertRaisesRegex(
                    DrawingSaveError,
                    "could not encode",
                ):
                    self.canvas.save(temporary_directory, "drawing.png")

            self.assertFalse(any(Path(temporary_directory).iterdir()))

    def test_open_output_directory_uses_windows_shell(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "Drawings"
            with patch("airgesture.drawing.canvas.os.startfile") as startfile:
                returned_path = open_output_directory(output_path)

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.is_dir())
            startfile.assert_called_once_with(str(output_path))


class ToolbarLayoutTests(unittest.TestCase):
    def test_save_and_folder_buttons_fit_hd_display(self) -> None:
        buttons = GestureToolbar().buttons(1280)

        self.assertIn(ToolbarAction.SAVE, [button.action for button in buttons])
        self.assertIn(
            ToolbarAction.OPEN_FOLDER,
            [button.action for button in buttons],
        )
        last_button = buttons[-1]
        self.assertGreaterEqual(buttons[0].rect[0], 0)
        self.assertLessEqual(last_button.rect[0] + last_button.rect[2], 1280)

    def test_toolbar_uses_two_rows_at_hd(self) -> None:
        buttons = GestureToolbar().buttons(1280, 720)

        self.assertEqual(len({button.rect[1] for button in buttons}), 2)

    def test_toolbar_uses_one_row_on_wide_display(self) -> None:
        buttons = GestureToolbar().buttons(1920, 1080)

        self.assertEqual(len({button.rect[1] for button in buttons}), 1)

    def test_toolbar_never_clips_on_narrow_display(self) -> None:
        buttons = GestureToolbar().buttons(640, 480)

        for button in buttons:
            x, y, width, height = button.rect
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + width, 640)
            self.assertLessEqual(y + height, 480)


if __name__ == "__main__":
    unittest.main()

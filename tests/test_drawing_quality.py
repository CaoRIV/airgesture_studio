from __future__ import annotations

import unittest

import numpy as np

from airgesture.core.smoothing import AdaptivePointSmoother, AdaptiveSmoothingConfig
from airgesture.drawing.canvas import CanvasConfig, DrawingCanvas
from airgesture.drawing.stroke_state import StrokeEndDebouncer


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


if __name__ == "__main__":
    unittest.main()

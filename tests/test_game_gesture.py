from __future__ import annotations

import unittest

import numpy as np

from airgesture.puzzle.gesture import PinchGesture, PinchGestureConfig
from airgesture.puzzle.hud import PUZZLE_PAPER, PUZZLE_YELLOW, draw_capture_hud, draw_play_hud


class PinchGestureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gesture = PinchGesture(
            config=PinchGestureConfig(
                pinch_threshold=40.0,
                release_threshold=60.0,
                missing_frame_tolerance=2,
            )
        )

    def test_uses_hysteresis_between_pinch_and_release(self) -> None:
        started = self.gesture.update((0, 0), (35, 0))
        middle = self.gesture.update((0, 0), (50, 0))
        released = self.gesture.update((0, 0), (65, 0))

        self.assertTrue(started.started)
        self.assertTrue(middle.active)
        self.assertTrue(released.released)

    def test_does_not_release_on_short_tracking_dropout(self) -> None:
        self.gesture.update((0, 0), (35, 0))

        first_missing = self.gesture.update(None, None)
        second_missing = self.gesture.update(None, None)
        recovered = self.gesture.update((0, 0), (35, 0))

        self.assertTrue(first_missing.active)
        self.assertTrue(second_missing.active)
        self.assertTrue(recovered.active)
        self.assertFalse(recovered.released)

    def test_releases_after_tracking_dropout_tolerance(self) -> None:
        self.gesture.update((0, 0), (35, 0))
        self.gesture.update(None, None)
        self.gesture.update(None, None)

        released = self.gesture.update(None, None)

        self.assertFalse(released.active)
        self.assertTrue(released.released)


class PuzzleCaptureHudTests(unittest.TestCase):
    def test_capture_hud_uses_pop_art_layout_without_fake_title_bar(self) -> None:
        frame = np.full((720, 1280, 3), (24, 28, 34), dtype=np.uint8)

        draw_capture_hud(
            frame,
            hand_count=0,
            capture_message="Show both hands in the camera",
            capture_progress=0.0,
            difficulty=3,
            fps=30.0,
        )

        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertTupleEqual(
            tuple(int(value) for value in frame[5, 640]),
            PUZZLE_PAPER,
        )
        self.assertTupleEqual(
            tuple(int(value) for value in frame[150, 150]),
            PUZZLE_YELLOW,
        )
        self.assertGreater(int(np.count_nonzero(frame)), 0)

    def test_play_hud_uses_matching_light_top_and_bottom_bars(self) -> None:
        frame = np.full((720, 1280, 3), (24, 28, 34), dtype=np.uint8)

        draw_play_hud(
            frame,
            elapsed_seconds=6.9,
            moves=0,
            pinch_active=False,
            selected_tile=None,
            difficulty=3,
        )

        self.assertTupleEqual(
            tuple(int(value) for value in frame[5, 640]),
            PUZZLE_PAPER,
        )
        self.assertTupleEqual(
            tuple(int(value) for value in frame[715, 640]),
            PUZZLE_PAPER,
        )


if __name__ == "__main__":
    unittest.main()

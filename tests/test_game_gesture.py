from __future__ import annotations

import unittest

from airgesture.puzzle.gesture import PinchGesture, PinchGestureConfig


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


if __name__ == "__main__":
    unittest.main()

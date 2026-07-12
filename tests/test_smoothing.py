from __future__ import annotations

import unittest

from airgesture.core.smoothing import (
    OneEuroConfig,
    OneEuroPointFilter,
    PointSmoother,
    SmoothingConfig,
)


class PointSmootherTests(unittest.TestCase):
    def test_holds_cursor_during_short_detection_dropout(self) -> None:
        smoother = PointSmoother(
            SmoothingConfig(alpha=0.5, missing_frame_tolerance=2)
        )

        self.assertEqual(smoother.update((100, 80)), (100, 80))
        self.assertEqual(smoother.update(None), (100, 80))
        self.assertEqual(smoother.update(None), (100, 80))
        self.assertIsNone(smoother.update(None))

    def test_recovers_cleanly_after_long_dropout(self) -> None:
        smoother = PointSmoother(
            SmoothingConfig(alpha=0.2, missing_frame_tolerance=1)
        )

        smoother.update((10, 10))
        smoother.update(None)
        smoother.update(None)

        self.assertEqual(smoother.update((500, 300)), (500, 300))


class OneEuroPointFilterTests(unittest.TestCase):
    def test_reduces_stationary_jitter(self) -> None:
        point_filter = OneEuroPointFilter(
            OneEuroConfig(min_cutoff=1.35, beta=0.45, max_jump=0.18)
        )
        raw_x_values = [0.49, 0.51] * 12
        filtered_x_values = []

        for index, x in enumerate(raw_x_values):
            filtered = point_filter.update((x, 0.5, 0.0), index / 30.0)
            filtered_x_values.append(filtered[0])

        filtered_tail = filtered_x_values[6:]
        self.assertLess(max(filtered_tail) - min(filtered_tail), 0.012)

    def test_limits_single_frame_outlier(self) -> None:
        config = OneEuroConfig(
            min_cutoff=1.35,
            beta=0.45,
            max_jump=0.10,
        )
        point_filter = OneEuroPointFilter(config)
        point_filter.update((0.2, 0.2, 0.0), 0.0)

        filtered = point_filter.update((0.9, 0.9, 0.0), 1.0 / 30.0)

        self.assertLess(filtered[0], 0.30)
        self.assertLess(filtered[1], 0.30)


if __name__ == "__main__":
    unittest.main()

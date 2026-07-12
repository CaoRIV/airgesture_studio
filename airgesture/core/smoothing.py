from __future__ import annotations

from dataclasses import dataclass
from math import hypot, pi
import time


@dataclass(frozen=True)
class SmoothingConfig:
    alpha: float = 0.35
    missing_frame_tolerance: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1]")
        if self.missing_frame_tolerance < 0:
            raise ValueError("missing_frame_tolerance cannot be negative")


class PointSmoother:
    """Exponential moving average smoother for fingertip coordinates."""

    def __init__(self, config: SmoothingConfig | None = None) -> None:
        self.config = config or SmoothingConfig()
        self._point: tuple[float, float] | None = None
        self._missing_frames = 0

    def reset(self) -> None:
        self._point = None
        self._missing_frames = 0

    def update(self, point: tuple[int, int] | None) -> tuple[int, int] | None:
        if point is None:
            self._missing_frames += 1
            if (
                self._point is not None
                and self._missing_frames <= self.config.missing_frame_tolerance
            ):
                return int(round(self._point[0])), int(round(self._point[1]))
            self.reset()
            return None

        self._missing_frames = 0

        if self._point is None:
            self._point = float(point[0]), float(point[1])
            return point

        alpha = self.config.alpha
        smoothed_x = self._point[0] * (1.0 - alpha) + point[0] * alpha
        smoothed_y = self._point[1] * (1.0 - alpha) + point[1] * alpha
        self._point = smoothed_x, smoothed_y
        return int(round(smoothed_x)), int(round(smoothed_y))


@dataclass(frozen=True)
class AdaptiveSmoothingConfig:
    """Velocity-based smoothing parameters for drawing input."""

    slow_alpha: float
    fast_alpha: float
    slow_speed: float
    fast_speed: float
    speed_smoothing_alpha: float
    missing_frame_tolerance: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.slow_alpha <= self.fast_alpha <= 1.0:
            raise ValueError("adaptive alpha values must satisfy 0 < slow <= fast <= 1")
        if self.slow_speed < 0.0 or self.fast_speed <= self.slow_speed:
            raise ValueError("fast_speed must be greater than non-negative slow_speed")
        if not 0.0 < self.speed_smoothing_alpha <= 1.0:
            raise ValueError("speed_smoothing_alpha must be in the range (0, 1]")
        if self.missing_frame_tolerance < 0:
            raise ValueError("missing_frame_tolerance cannot be negative")


class AdaptivePointSmoother:
    """Smooth slow movement strongly and follow fast movement responsively."""

    def __init__(self, config: AdaptiveSmoothingConfig) -> None:
        self.config = config
        self._point: tuple[float, float] | None = None
        self._raw_point: tuple[int, int] | None = None
        self._timestamp_seconds: float | None = None
        self._speed = 0.0
        self._alpha = config.slow_alpha
        self._missing_frames = 0

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def alpha(self) -> float:
        return self._alpha

    def reset(self) -> None:
        self._point = None
        self._raw_point = None
        self._timestamp_seconds = None
        self._speed = 0.0
        self._alpha = self.config.slow_alpha
        self._missing_frames = 0

    def update(
        self,
        point: tuple[int, int] | None,
        timestamp_seconds: float | None = None,
    ) -> tuple[int, int] | None:
        now = time.perf_counter() if timestamp_seconds is None else timestamp_seconds
        if point is None:
            self._missing_frames += 1
            if (
                self._point is not None
                and self._missing_frames <= self.config.missing_frame_tolerance
            ):
                return int(round(self._point[0])), int(round(self._point[1]))
            self.reset()
            return None

        self._missing_frames = 0
        if self._point is None or self._raw_point is None or self._timestamp_seconds is None:
            self._point = float(point[0]), float(point[1])
            self._raw_point = point
            self._timestamp_seconds = now
            return point

        dt = max(now - self._timestamp_seconds, 1.0 / 120.0)
        instant_speed = hypot(
            point[0] - self._raw_point[0],
            point[1] - self._raw_point[1],
        ) / dt
        # React to acceleration immediately, then decay speed gradually when slowing down.
        speed_alpha = (
            1.0
            if instant_speed > self._speed
            else self.config.speed_smoothing_alpha
        )
        self._speed += speed_alpha * (instant_speed - self._speed)
        self._alpha = self._alpha_for_speed(self._speed)

        smoothed_x = self._point[0] + self._alpha * (point[0] - self._point[0])
        smoothed_y = self._point[1] + self._alpha * (point[1] - self._point[1])
        self._point = smoothed_x, smoothed_y
        self._raw_point = point
        self._timestamp_seconds = now
        return int(round(smoothed_x)), int(round(smoothed_y))

    def _alpha_for_speed(self, speed: float) -> float:
        speed_range = self.config.fast_speed - self.config.slow_speed
        ratio = (speed - self.config.slow_speed) / speed_range
        ratio = min(max(ratio, 0.0), 1.0)
        # Smoothstep avoids an abrupt feel around either speed threshold.
        ratio = ratio * ratio * (3.0 - 2.0 * ratio)
        return self.config.slow_alpha + ratio * (
            self.config.fast_alpha - self.config.slow_alpha
        )


@dataclass(frozen=True)
class OneEuroConfig:
    """Parameters for adaptive low-pass filtering of normalized landmarks."""

    min_cutoff: float = 1.7
    beta: float = 0.35
    derivative_cutoff: float = 1.0
    max_jump: float = 0.22

    def __post_init__(self) -> None:
        if self.min_cutoff <= 0.0 or self.derivative_cutoff <= 0.0:
            raise ValueError("filter cutoff values must be positive")
        if self.beta < 0.0:
            raise ValueError("beta cannot be negative")
        if self.max_jump <= 0.0:
            raise ValueError("max_jump must be positive")


class OneEuroPointFilter:
    """Smooth a 3D point while preserving responsiveness during fast motion."""

    def __init__(self, config: OneEuroConfig | None = None) -> None:
        self.config = config or OneEuroConfig()
        self._point: tuple[float, float, float] | None = None
        self._derivative = (0.0, 0.0, 0.0)
        self._timestamp_seconds: float | None = None

    def reset(self) -> None:
        self._point = None
        self._derivative = (0.0, 0.0, 0.0)
        self._timestamp_seconds = None

    def update(
        self,
        point: tuple[float, float, float],
        timestamp_seconds: float,
    ) -> tuple[float, float, float]:
        if self._point is None or self._timestamp_seconds is None:
            self._point = point
            self._timestamp_seconds = timestamp_seconds
            return point

        dt = max(timestamp_seconds - self._timestamp_seconds, 1.0 / 120.0)
        limited_point = self._limit_jump(point)
        raw_derivative = tuple(
            (value - previous) / dt
            for value, previous in zip(limited_point, self._point)
        )
        derivative_alpha = self._alpha(self.config.derivative_cutoff, dt)
        derivative = tuple(
            previous + derivative_alpha * (value - previous)
            for value, previous in zip(raw_derivative, self._derivative)
        )
        speed = sum(value * value for value in derivative) ** 0.5
        cutoff = self.config.min_cutoff + self.config.beta * speed
        point_alpha = self._alpha(cutoff, dt)
        filtered = tuple(
            previous + point_alpha * (value - previous)
            for value, previous in zip(limited_point, self._point)
        )

        self._point = filtered
        self._derivative = derivative
        self._timestamp_seconds = timestamp_seconds
        return filtered

    def _limit_jump(
        self,
        point: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        assert self._point is not None
        delta_x = point[0] - self._point[0]
        delta_y = point[1] - self._point[1]
        distance = (delta_x * delta_x + delta_y * delta_y) ** 0.5
        if distance <= self.config.max_jump or distance == 0.0:
            return point

        scale = self.config.max_jump / distance
        return (
            self._point[0] + delta_x * scale,
            self._point[1] + delta_y * scale,
            self._point[2] + (point[2] - self._point[2]) * scale,
        )

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        time_constant = 1.0 / (2.0 * pi * max(cutoff, 0.001))
        return 1.0 / (1.0 + time_constant / dt)

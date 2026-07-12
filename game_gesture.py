from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class PinchState:
    active: bool
    started: bool
    released: bool
    distance: float | None


@dataclass(frozen=True)
class PinchGestureConfig:
    pinch_threshold: float = 48.0
    release_threshold: float = 72.0
    missing_frame_tolerance: int = 2

    def __post_init__(self) -> None:
        if self.pinch_threshold <= 0.0:
            raise ValueError("pinch_threshold must be positive")
        if self.release_threshold <= self.pinch_threshold:
            raise ValueError("release_threshold must be greater than pinch_threshold")
        if self.missing_frame_tolerance < 0:
            raise ValueError("missing_frame_tolerance cannot be negative")


class PinchGesture:
    """Hysteresis-based pinch detector using thumb and index fingertips."""

    def __init__(
        self,
        pinch_threshold: float = 48.0,
        release_threshold: float = 72.0,
        missing_frame_tolerance: int = 2,
        config: PinchGestureConfig | None = None,
    ) -> None:
        resolved_config = config or PinchGestureConfig(
            pinch_threshold=pinch_threshold,
            release_threshold=release_threshold,
            missing_frame_tolerance=missing_frame_tolerance,
        )
        self.pinch_threshold = resolved_config.pinch_threshold
        self.release_threshold = resolved_config.release_threshold
        self.missing_frame_tolerance = resolved_config.missing_frame_tolerance
        self._active = False
        self._missing_frames = 0

    def update(
        self,
        thumb_tip: tuple[int, int] | None,
        index_tip: tuple[int, int] | None,
    ) -> PinchState:
        if thumb_tip is None or index_tip is None:
            self._missing_frames += 1
            if self._active and self._missing_frames <= self.missing_frame_tolerance:
                return PinchState(True, False, False, None)
            was_active = self._active
            self._active = False
            return PinchState(False, False, was_active, None)

        self._missing_frames = 0
        distance = hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
        started = False
        released = False

        if not self._active and distance <= self.pinch_threshold:
            self._active = True
            started = True
        elif self._active and distance >= self.release_threshold:
            self._active = False
            released = True

        return PinchState(self._active, started, released, distance)

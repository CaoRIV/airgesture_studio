from __future__ import annotations


class StrokeEndDebouncer:
    """Delay stroke finalization across brief pinch or tracking interruptions."""

    def __init__(self, delay_seconds: float, grace_frames: int) -> None:
        if delay_seconds < 0.0:
            raise ValueError("delay_seconds cannot be negative")
        if grace_frames < 0:
            raise ValueError("grace_frames cannot be negative")
        self.delay_seconds = delay_seconds
        self.grace_frames = grace_frames
        self.reset()

    @property
    def waiting(self) -> bool:
        return self._release_started_at is not None

    def reset(self) -> None:
        self._release_started_at: float | None = None
        self._inactive_frames = 0

    def mark_active(self) -> None:
        self.reset()

    def should_finalize(self, now_seconds: float, has_open_stroke: bool) -> bool:
        if not has_open_stroke:
            self.reset()
            return False

        self._inactive_frames += 1
        if self._release_started_at is None:
            self._release_started_at = now_seconds

        delay_elapsed = now_seconds - self._release_started_at >= self.delay_seconds
        grace_elapsed = self._inactive_frames > self.grace_frames
        return delay_elapsed and grace_elapsed

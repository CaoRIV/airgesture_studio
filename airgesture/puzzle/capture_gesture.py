from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot


class CapturePhase(Enum):
    WAITING = "Waiting"
    CENTER = "Center"
    SPREAD = "Spread"
    HOLD = "Hold"


@dataclass(frozen=True)
class CaptureGestureResult:
    captured: bool
    phase: CapturePhase
    message: str
    points: list[tuple[int, int]]
    bounds: tuple[int, int, int, int] | None
    progress: float


@dataclass(frozen=True)
class CaptureGestureConfig:
    stable_frames_required: int = 5
    centered_radius_ratio: float = 0.30
    cluster_radius_ratio: float = 0.26
    spread_ratio_required: float = 0.34
    stable_motion_threshold: float = 64.0
    dynamic_motion_ratio: float = 0.08

    def __post_init__(self) -> None:
        if self.stable_frames_required < 1:
            raise ValueError("stable_frames_required must be at least 1")
        ratio_values = (
            self.centered_radius_ratio,
            self.cluster_radius_ratio,
            self.spread_ratio_required,
            self.dynamic_motion_ratio,
        )
        if any(not 0.0 < value <= 1.0 for value in ratio_values):
            raise ValueError("capture ratios must be in the range (0, 1]")
        if self.stable_motion_threshold < 0.0:
            raise ValueError("stable_motion_threshold cannot be negative")


class TwoHandSpreadCaptureGesture:
    """Detects two hands framing a shot, then holding still briefly."""

    THUMB_TIP = 4
    INDEX_TIP = 8

    def __init__(
        self,
        stable_frames_required: int = 5,
        centered_radius_ratio: float = 0.30,
        cluster_radius_ratio: float = 0.26,
        spread_ratio_required: float = 0.34,
        stable_motion_threshold: float = 64.0,
        dynamic_motion_ratio: float = 0.08,
        config: CaptureGestureConfig | None = None,
    ) -> None:
        resolved_config = config or CaptureGestureConfig(
            stable_frames_required=stable_frames_required,
            centered_radius_ratio=centered_radius_ratio,
            cluster_radius_ratio=cluster_radius_ratio,
            spread_ratio_required=spread_ratio_required,
            stable_motion_threshold=stable_motion_threshold,
            dynamic_motion_ratio=dynamic_motion_ratio,
        )
        self.stable_frames_required = resolved_config.stable_frames_required
        self.centered_radius_ratio = resolved_config.centered_radius_ratio
        self.cluster_radius_ratio = resolved_config.cluster_radius_ratio
        self.spread_ratio_required = resolved_config.spread_ratio_required
        self.stable_motion_threshold = resolved_config.stable_motion_threshold
        self.dynamic_motion_ratio = resolved_config.dynamic_motion_ratio
        self._phase = CapturePhase.WAITING
        self._stable_frames = 0
        self._last_bounds: tuple[int, int, int, int] | None = None
        self._capture_latched = False

    def reset(self) -> None:
        self._phase = CapturePhase.WAITING
        self._stable_frames = 0
        self._last_bounds = None
        self._capture_latched = False

    def update(self, results, frame_shape) -> CaptureGestureResult:
        points = self._finger_points(results, frame_shape)
        if len(points) < 4:
            self.reset()
            return CaptureGestureResult(
                False,
                CapturePhase.WAITING,
                "Show both hands in the camera",
                points,
                None,
                0.0,
            )

        bounds = self._bounds(points)
        frame_height, frame_width = frame_shape[:2]
        frame_min = min(frame_width, frame_height)
        center = self._center(points)
        frame_center = (frame_width / 2.0, frame_height / 2.0)
        center_distance = self._distance(center, frame_center)
        cluster_radius = max(self._distance(point, center) for point in points)
        spread_ratio = max(bounds[2] / frame_width, bounds[3] / frame_height)

        centered = center_distance <= frame_min * self.centered_radius_ratio
        clustered = cluster_radius <= frame_min * self.cluster_radius_ratio
        spread_enough = spread_ratio >= self.spread_ratio_required

        captured = False
        if spread_enough:
            if self._phase != CapturePhase.HOLD:
                self._phase = CapturePhase.HOLD
                self._stable_frames = 0

            if self._is_stable(bounds, frame_shape):
                self._stable_frames += 1
                if self._stable_frames >= self.stable_frames_required and not self._capture_latched:
                    captured = True
                    self._capture_latched = True
            else:
                self._stable_frames = 0
        elif centered and clustered:
            self._phase = CapturePhase.CENTER
        else:
            self._phase = CapturePhase.SPREAD
            self._stable_frames = 0

        self._last_bounds = bounds
        progress = self._progress(spread_ratio, spread_enough)
        return CaptureGestureResult(
            captured,
            self._phase,
            self._message(self._phase),
            points,
            bounds,
            progress,
        )

    def _finger_points(self, results, frame_shape) -> list[tuple[int, int]]:
        if not results.hand_landmarks or len(results.hand_landmarks) < 2:
            return []

        frame_height, frame_width = frame_shape[:2]
        points = []
        for hand_landmarks in results.hand_landmarks[:2]:
            for landmark_index in (self.THUMB_TIP, self.INDEX_TIP):
                landmark = hand_landmarks[landmark_index]
                x = min(max(int(landmark.x * frame_width), 0), frame_width - 1)
                y = min(max(int(landmark.y * frame_height), 0), frame_height - 1)
                points.append((x, y))
        return points

    def _bounds(self, points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return min_x, min_y, max_x - min_x, max_y - min_y

    def _center(self, points: list[tuple[int, int]]) -> tuple[float, float]:
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    def _is_stable(self, bounds: tuple[int, int, int, int], frame_shape) -> bool:
        if self._last_bounds is None:
            return False

        x, y, width, height = bounds
        last_x, last_y, last_width, last_height = self._last_bounds
        motion = (
            abs(x - last_x)
            + abs(y - last_y)
            + abs(width - last_width)
            + abs(height - last_height)
        )
        frame_height, frame_width = frame_shape[:2]
        dynamic_threshold = max(
            self.stable_motion_threshold,
            min(frame_width, frame_height) * self.dynamic_motion_ratio,
        )
        return motion <= dynamic_threshold

    def _distance(self, first, second) -> float:
        return hypot(first[0] - second[0], first[1] - second[1])

    def _progress(self, spread_ratio: float, spread_enough: bool) -> float:
        spread_progress = min(1.0, spread_ratio / self.spread_ratio_required)
        if not spread_enough:
            return spread_progress * 0.70
        hold_progress = min(1.0, self._stable_frames / max(1, self.stable_frames_required))
        return 0.70 + hold_progress * 0.30

    def _message(self, phase: CapturePhase) -> str:
        if phase == CapturePhase.CENTER:
            return "Good. Open both hands outward to frame the shot"
        if phase == CapturePhase.SPREAD:
            return "Open both hands wider, then hold still"
        if phase == CapturePhase.HOLD:
            return "Hold still: auto capture is armed"
        return "Show both hands in the camera"

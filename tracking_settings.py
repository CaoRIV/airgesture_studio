from __future__ import annotations

from camera import CameraConfig
from game_gesture import PinchGestureConfig
from hand_tracker import HandTrackerConfig
from smoothing import OneEuroConfig, SmoothingConfig


CAMERA_CONFIG = CameraConfig(
    camera_index=0,
    mirror=True,
    width=1280,
    height=720,
    fps=30,
)

AIR_DRAWING_TRACKER_CONFIG = HandTrackerConfig(
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_hand_presence_confidence=0.50,
    min_tracking_confidence=0.55,
    filter_reset_after_missing_frames=5,
    landmark_filter=OneEuroConfig(
        min_cutoff=1.35,
        beta=0.45,
        derivative_cutoff=1.0,
        max_jump=0.18,
    ),
)

PUZZLE_TRACKER_CONFIG = HandTrackerConfig(
    max_num_hands=2,
    min_detection_confidence=0.50,
    min_hand_presence_confidence=0.40,
    min_tracking_confidence=0.45,
    filter_reset_after_missing_frames=4,
    landmark_filter=OneEuroConfig(
        min_cutoff=1.65,
        beta=0.55,
        derivative_cutoff=1.0,
        max_jump=0.22,
    ),
)

AIR_DRAWING_CURSOR_CONFIG = SmoothingConfig(
    alpha=0.55,
    missing_frame_tolerance=2,
)

PUZZLE_CURSOR_CONFIG = SmoothingConfig(
    alpha=0.60,
    missing_frame_tolerance=2,
)

AIR_DRAWING_PINCH_CONFIG = PinchGestureConfig(
    pinch_threshold=46.0,
    release_threshold=68.0,
    missing_frame_tolerance=2,
)

PUZZLE_PINCH_CONFIG = PinchGestureConfig(
    pinch_threshold=48.0,
    release_threshold=72.0,
    missing_frame_tolerance=2,
)


def calibration_tracker_config(required_hands: int) -> HandTrackerConfig:
    return HandTrackerConfig(
        max_num_hands=max(1, required_hands),
        min_detection_confidence=0.55,
        min_hand_presence_confidence=0.40,
        min_tracking_confidence=0.45,
        filter_reset_after_missing_frames=4,
        landmark_filter=OneEuroConfig(
            min_cutoff=1.5,
            beta=0.45,
            derivative_cutoff=1.0,
            max_jump=0.22,
        ),
    )

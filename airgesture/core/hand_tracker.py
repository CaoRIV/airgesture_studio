from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
import time

import cv2

from airgesture.paths import BUNDLED_MODELS_DIR, CACHE_DIR

_matplotlib_cache_dir = CACHE_DIR / "matplotlib"
try:
    _matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
except OSError:
    _matplotlib_cache_dir = Path(tempfile.gettempdir()) / "AirGesture" / "matplotlib"
    _matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_matplotlib_cache_dir)

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

from airgesture.core.smoothing import OneEuroConfig, OneEuroPointFilter
from airgesture.errors import HandTrackingError


@dataclass(frozen=True)
class HandTrackerConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    filter_reset_after_missing_frames: int = 5
    landmark_filter: OneEuroConfig = field(default_factory=OneEuroConfig)
    model_asset_path: str = str(
        BUNDLED_MODELS_DIR / "hand_landmarker.task"
    )

    def __post_init__(self) -> None:
        if self.max_num_hands < 1:
            raise ValueError("max_num_hands must be at least 1")
        confidence_values = (
            self.min_detection_confidence,
            self.min_hand_presence_confidence,
            self.min_tracking_confidence,
        )
        if any(not 0.0 <= value <= 1.0 for value in confidence_values):
            raise ValueError("confidence values must be in the range [0, 1]")
        if self.filter_reset_after_missing_frames < 0:
            raise ValueError("filter_reset_after_missing_frames cannot be negative")


class HandTracker:
    """MediaPipe Hands wrapper for detecting and drawing hand landmarks."""

    def __init__(self, config: HandTrackerConfig | None = None) -> None:
        self.config = config or HandTrackerConfig()
        model_path = Path(self.config.model_asset_path)
        if not model_path.exists():
            raise HandTrackingError(
                "The bundled hand-tracking model is missing. "
                "Reinstall AirGesture Studio."
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.max_num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise HandTrackingError(
                "Could not initialize hand tracking. The model may be damaged "
                "or incompatible with this installation."
            ) from exc
        self._connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self._started_at = time.perf_counter()
        self._last_timestamp_ms = -1
        self._filters: dict[str, list[OneEuroPointFilter]] = {}
        self._missing_result_frames = 0

    def detect(self, frame) -> HandLandmarkerResult:
        try:
            timestamp_ms = self._next_timestamp_ms()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._landmarker.detect_for_video(image, timestamp_ms)
            return self._smooth_result(result, timestamp_ms / 1000.0)
        except HandTrackingError:
            raise
        except Exception as exc:
            raise HandTrackingError(
                "Hand tracking stopped while processing the camera image."
            ) from exc

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int((time.perf_counter() - self._started_at) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _smooth_result(
        self,
        result: HandLandmarkerResult,
        timestamp_seconds: float,
    ) -> HandLandmarkerResult:
        if not result.hand_landmarks:
            self._missing_result_frames += 1
            if self._missing_result_frames > self.config.filter_reset_after_missing_frames:
                self._filters.clear()
            return result

        self._missing_result_frames = 0
        active_keys: set[str] = set()
        smoothed_hands = []
        for hand_index, landmarks in enumerate(result.hand_landmarks):
            hand_key = self._hand_key(result, hand_index)
            active_keys.add(hand_key)
            filters = self._filters.get(hand_key)
            if filters is None or len(filters) != len(landmarks):
                filters = [
                    OneEuroPointFilter(self.config.landmark_filter)
                    for _ in landmarks
                ]
                self._filters[hand_key] = filters

            smoothed_landmarks = []
            for landmark, point_filter in zip(landmarks, filters):
                x, y, z = point_filter.update(
                    (float(landmark.x), float(landmark.y), float(landmark.z)),
                    timestamp_seconds,
                )
                smoothed_landmarks.append(
                    NormalizedLandmark(
                        x=x,
                        y=y,
                        z=z,
                        visibility=landmark.visibility,
                        presence=landmark.presence,
                        name=landmark.name,
                    )
                )
            smoothed_hands.append(smoothed_landmarks)

        self._filters = {
            key: filters
            for key, filters in self._filters.items()
            if key in active_keys
        }
        return HandLandmarkerResult(
            handedness=result.handedness,
            hand_landmarks=smoothed_hands,
            hand_world_landmarks=result.hand_world_landmarks,
        )

    @staticmethod
    def _hand_key(result: HandLandmarkerResult, hand_index: int) -> str:
        if hand_index < len(result.handedness) and result.handedness[hand_index]:
            category = result.handedness[hand_index][0]
            label = category.category_name or category.display_name
            if label:
                return label.lower()
        return f"hand-{hand_index}"

    def get_landmark_pixel(
        self,
        results,
        frame_shape,
        landmark_index: int,
        hand_index: int = 0,
    ) -> tuple[int, int] | None:
        if not results.hand_landmarks or hand_index >= len(results.hand_landmarks):
            return None

        hand_landmarks = results.hand_landmarks[hand_index]
        if landmark_index >= len(hand_landmarks):
            return None

        frame_height, frame_width = frame_shape[:2]
        landmark = hand_landmarks[landmark_index]
        x = min(max(int(landmark.x * frame_width), 0), frame_width - 1)
        y = min(max(int(landmark.y * frame_height), 0), frame_height - 1)
        return x, y

    def draw_landmarks(self, frame, results) -> None:
        if not results.hand_landmarks:
            return

        frame_height, frame_width = frame.shape[:2]
        for hand_landmarks in results.hand_landmarks:
            points = [
                (int(landmark.x * frame_width), int(landmark.y * frame_height))
                for landmark in hand_landmarks
            ]

            for connection in self._connections:
                cv2.line(
                    frame,
                    points[connection.start],
                    points[connection.end],
                    (80, 220, 120),
                    2,
                )

            for point in points:
                cv2.circle(frame, point, 4, (40, 120, 255), -1)
                cv2.circle(frame, point, 5, (255, 255, 255), 1)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import cv2

_matplotlib_cache_dir = Path(__file__).resolve().parent / ".cache" / "matplotlib"
_matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache_dir))

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


@dataclass
class HandTrackerConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_asset_path: str = str(
        Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
    )


class HandTracker:
    """MediaPipe Hands wrapper for detecting and drawing hand landmarks."""

    def __init__(self, config: HandTrackerConfig | None = None) -> None:
        self.config = config or HandTrackerConfig()
        model_path = Path(self.config.model_asset_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand landmark model not found: {model_path}"
            )

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.config.max_num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def detect(self, frame) -> Any:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect(image)

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

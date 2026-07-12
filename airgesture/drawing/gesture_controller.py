from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GestureMode(Enum):
    IDLE = "Idle"
    DRAW = "Draw"
    MOVE = "Move"


@dataclass(frozen=True)
class FingerState:
    index: bool
    middle: bool
    ring: bool
    pinky: bool


@dataclass(frozen=True)
class GestureState:
    mode: GestureMode
    index_tip: tuple[int, int] | None
    fingers: FingerState | None


class GestureController:
    """Converts hand landmarks into simple drawing modes."""

    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18

    def analyze(self, results, frame_shape) -> GestureState:
        if not results.hand_landmarks:
            return GestureState(GestureMode.IDLE, None, None)

        hand_landmarks = results.hand_landmarks[0]
        index_tip = self._to_pixel(hand_landmarks[self.INDEX_TIP], frame_shape)
        fingers = FingerState(
            index=self._is_finger_up(hand_landmarks, self.INDEX_TIP, self.INDEX_PIP),
            middle=self._is_finger_up(hand_landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP),
            ring=self._is_finger_up(hand_landmarks, self.RING_TIP, self.RING_PIP),
            pinky=self._is_finger_up(hand_landmarks, self.PINKY_TIP, self.PINKY_PIP),
        )

        if fingers.index and not fingers.middle and not fingers.ring and not fingers.pinky:
            mode = GestureMode.DRAW
        elif fingers.index and fingers.middle and not fingers.ring and not fingers.pinky:
            mode = GestureMode.MOVE
        else:
            mode = GestureMode.IDLE

        return GestureState(mode, index_tip, fingers)

    def _is_finger_up(self, hand_landmarks, tip_index: int, pip_index: int) -> bool:
        return hand_landmarks[tip_index].y < hand_landmarks[pip_index].y

    def _to_pixel(self, landmark, frame_shape) -> tuple[int, int]:
        frame_height, frame_width = frame_shape[:2]
        x = min(max(int(landmark.x * frame_width), 0), frame_width - 1)
        y = min(max(int(landmark.y * frame_height), 0), frame_height - 1)
        return x, y

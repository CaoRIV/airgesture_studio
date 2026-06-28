from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from camera import Camera, CameraConfig
from hand_tracker import HandTracker, HandTrackerConfig


WINDOW_NAME = "AirGesture Calibration"


@dataclass(frozen=True)
class CalibrationConfig:
    title: str
    required_hands: int
    min_brightness: float = 55.0
    max_brightness: float = 220.0


def run_calibration(config: CalibrationConfig) -> bool:
    camera = Camera(CameraConfig(camera_index=0, mirror=True, width=1280, height=720, fps=30))
    if not camera.open():
        print("Error: Could not open webcam for calibration.")
        return False

    hand_tracker_config = HandTrackerConfig(
        max_num_hands=max(1, config.required_hands),
        min_detection_confidence=0.55,
        min_hand_presence_confidence=0.40,
        min_tracking_confidence=0.40,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        with HandTracker(hand_tracker_config) as hand_tracker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Error: Could not read webcam frame during calibration.")
                    return False

                results = hand_tracker.detect(frame)
                hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0
                brightness = average_brightness(frame)
                ready = is_ready(config, hand_count, brightness)

                hand_tracker.draw_landmarks(frame, results)
                draw_calibration_hud(frame, config, hand_count, brightness, ready)
                cv2.imshow(WINDOW_NAME, frame)

                key_code = cv2.waitKey(1) & 0xFF
                if key_code in (27, ord("q"), ord("Q")):
                    return False
                if key_code == ord(" "):
                    return ready
                if key_code in (13, 10):
                    return True
    finally:
        camera.release()
        cv2.destroyWindow(WINDOW_NAME)


def average_brightness(frame) -> float:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(grayscale))


def is_ready(config: CalibrationConfig, hand_count: int, brightness: float) -> bool:
    return (
        hand_count >= config.required_hands
        and config.min_brightness <= brightness <= config.max_brightness
    )


def draw_calibration_hud(
    frame,
    config: CalibrationConfig,
    hand_count: int,
    brightness: float,
    ready: bool,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 116), (10, 12, 18), -1)
    cv2.rectangle(overlay, (0, frame.shape[0] - 92), (frame.shape[1], frame.shape[0]), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

    status_text = "Ready" if ready else "Adjust camera / hands"
    status_color = (0, 230, 140) if ready else (0, 180, 255)

    put_text(frame, "Calibration", (28, 44), 1.0, (245, 247, 250), 2)
    put_text(frame, config.title, (28, 86), 0.72, (200, 210, 224), 1)
    put_text(frame, status_text, (frame.shape[1] - 260, 48), 0.82, status_color, 2)

    draw_metric(
        frame,
        "Hands",
        f"{hand_count}/{config.required_hands}",
        hand_count >= config.required_hands,
        (28, frame.shape[0] - 54),
    )
    draw_metric(
        frame,
        "Brightness",
        f"{brightness:05.1f}",
        config.min_brightness <= brightness <= config.max_brightness,
        (300, frame.shape[0] - 54),
    )
    put_text(
        frame,
        "Space: Continue when ready   Enter: Skip calibration   Q/Esc: Cancel",
        (560, frame.shape[0] - 42),
        0.58,
        (225, 230, 238),
        1,
    )


def draw_metric(frame, label: str, value: str, ok: bool, origin: tuple[int, int]) -> None:
    color = (0, 230, 140) if ok else (0, 180, 255)
    x, y = origin
    cv2.circle(frame, (x, y - 7), 9, color, -1, cv2.LINE_AA)
    put_text(frame, f"{label}: {value}", (x + 24, y), 0.66, (245, 247, 250), 2)


def put_text(frame, text: str, origin: tuple[int, int], scale: float, color, thickness: int) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

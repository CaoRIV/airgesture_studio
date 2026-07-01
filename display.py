from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 1280
    height: int = 720
    background_color: tuple[int, int, int] = (18, 20, 24)
    border_color: tuple[int, int, int] = (92, 180, 255)


def fit_frame_to_display(frame, config: DisplayConfig):
    """Resize a frame without distortion and center it on a 16:9 canvas."""
    frame_height, frame_width = frame.shape[:2]
    scale = min(config.width / frame_width, config.height / frame_height)
    resized_width = int(frame_width * scale)
    resized_height = int(frame_height * scale)

    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (config.height, config.width, 3),
        config.background_color,
        dtype=np.uint8,
    )
    x = (config.width - resized_width) // 2
    y = (config.height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas, (x, y, resized_width, resized_height)


def frame_point_to_display(
    point: tuple[int, int] | None,
    frame_shape,
    frame_bounds,
) -> tuple[int, int] | None:
    if point is None:
        return None

    frame_height, frame_width = frame_shape[:2]
    x, y, display_width, display_height = frame_bounds
    scale_x = display_width / frame_width
    scale_y = display_height / frame_height
    return int(x + point[0] * scale_x), int(y + point[1] * scale_y)


def draw_app_overlay(
    display_frame,
    frame_bounds,
    hand_detected: bool,
    mode: str,
    fps: float,
    detected_symbol: str | None = None,
) -> None:
    x, y, width, height = frame_bounds

    cv2.rectangle(
        display_frame,
        (x, y),
        (x + width - 1, y + height - 1),
        (92, 180, 255),
        2,
    )

    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 74), (12, 14, 18), -1)
    cv2.addWeighted(overlay, 0.78, display_frame, 0.22, 0, display_frame)

    if mode == "Draw":
        status = "Mode: Draw"
        status_color = (90, 230, 140)
    elif mode == "Move":
        status = "Mode: Move"
        status_color = (80, 170, 255)
    else:
        status = "Mode: Idle" if hand_detected else "Hand: No"
        status_color = (180, 190, 205)

    cv2.putText(
        display_frame,
        "Hand Gesture Air Drawing",
        (28, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )
    if detected_symbol is not None:
        cv2.putText(
            display_frame,
            f"Phat hien: {detected_symbol}",
            (430, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (80, 240, 150),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        display_frame,
        status,
        (display_frame.shape[1] - 390, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_frame,
        f"FPS: {fps:04.1f}",
        (display_frame.shape[1] - 210, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_frame,
        "Pinch: Draw/Erase   2 fingers: Move/Select toolbar   C: Clear   Q / Esc: Exit",
        (28, display_frame.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 224, 230),
        1,
        cv2.LINE_AA,
    )

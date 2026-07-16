from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from airgesture.ui import theme as ui


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 1280
    height: int = 720
    background_color: tuple[int, int, int] = ui.BG
    border_color: tuple[int, int, int] = ui.CYAN


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
    recognition_suggestions: tuple[tuple[str, float], ...] = (),
    notification_message: str | None = None,
    notification_is_error: bool = False,
) -> None:
    x, y, width, height = frame_bounds
    layout = ui.layout_for(display_frame)

    _draw_frame_border(display_frame, (x, y, width, height))
    ui.blend_rect(display_frame, layout.point(0, 0), layout.point(1280, 78), (10, 13, 20), 0.82)
    cv2.line(display_frame, layout.point(0, 78), layout.point(1280, 78), ui.BORDER_SOFT, layout.px(1), cv2.LINE_AA)

    if mode == "Draw":
        status = "MODE DRAW"
        status_color = ui.GREEN
    elif mode == "Move":
        status = "MODE MOVE"
        status_color = ui.CYAN
    else:
        status = "MODE IDLE"
        status_color = ui.TEXT_DIM

    ui.put_text(
        display_frame,
        "Air Drawing",
        layout.point(28, 46),
        layout.font(0.94),
        ui.TEXT,
        layout.px(2),
    )
    ui.put_text(
        display_frame,
        "Pinch to draw. Release to clean or detect symbols.",
        layout.point(30, 68),
        layout.font(0.42),
        ui.TEXT_MUTED,
        layout.px(1),
    )
    if detected_symbol is not None:
        ui.chip(
            display_frame,
            layout.rect(418, 22, 350, 34),
            f"DETECTED: {detected_symbol}",
            color=ui.GREEN,
            active=True,
        )
    elif recognition_suggestions:
        suggestion_text = "  ".join(
            f"{symbol} {confidence * 100:.0f}%"
            for symbol, confidence in recognition_suggestions
        )
        ui.chip(
            display_frame,
            layout.rect(418, 22, 350, 34),
            f"SUGGESTIONS: {suggestion_text}",
            color=ui.YELLOW,
            active=True,
        )
    ui.chip(
        display_frame,
        layout.rect(796, 22, 142, 34),
        status,
        color=status_color,
        active=mode in ("Draw", "Move"),
    )
    ui.chip(
        display_frame,
        layout.rect(948, 22, 126, 34),
        "HAND OK" if hand_detected else "HAND --",
        color=ui.GREEN if hand_detected else ui.YELLOW,
        active=hand_detected,
    )
    ui.chip(
        display_frame,
        layout.rect(1084, 22, 150, 34),
        f"FPS {fps:04.1f}",
        color=ui.CYAN,
        active=False,
    )
    _draw_bottom_help(display_frame)
    if notification_message:
        ui.chip(
            display_frame,
            layout.rect(28, 612, 1224, 38),
            notification_message,
            color=ui.RED if notification_is_error else ui.GREEN,
            active=True,
        )


def _draw_frame_border(display_frame, bounds: tuple[int, int, int, int]) -> None:
    x, y, width, height = bounds
    cv2.rectangle(display_frame, (x, y), (x + width - 1, y + height - 1), ui.BORDER_SOFT, 1)
    corner = 34
    for start, end in [
        ((x, y), (x + corner, y)),
        ((x, y), (x, y + corner)),
        ((x + width - corner, y), (x + width - 1, y)),
        ((x + width - 1, y), (x + width - 1, y + corner)),
        ((x, y + height - 1), (x + corner, y + height - 1)),
        ((x, y + height - corner), (x, y + height - 1)),
        ((x + width - corner, y + height - 1), (x + width - 1, y + height - 1)),
        ((x + width - 1, y + height - corner), (x + width - 1, y + height - 1)),
    ]:
        cv2.line(display_frame, start, end, ui.CYAN, 2, cv2.LINE_AA)


def _draw_bottom_help(display_frame) -> None:
    layout = ui.layout_for(display_frame)
    ui.blend_rect(display_frame, layout.point(0, 666), layout.point(1280, 720), (10, 13, 20), 0.78)
    cv2.line(display_frame, layout.point(0, 666), layout.point(1280, 666), ui.BORDER_SOFT, layout.px(1), cv2.LINE_AA)
    ui.put_text(
        display_frame,
        "Pinch: Draw/Erase    2 fingers: Move/Select    U/Z: Undo    C: Clear    F11: Fullscreen    Q/Esc: Exit",
        layout.point(28, 698),
        layout.font(0.46),
        ui.TEXT_MUTED,
        layout.px(1),
    )

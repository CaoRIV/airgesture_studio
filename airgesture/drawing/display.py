from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from airgesture.drawing import style
from airgesture.ui import theme as ui


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 1280
    height: int = 720
    background_color: tuple[int, int, int] = style.PAPER
    border_color: tuple[int, int, int] = style.CYAN
    workspace: bool = True


@dataclass(frozen=True)
class FrameBounds:
    x: int
    y: int
    width: int
    height: int
    source_x: int = 0
    source_y: int = 0
    source_width: int | None = None
    source_height: int | None = None

    def __iter__(self) -> Iterator[int]:
        return iter((self.x, self.y, self.width, self.height))

    def __getitem__(self, index: int) -> int:
        return (self.x, self.y, self.width, self.height)[index]


def fit_frame_to_display(frame, config: DisplayConfig):
    """Place the camera feed in either the drawing workspace or a plain view."""
    if not config.workspace:
        return _fit_plain_camera(frame, config)

    canvas = np.full(
        (config.height, config.width, 3),
        config.background_color,
        dtype=np.uint8,
    )
    layout = ui.Layout(config.width, config.height)
    style.draw_background(canvas, layout)

    outer_rect = layout.rect(6, 92, 1268, 544)
    style.panel(canvas, outer_rect, layout, fill=style.PAPER_ALT, radius=6)
    x, y, display_width, display_height = layout.rect(16, 103, 1248, 520)
    source_height, source_width = frame.shape[:2]
    target_aspect = display_width / max(display_height, 1)
    source_aspect = source_width / max(source_height, 1)
    crop_x = 0
    crop_y = 0
    crop_width = source_width
    crop_height = source_height
    if source_aspect < target_aspect:
        crop_height = max(1, min(source_height, int(round(source_width / target_aspect))))
        crop_y = (source_height - crop_height) // 2
    elif source_aspect > target_aspect:
        crop_width = max(1, min(source_width, int(round(source_height * target_aspect))))
        crop_x = (source_width - crop_width) // 2
    visible_frame = frame[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    interpolation = (
        cv2.INTER_AREA
        if display_width < crop_width or display_height < crop_height
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(
        visible_frame,
        (display_width, display_height),
        interpolation=interpolation,
    )
    canvas[y : y + display_height, x : x + display_width] = resized
    return canvas, FrameBounds(
        x,
        y,
        display_width,
        display_height,
        crop_x,
        crop_y,
        crop_width,
        crop_height,
    )


def _fit_plain_camera(frame, config: DisplayConfig):
    frame_height, frame_width = frame.shape[:2]
    scale = min(config.width / frame_width, config.height / frame_height)
    display_width = max(1, int(round(frame_width * scale)))
    display_height = max(1, int(round(frame_height * scale)))
    resized = cv2.resize(
        frame,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full(
        (config.height, config.width, 3),
        ui.BG,
        dtype=np.uint8,
    )
    x = (config.width - display_width) // 2
    y = (config.height - display_height) // 2
    canvas[y : y + display_height, x : x + display_width] = resized
    return canvas, FrameBounds(x, y, display_width, display_height)


def frame_point_to_display(
    point: tuple[int, int] | None,
    frame_shape,
    frame_bounds,
) -> tuple[int, int] | None:
    if point is None:
        return None

    frame_height, frame_width = frame_shape[:2]
    x, y, display_width, display_height = frame_bounds
    if isinstance(frame_bounds, FrameBounds):
        source_x = frame_bounds.source_x
        source_y = frame_bounds.source_y
        source_width = frame_bounds.source_width or frame_width
        source_height = frame_bounds.source_height or frame_height
    else:
        source_x = 0
        source_y = 0
        source_width = frame_width
        source_height = frame_height
    scale_x = display_width / source_width
    scale_y = display_height / source_height
    return (
        int(x + (point[0] - source_x) * scale_x),
        int(y + (point[1] - source_y) * scale_y),
    )


def frame_point_to_workspace(
    point: tuple[int, int] | None,
    frame_shape,
    display_shape,
) -> tuple[int, int] | None:
    if point is None:
        return None
    frame_height, frame_width = frame_shape[:2]
    display_height, display_width = display_shape[:2]
    return (
        int(point[0] * display_width / frame_width),
        int(point[1] * display_height / frame_height),
    )


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

    if mode == "Draw":
        status = "DRAW"
        status_color = style.LIME
    elif mode == "Move":
        status = "MOVE"
        status_color = style.CYAN
    else:
        status = "IDLE"
        status_color = style.CYAN

    if detected_symbol is not None:
        _draw_camera_message(
            display_frame,
            layout,
            frame_bounds,
            f"DETECTED: {detected_symbol}",
            style.LIME,
        )
    elif recognition_suggestions:
        suggestion_text = "  ".join(
            f"{symbol} {confidence * 100:.0f}%"
            for symbol, confidence in recognition_suggestions
        )
        _draw_camera_message(
            display_frame,
            layout,
            frame_bounds,
            f"SUGGESTIONS: {suggestion_text}",
            style.YELLOW,
        )
    _draw_bottom_help(
        display_frame,
        status=status,
        status_color=status_color,
        hand_detected=hand_detected,
        fps=fps,
    )
    if notification_message:
        _draw_camera_message(
            display_frame,
            layout,
            frame_bounds,
            notification_message,
            style.RED if notification_is_error else style.LIME,
            bottom=True,
        )


def _draw_frame_border(display_frame, bounds: tuple[int, int, int, int]) -> None:
    x, y, width, height = bounds
    layout = ui.layout_for(display_frame)
    cv2.rectangle(
        display_frame,
        (x, y),
        (x + width - 1, y + height - 1),
        style.INK,
        layout.px(3),
        cv2.LINE_AA,
    )
    corner = layout.px(25)
    inset = layout.px(10)
    for start, end in [
        ((x + inset, y + inset), (x + inset + corner, y + inset)),
        ((x + inset, y + inset), (x + inset, y + inset + corner)),
        ((x + width - inset - corner, y + inset), (x + width - inset, y + inset)),
        ((x + width - inset, y + inset), (x + width - inset, y + inset + corner)),
        ((x + inset, y + height - inset), (x + inset + corner, y + height - inset)),
        ((x + inset, y + height - inset - corner), (x + inset, y + height - inset)),
        ((x + width - inset - corner, y + height - inset), (x + width - inset, y + height - inset)),
        ((x + width - inset, y + height - inset - corner), (x + width - inset, y + height - inset)),
    ]:
        cv2.line(display_frame, start, end, style.CYAN, layout.px(3), cv2.LINE_AA)


def _draw_bottom_help(
    display_frame,
    *,
    status: str,
    status_color: tuple[int, int, int],
    hand_detected: bool,
    fps: float,
) -> None:
    layout = ui.layout_for(display_frame)
    x, y, width, height = layout.rect(6, 648, 1268, 48)
    style.panel(display_frame, (x, y, width, height), layout, radius=4)

    segments = [
        (f"MODE  {status}", 0, 126, status_color),
        (f"HAND  {'OK' if hand_detected else '--'}", 126, 247, style.LIME),
        (f"FPS  {fps:04.1f}", 247, 360, style.CYAN),
        ("Pinch: Draw/Erase", 360, 552, None),
        ("2 fingers: Move", 552, 724, None),
        ("U/Z: Undo", 724, 842, None),
        ("C: Clear", 842, 944, None),
        ("F11: Fullscreen", 944, 1104, None),
        ("Q/Esc: Exit", 1104, 1252, None),
    ]
    for index, (label, start, end, accent) in enumerate(segments):
        if index > 0:
            separator_x = x + layout.px(start)
            cv2.line(
                display_frame,
                (separator_x, y + layout.px(9)),
                (separator_x, y + height - layout.px(9)),
                style.CYAN,
                layout.px(2),
                cv2.LINE_AA,
            )
        if accent is not None:
            cv2.rectangle(
                display_frame,
                (x + layout.px(start), y),
                (x + layout.px(start + 7), y + height),
                accent,
                -1,
            )
        style.put_center(
            display_frame,
            label,
            (x + layout.px((start + end) / 2), y + height // 2 + layout.px(1)),
            layout.font(0.31),
            style.INK,
            layout.px(1),
        )


def _draw_camera_message(
    frame,
    layout: ui.Layout,
    frame_bounds: FrameBounds | tuple[int, int, int, int],
    message: str,
    color: tuple[int, int, int],
    *,
    bottom: bool = False,
) -> None:
    x, y, width, height = frame_bounds
    message_width = min(width - layout.px(28), layout.px(520))
    message_height = layout.px(35)
    message_x = x + (width - message_width) // 2
    message_y = y + height - message_height - layout.px(16) if bottom else y + layout.px(16)
    style.panel(
        frame,
        (message_x, message_y, message_width, message_height),
        layout,
        fill=style.PAPER_ALT,
        radius=3,
        shadow=True,
        thickness=2,
    )
    cv2.rectangle(
        frame,
        (message_x, message_y),
        (message_x + layout.px(7), message_y + message_height),
        color,
        -1,
    )
    style.put_center(
        frame,
        message,
        (message_x + message_width // 2, message_y + message_height // 2),
        layout.font(0.36),
        style.INK,
        layout.px(1),
    )

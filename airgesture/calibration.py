from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from airgesture.config import SETTINGS
from airgesture.config import require_valid_settings
from airgesture.core.camera import Camera
from airgesture.core.hand_tracker import HandTracker
from airgesture.drawing.display import DisplayConfig, fit_frame_to_display
from airgesture.ui import theme as ui
from airgesture.ui.runtime_errors import run_with_error_dialog
from airgesture.ui.window import ResponsiveWindow


WINDOW_NAME = "AirGesture Camera Check"

CAMERA_CHECK_PAPER = (250, 250, 247)
CAMERA_CHECK_INK = (8, 9, 10)
CAMERA_CHECK_BODY = (46, 48, 52)
CAMERA_CHECK_GRID = (224, 226, 222)
CAMERA_CHECK_CYAN = (220, 222, 24)
CAMERA_CHECK_LIME = (36, 238, 164)
CAMERA_CHECK_YELLOW = (0, 225, 255)
CAMERA_CHECK_PINK = (168, 70, 242)
CAMERA_CHECK_GREEN = (57, 201, 49)
CAMERA_CHECK_FONT = cv2.FONT_HERSHEY_DUPLEX
CAMERA_CHECK_BODY_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class CameraCheckConfig:
    title: str = "Check camera, lighting, and hand tracking."
    required_hands: int = 1
    min_brightness: float = SETTINGS.calibration.min_brightness
    max_brightness: float = SETTINGS.calibration.max_brightness

    def __post_init__(self) -> None:
        if self.required_hands < 1:
            raise ValueError("required_hands must be at least 1")
        if not 0.0 <= self.min_brightness < self.max_brightness <= 255.0:
            raise ValueError("brightness must satisfy 0 <= min < max <= 255")


def run_camera_check(config: CameraCheckConfig | None = None) -> int:
    return run_with_error_dialog(
        WINDOW_NAME,
        lambda: _run_camera_check(config),
    )


def _run_camera_check(config: CameraCheckConfig | None = None) -> int:
    settings = require_valid_settings()
    resolved_config = config or CameraCheckConfig()
    camera = Camera(settings.camera)
    camera.open_or_raise()

    tracker_config = settings.calibration_tracker(resolved_config.required_hands)
    window = ResponsiveWindow(WINDOW_NAME)
    try:
        window.create()
        camera.apply_window_title(WINDOW_NAME)
        with HandTracker(tracker_config) as hand_tracker:
            while True:
                frame = camera.read_or_raise()

                results = hand_tracker.detect(frame)
                hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0
                brightness = average_brightness(frame)
                ready = is_ready(resolved_config, hand_count, brightness)

                hand_tracker.draw_landmarks(frame, results)
                viewport = window.viewport()
                display_frame, frame_bounds = fit_frame_to_display(
                    frame,
                    DisplayConfig(
                        width=viewport.width,
                        height=viewport.height,
                        workspace=False,
                    ),
                )
                draw_camera_check_hud(
                    display_frame,
                    frame_bounds,
                    resolved_config,
                    hand_count,
                    brightness,
                    ready,
                    camera_label=camera.status_label,
                )
                window.present(display_frame)

                key_code = cv2.waitKeyEx(1)
                if window.handle_window_key(key_code):
                    continue
                if key_code in (
                    27,
                    10,
                    13,
                    ord(" "),
                    ord("k"),
                    ord("K"),
                    ord("q"),
                    ord("Q"),
                ):
                    return 0
    finally:
        camera.release()
        cv2.destroyWindow(WINDOW_NAME)


def average_brightness(frame) -> float:
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(grayscale))


def is_ready(config: CameraCheckConfig, hand_count: int, brightness: float) -> bool:
    return (
        hand_count >= config.required_hands
        and config.min_brightness <= brightness <= config.max_brightness
    )


def draw_camera_check_hud(
    frame,
    frame_bounds: tuple[int, int, int, int],
    config: CameraCheckConfig,
    hand_count: int,
    brightness: float,
    ready: bool,
    camera_label: str | None = None,
) -> None:
    camera_frame = _extract_camera_frame(frame, frame_bounds)
    layout = ui.layout_for(frame)
    frame[:] = CAMERA_CHECK_PAPER
    _draw_camera_check_grid(frame, layout)

    cv2.rectangle(frame, layout.point(0, 0), layout.point(1280, 112), CAMERA_CHECK_PAPER, -1)
    cv2.line(
        frame,
        layout.point(0, 112),
        layout.point(1280, 112),
        CAMERA_CHECK_INK,
        layout.px(2),
        cv2.LINE_AA,
    )
    _camera_check_text(
        frame,
        "CAMERA CHECK",
        layout.point(42, 58),
        layout.font(1.16),
        CAMERA_CHECK_INK,
        layout.px(4),
    )
    _camera_check_text(
        frame,
        config.title,
        layout.point(44, 83),
        layout.font(0.43),
        CAMERA_CHECK_BODY,
        layout.px(1),
        font=CAMERA_CHECK_BODY_FONT,
    )
    if camera_label:
        _camera_check_text(
            frame,
            camera_label,
            layout.point(44, 102),
            layout.font(0.28),
            CAMERA_CHECK_BODY,
            layout.px(1),
            font=CAMERA_CHECK_BODY_FONT,
        )

    _draw_camera_check_status(frame, layout, ready)

    camera_rect = layout.rect(44, 124, 1192, 390)
    _draw_wide_camera_preview(frame, camera_frame, camera_rect, layout, ready)

    hands_ok = hand_count >= config.required_hands
    brightness_ok = config.min_brightness <= brightness <= config.max_brightness
    _draw_camera_metric(
        frame,
        layout,
        layout.rect(44, 532, 350, 82),
        "HANDS",
        f"{hand_count} / {config.required_hands}",
        CAMERA_CHECK_GREEN if hands_ok else CAMERA_CHECK_PINK,
        "hand",
    )
    _draw_camera_metric(
        frame,
        layout,
        layout.rect(414, 532, 402, 82),
        "BRIGHTNESS",
        f"{brightness:05.1f}",
        CAMERA_CHECK_LIME if brightness_ok else CAMERA_CHECK_YELLOW,
        "sun",
    )
    _draw_camera_metric(
        frame,
        layout,
        layout.rect(836, 532, 400, 82),
        "FRAME",
        f"{camera_frame.shape[1]} x {camera_frame.shape[0]}",
        CAMERA_CHECK_CYAN,
        "frame",
    )
    _draw_camera_check_footer(frame, layout)


def _extract_camera_frame(frame, frame_bounds: tuple[int, int, int, int]):
    x, y, width, height = frame_bounds
    frame_height, frame_width = frame.shape[:2]
    left = max(0, min(frame_width, x))
    top = max(0, min(frame_height, y))
    right = max(left, min(frame_width, x + width))
    bottom = max(top, min(frame_height, y + height))
    if right <= left or bottom <= top:
        return frame.copy()
    return frame[top:bottom, left:right].copy()


def _draw_camera_check_grid(frame, layout: ui.Layout) -> None:
    for design_x in range(24, 1280, 48):
        x = layout.x(design_x)
        cv2.line(frame, (x, 0), (x, frame.shape[0]), CAMERA_CHECK_GRID, layout.px(1), cv2.LINE_AA)
    for design_y in range(16, 720, 48):
        y = layout.y(design_y)
        cv2.line(frame, (0, y), (frame.shape[1], y), CAMERA_CHECK_GRID, layout.px(1), cv2.LINE_AA)


def _draw_camera_check_status(frame, layout: ui.Layout, ready: bool) -> None:
    rect = layout.rect(916, 28, 320, 64)
    fill = CAMERA_CHECK_GREEN if ready else CAMERA_CHECK_YELLOW
    _camera_check_panel(frame, rect, layout, fill)
    x, y, _, height = rect
    _draw_camera_icon(
        frame,
        (x + layout.px(22), y + (height - layout.px(34)) // 2, layout.px(42), layout.px(34)),
        CAMERA_CHECK_INK,
        layout,
    )
    _camera_check_text(
        frame,
        "READY" if ready else "ADJUST CAMERA",
        (x + layout.px(80), y + layout.px(42)),
        layout.font(0.62 if ready else 0.55),
        CAMERA_CHECK_INK,
        layout.px(2),
    )


def _draw_wide_camera_preview(
    frame,
    camera_frame,
    rect: tuple[int, int, int, int],
    layout: ui.Layout,
    ready: bool,
) -> None:
    x, y, width, height = rect
    offset = layout.px(7)
    cv2.rectangle(
        frame,
        (x + offset, y + offset),
        (x + width + offset, y + height + offset),
        CAMERA_CHECK_INK,
        -1,
    )
    source_height, source_width = camera_frame.shape[:2]
    target_aspect = width / max(height, 1)
    source_aspect = source_width / max(source_height, 1)
    crop_x = 0
    crop_y = 0
    crop_width = source_width
    crop_height = source_height
    if source_aspect < target_aspect:
        crop_height = max(1, int(round(source_width / target_aspect)))
        crop_y = max(0, (source_height - crop_height) // 2)
    elif source_aspect > target_aspect:
        crop_width = max(1, int(round(source_height * target_aspect)))
        crop_x = max(0, (source_width - crop_width) // 2)
    visible = camera_frame[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    frame[y : y + height, x : x + width] = cv2.resize(visible, (width, height), interpolation=cv2.INTER_AREA)
    border = CAMERA_CHECK_GREEN if ready else CAMERA_CHECK_INK
    cv2.rectangle(frame, (x, y), (x + width, y + height), border, layout.px(3), cv2.LINE_AA)
    _draw_camera_corners(frame, rect, layout)


def _draw_camera_corners(frame, rect: tuple[int, int, int, int], layout: ui.Layout) -> None:
    x, y, width, height = rect
    inset = layout.px(18)
    length = layout.px(28)
    left = x + inset
    right = x + width - inset
    top = y + inset
    bottom = y + height - inset
    for start, end in (
        ((left, top), (left + length, top)),
        ((left, top), (left, top + length)),
        ((right - length, top), (right, top)),
        ((right, top), (right, top + length)),
        ((left, bottom), (left + length, bottom)),
        ((left, bottom - length), (left, bottom)),
        ((right - length, bottom), (right, bottom)),
        ((right, bottom - length), (right, bottom)),
    ):
        cv2.line(frame, start, end, CAMERA_CHECK_CYAN, layout.px(3), cv2.LINE_AA)


def _draw_camera_metric(
    frame,
    layout: ui.Layout,
    rect: tuple[int, int, int, int],
    label: str,
    value: str,
    accent: tuple[int, int, int],
    icon: str,
) -> None:
    _camera_check_panel(frame, rect, layout, CAMERA_CHECK_PAPER)
    x, y, _, height = rect
    icon_width = layout.px(78)
    cv2.rectangle(frame, (x, y), (x + icon_width, y + height), accent, -1)
    cv2.line(frame, (x + icon_width, y), (x + icon_width, y + height), CAMERA_CHECK_INK, layout.px(2), cv2.LINE_AA)
    _draw_metric_icon(frame, layout, icon, (x, y, icon_width, height))
    _camera_check_text(
        frame,
        label,
        (x + layout.px(96), y + layout.px(30)),
        layout.font(0.31),
        CAMERA_CHECK_BODY,
        layout.px(1),
        font=CAMERA_CHECK_BODY_FONT,
    )
    _camera_check_text(
        frame,
        value,
        (x + layout.px(96), y + layout.px(59)),
        layout.font(0.58),
        CAMERA_CHECK_INK,
        layout.px(1),
        font=CAMERA_CHECK_BODY_FONT,
    )
    ruler_y = y + layout.px(70)
    for tick in range(12):
        tick_x = x + layout.px(96 + tick * 18)
        cv2.line(
            frame,
            (tick_x, ruler_y),
            (tick_x, ruler_y + layout.px(4)),
            CAMERA_CHECK_BODY,
            layout.px(1),
            cv2.LINE_AA,
        )


def _draw_metric_icon(frame, layout: ui.Layout, icon: str, rect: tuple[int, int, int, int]) -> None:
    x, y, width, height = rect
    center = (x + width // 2, y + height // 2)
    if icon == "sun":
        radius = layout.px(13)
        cv2.circle(frame, center, radius, CAMERA_CHECK_INK, layout.px(2), cv2.LINE_AA)
        for dx, dy in ((0, -25), (0, 25), (-25, 0), (25, 0), (-18, -18), (18, -18), (-18, 18), (18, 18)):
            start = (
                center[0] + int(round(dx * 0.72 * layout.scale)),
                center[1] + int(round(dy * 0.72 * layout.scale)),
            )
            end = (
                center[0] + int(round(dx * layout.scale)),
                center[1] + int(round(dy * layout.scale)),
            )
            cv2.line(frame, start, end, CAMERA_CHECK_INK, layout.px(2), cv2.LINE_AA)
    elif icon == "frame":
        size = layout.px(14)
        for column, row in ((0, 0), (1, 1), (0, 2)):
            left = center[0] - size + column * size
            top = center[1] - size * 3 // 2 + row * size
            cv2.rectangle(frame, (left, top), (left + size, top + size), CAMERA_CHECK_INK, -1)
    else:
        palm_center = (center[0], center[1] + layout.px(8))
        cv2.ellipse(frame, palm_center, (layout.px(14), layout.px(18)), 0, 0, 180, CAMERA_CHECK_INK, layout.px(3), cv2.LINE_AA)
        for index in range(4):
            finger_x = center[0] - layout.px(11) + index * layout.px(7)
            cv2.line(
                frame,
                (finger_x, center[1] + layout.px(5)),
                (finger_x, center[1] - layout.px(16 + (index % 2) * 5)),
                CAMERA_CHECK_INK,
                layout.px(3),
                cv2.LINE_AA,
            )


def _draw_camera_check_footer(frame, layout: ui.Layout) -> None:
    rect = layout.rect(44, 634, 1192, 68)
    _camera_check_panel(frame, rect, layout, CAMERA_CHECK_PAPER)
    segments = [
        (70, "ENTER / SPACE", "BACK"),
        (390, "K", "BACK"),
        (560, "F11", "FULLSCREEN"),
        (840, "Q / ESC", "BACK"),
    ]
    for index, (x, primary, secondary) in enumerate(segments):
        if index:
            cv2.line(
                frame,
                layout.point(x - 28, 650),
                layout.point(x - 28, 686),
                CAMERA_CHECK_CYAN,
                layout.px(2),
                cv2.LINE_AA,
            )
        _camera_check_text(
            frame,
            primary,
            layout.point(x, 674),
            layout.font(0.43),
            CAMERA_CHECK_INK,
            layout.px(1),
            font=CAMERA_CHECK_BODY_FONT,
        )
        primary_width = cv2.getTextSize(primary, CAMERA_CHECK_BODY_FONT, layout.font(0.43), layout.px(1))[0][0]
        _camera_check_text(
            frame,
            secondary,
            (layout.x(x) + primary_width + layout.px(18), layout.y(674)),
            layout.font(0.30),
            CAMERA_CHECK_BODY,
            layout.px(1),
            font=CAMERA_CHECK_BODY_FONT,
        )


def _draw_camera_icon(
    frame,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    layout: ui.Layout,
) -> None:
    x, y, width, height = rect
    body_top = y + layout.px(6)
    cv2.rectangle(frame, (x, body_top), (x + width, y + height), color, layout.px(2), cv2.LINE_AA)
    cv2.rectangle(
        frame,
        (x + layout.px(10), y),
        (x + layout.px(26), body_top + layout.px(2)),
        color,
        -1,
    )
    cv2.circle(frame, (x + width // 2, body_top + (height - layout.px(6)) // 2), layout.px(8), color, layout.px(2), cv2.LINE_AA)


def _camera_check_panel(
    frame,
    rect: tuple[int, int, int, int],
    layout: ui.Layout,
    fill: tuple[int, int, int],
) -> None:
    x, y, width, height = rect
    offset = layout.px(6)
    cv2.rectangle(frame, (x + offset, y + offset), (x + width + offset, y + height + offset), CAMERA_CHECK_INK, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), CAMERA_CHECK_INK, layout.px(3), cv2.LINE_AA)


def _camera_check_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    *,
    font: int = CAMERA_CHECK_FONT,
) -> None:
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


# Backward-compatible names for integrations using the previous API.
CalibrationConfig = CameraCheckConfig
run_calibration = run_camera_check

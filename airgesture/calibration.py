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
    height, width = frame.shape[:2]
    layout = ui.layout_for(frame)
    x, y, camera_width, camera_height = frame_bounds
    cv2.rectangle(
        frame,
        (x, y),
        (x + camera_width - 1, y + camera_height - 1),
        ui.GREEN if ready else ui.BORDER_SOFT,
        2,
        cv2.LINE_AA,
    )

    ui.blend_rect(frame, layout.point(0, 0), layout.point(1280, 92), (10, 13, 20), 0.86)
    cv2.line(frame, layout.point(0, 92), layout.point(1280, 92), ui.BORDER_SOFT, layout.px(1), cv2.LINE_AA)
    ui.put_text(frame, "CAMERA CHECK", layout.point(28, 38), layout.font(0.84), ui.TEXT, layout.px(2))
    ui.put_text(frame, config.title, layout.point(30, 68), layout.font(0.50), ui.TEXT_MUTED, layout.px(1))
    if camera_label:
        ui.put_text(frame, camera_label, layout.point(30, 87), layout.font(0.40), ui.TEXT_DIM, layout.px(1))

    status_text = "READY" if ready else "ADJUST CAMERA"
    status_color = ui.GREEN if ready else ui.YELLOW
    status_width = 190 if ready else 230
    ui.chip(
        frame,
        layout.rect(1280 - status_width - 28, 27, status_width, 38),
        status_text,
        color=status_color,
        active=True,
    )

    footer_top = layout.y(608)
    footer_bottom = layout.y(720)
    ui.blend_rect(frame, (layout.x(0), footer_top), (layout.x(1280), footer_bottom), (10, 13, 20), 0.88)
    cv2.line(frame, (layout.x(0), footer_top), (layout.x(1280), footer_top), ui.BORDER_SOFT, layout.px(1), cv2.LINE_AA)

    margin = layout.px(28)
    gap = layout.px(12)
    content_left = layout.x(0)
    content_width = layout.px(1280)
    metrics_width = content_width - margin * 2 - gap * 2
    hands_width = int(metrics_width * 0.25)
    brightness_width = int(metrics_width * 0.33)
    frame_width = metrics_width - hands_width - brightness_width
    hands_x = content_left + margin
    brightness_x = hands_x + hands_width + gap
    frame_x = brightness_x + brightness_width + gap

    hands_ok = hand_count >= config.required_hands
    brightness_ok = config.min_brightness <= brightness <= config.max_brightness
    ui.chip(
        frame,
        (hands_x, footer_top + layout.px(18), hands_width, layout.px(36)),
        f"HANDS  {hand_count}/{config.required_hands}",
        color=ui.GREEN if hands_ok else ui.YELLOW,
        active=hands_ok,
    )
    ui.chip(
        frame,
        (brightness_x, footer_top + layout.px(18), brightness_width, layout.px(36)),
        f"BRIGHTNESS  {brightness:05.1f}",
        color=ui.GREEN if brightness_ok else ui.YELLOW,
        active=brightness_ok,
    )
    ui.chip(
        frame,
        (frame_x, footer_top + layout.px(18), frame_width, layout.px(36)),
        f"FRAME  {camera_width}x{camera_height}",
        color=ui.CYAN,
        active=False,
    )
    ui.put_text(
        frame,
        "Enter / Space / K: Back    F11: Fullscreen    Q/Esc: Back",
        layout.point(30, 698),
        layout.font(0.50),
        ui.TEXT_MUTED,
        layout.px(1),
    )


# Backward-compatible names for integrations using the previous API.
CalibrationConfig = CameraCheckConfig
run_calibration = run_camera_check

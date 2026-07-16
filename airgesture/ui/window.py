from __future__ import annotations

import ctypes
from dataclasses import dataclass
import sys

import cv2
import numpy as np


DESIGN_WIDTH = 1280
DESIGN_HEIGHT = 720
F11_KEY_CODES = frozenset({0x7A, 0x7A0000})


@dataclass(frozen=True)
class ScreenMetrics:
    width: int
    height: int
    dpi_scale: float = 1.0


@dataclass(frozen=True)
class Viewport:
    width: int
    height: int
    dpi_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("viewport dimensions must be positive")
        if self.dpi_scale <= 0.0:
            raise ValueError("dpi_scale must be positive")


def enable_dpi_awareness() -> None:
    """Opt into physical-pixel coordinates before creating Windows windows."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass


def screen_metrics() -> ScreenMetrics:
    if sys.platform != "win32":
        return ScreenMetrics(DESIGN_WIDTH, DESIGN_HEIGHT, 1.0)

    user32 = ctypes.windll.user32
    width = max(1, int(user32.GetSystemMetrics(0)))
    height = max(1, int(user32.GetSystemMetrics(1)))
    dpi = 96
    try:
        dpi = max(96, int(user32.GetDpiForSystem()))
    except (AttributeError, OSError):
        pass
    return ScreenMetrics(width, height, dpi / 96.0)


def fit_frame_to_viewport(
    frame,
    viewport: Viewport,
    background_color: tuple[int, int, int] = (10, 12, 17),
):
    """Fit an image into a viewport without stretching its aspect ratio."""
    source_height, source_width = frame.shape[:2]
    scale = min(viewport.width / source_width, viewport.height / source_height)
    target_width = max(1, int(round(source_width * scale)))
    target_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
    canvas = np.full(
        (viewport.height, viewport.width, 3),
        background_color,
        dtype=np.uint8,
    )
    x = (viewport.width - target_width) // 2
    y = (viewport.height - target_height) // 2
    canvas[y : y + target_height, x : x + target_width] = resized
    return canvas, (x, y, target_width, target_height)


def is_f11_key(key_code: int) -> bool:
    return key_code in F11_KEY_CODES


class ResponsiveWindow:
    """OpenCV window with resize, DPI-aware sizing, and fullscreen toggling."""

    def __init__(
        self,
        name: str,
        *,
        start_maximized: bool = False,
        design_size: tuple[int, int] = (DESIGN_WIDTH, DESIGN_HEIGHT),
    ) -> None:
        enable_dpi_awareness()
        self.name = name
        self.start_maximized = start_maximized
        self.design_size = design_size
        self.metrics = screen_metrics()
        self.is_fullscreen = False
        self._windowed_size = self._initial_window_size()

    def _initial_window_size(self) -> tuple[int, int]:
        screen_width = max(640, self.metrics.width)
        screen_height = max(480, self.metrics.height)
        if self.start_maximized:
            width = int(screen_width * 0.94)
            height = int(screen_height * 0.90)
        else:
            dpi_growth = min(max(self.metrics.dpi_scale, 1.0), 1.5)
            width = int(self.design_size[0] * dpi_growth)
            height = int(self.design_size[1] * dpi_growth)
            width = min(width, int(screen_width * 0.90))
            height = min(height, int(screen_height * 0.84))
        return max(640, width), max(480, height)

    def create(self) -> None:
        enable_dpi_awareness()
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, *self._windowed_size)

    def recreate(self) -> None:
        self.is_fullscreen = False
        self.create()

    def viewport(self) -> Viewport:
        try:
            _, _, width, height = cv2.getWindowImageRect(self.name)
        except (AttributeError, cv2.error):
            width, height = self._windowed_size
        if width < 1 or height < 1:
            width, height = self._windowed_size
        return Viewport(width, height, self.metrics.dpi_scale)

    def present(self, frame) -> tuple[int, int, int, int]:
        viewport = self.viewport()
        output, bounds = fit_frame_to_viewport(frame, viewport)
        cv2.imshow(self.name, output)
        return bounds

    def toggle_fullscreen(self) -> None:
        if self.is_fullscreen:
            self.leave_fullscreen()
            return
        current = self.viewport()
        self._windowed_size = (current.width, current.height)
        cv2.setWindowProperty(
            self.name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        self.is_fullscreen = True

    def leave_fullscreen(self) -> None:
        cv2.setWindowProperty(
            self.name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL,
        )
        cv2.resizeWindow(self.name, *self._windowed_size)
        self.is_fullscreen = False

    def handle_window_key(self, key_code: int) -> bool:
        """Handle F11 and fullscreen Escape; return True when consumed."""
        if is_f11_key(key_code):
            self.toggle_fullscreen()
            return True
        if key_code == 27 and self.is_fullscreen:
            self.leave_fullscreen()
            return True
        return False

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CanvasConfig:
    brush_color: tuple[int, int, int] = (40, 120, 255)
    brush_thickness: int = 8
    cursor_color: tuple[int, int, int] = (255, 255, 255)


class DrawingCanvas:
    """Stores drawn strokes separately from the camera frame."""

    def __init__(self, config: CanvasConfig | None = None) -> None:
        self.config = config or CanvasConfig()
        self._canvas = None

    def ensure_size(self, frame_shape) -> None:
        height, width = frame_shape[:2]
        if self._canvas is None or self._canvas.shape[:2] != (height, width):
            self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def clear(self) -> None:
        if self._canvas is not None:
            self._canvas[:] = 0

    def draw_line(self, start_point: tuple[int, int], end_point: tuple[int, int]) -> None:
        if self._canvas is None:
            return

        cv2.line(
            self._canvas,
            start_point,
            end_point,
            self.config.brush_color,
            self.config.brush_thickness,
            cv2.LINE_AA,
        )

    def compose(self, frame):
        if self._canvas is None:
            return frame

        return cv2.addWeighted(frame, 1.0, self._canvas, 1.0, 0)

    def draw_cursor(self, frame, point: tuple[int, int]) -> None:
        cv2.circle(frame, point, 13, self.config.brush_color, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 4, self.config.cursor_color, -1, cv2.LINE_AA)

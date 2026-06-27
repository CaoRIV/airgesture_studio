from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CanvasConfig:
    brush_color: tuple[int, int, int] = (0, 0, 255)
    brush_thickness: int = 10
    eraser_thickness: int = 34
    cursor_color: tuple[int, int, int] = (255, 255, 255)


class DrawingCanvas:
    """Stores drawn strokes separately from the camera frame."""

    def __init__(self, config: CanvasConfig | None = None) -> None:
        self.config = config or CanvasConfig()
        self._canvas = None
        self._stroke_canvas = None

    def ensure_size(self, frame_shape) -> None:
        height, width = frame_shape[:2]
        if self._canvas is None or self._canvas.shape[:2] != (height, width):
            self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
            self._stroke_canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def clear(self) -> None:
        if self._canvas is not None:
            self._canvas[:] = 0
        self.clear_stroke()

    def clear_stroke(self) -> None:
        if self._stroke_canvas is not None:
            self._stroke_canvas[:] = 0

    def commit_stroke(self) -> None:
        if self._canvas is None or self._stroke_canvas is None:
            return

        stroke_mask = np.any(self._stroke_canvas > 0, axis=2)
        self._canvas[stroke_mask] = self._stroke_canvas[stroke_mask]
        self.clear_stroke()

    def commit_clean_stroke(self, points: list[tuple[int, int]]) -> None:
        if self._canvas is None:
            return

        self.clear_stroke()
        if len(points) < 2:
            return

        for start_point, end_point in zip(points, points[1:]):
            cv2.line(
                self._canvas,
                start_point,
                end_point,
                self.config.brush_color,
                self.config.brush_thickness,
                cv2.LINE_AA,
            )

    def set_brush_color(self, color: tuple[int, int, int]) -> None:
        self.config.brush_color = color

    def set_brush_thickness(self, thickness: int) -> None:
        self.config.brush_thickness = max(2, thickness)

    def draw_line(self, start_point: tuple[int, int], end_point: tuple[int, int]) -> None:
        if self._stroke_canvas is None:
            return

        cv2.line(
            self._stroke_canvas,
            start_point,
            end_point,
            self.config.brush_color,
            self.config.brush_thickness,
            cv2.LINE_AA,
        )

    def erase_line(self, start_point: tuple[int, int], end_point: tuple[int, int]) -> None:
        if self._canvas is None:
            return

        cv2.line(
            self._canvas,
            start_point,
            end_point,
            (0, 0, 0),
            self.config.eraser_thickness,
            cv2.LINE_AA,
        )
        self.clear_stroke()

    def draw_clean_letter(self, letter: str, bounds: tuple[int, int, int, int]) -> None:
        if self._canvas is None:
            return

        x, y, width, height = bounds
        frame_height, frame_width = self._canvas.shape[:2]
        width = min(width, frame_width - x)
        height = min(height, frame_height - y)
        if width <= 0 or height <= 0:
            return

        thickness = max(self.config.brush_thickness + 2, 8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        text_size, baseline = cv2.getTextSize(letter, font, scale, thickness)
        target_width = max(1, int(width * 0.85))
        target_height = max(1, int(height * 0.85))
        scale = min(target_width / text_size[0], target_height / text_size[1])
        text_size, baseline = cv2.getTextSize(letter, font, scale, thickness)
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        text_y = min(text_y, frame_height - baseline - 1)

        cv2.putText(
            self._canvas,
            letter,
            (text_x, text_y),
            font,
            scale,
            self.config.brush_color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_clean_c(self, bounds: tuple[int, int, int, int]) -> None:
        self.draw_clean_letter("C", bounds)

    def compose(self, frame):
        if self._canvas is None:
            return frame

        composed = frame.copy()
        self._overlay_layer(composed, self._canvas)
        if self._stroke_canvas is not None:
            self._overlay_layer(composed, self._stroke_canvas)

        return composed

    def _overlay_layer(self, frame, layer) -> None:
        if layer is None:
            return

        mask = np.any(layer > 0, axis=2)
        frame[mask] = layer[mask]

    def save(self, output_dir: str | Path, filename: str) -> Path | None:
        if self._canvas is None:
            return None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / filename
        cv2.imwrite(str(file_path), self._canvas)
        return file_path

    def draw_cursor(self, frame, point: tuple[int, int], erasing: bool = False) -> None:
        color = (230, 235, 240) if erasing else self.config.brush_color
        radius = self.config.eraser_thickness // 2 if erasing else max(10, self.config.brush_thickness + 5)
        cv2.circle(frame, point, radius, color, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 4, self.config.cursor_color, -1, cv2.LINE_AA)

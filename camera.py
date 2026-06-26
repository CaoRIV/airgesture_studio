from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class CameraConfig:
    camera_index: int = 0
    mirror: bool = True
    width: int = 1280
    height: int = 720
    fps: int = 30


class Camera:
    """Small wrapper around OpenCV webcam capture."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._capture = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)
        if self.is_opened:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        return self.is_opened

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def read(self):
        if not self.is_opened:
            return False, None

        success, frame = self._capture.read()
        if not success:
            return False, None

        if self.config.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        if not self.open():
            raise RuntimeError(
                f"Could not open webcam at index {self.config.camera_index}. "
                "Check that a webcam is connected and not being used by another app."
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

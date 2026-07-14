from __future__ import annotations

from dataclasses import dataclass

import cv2

from airgesture.errors import CameraError


@dataclass
class CameraConfig:
    camera_index: int = 0
    mirror: bool = True
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera_index cannot be negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.fps <= 0:
            raise ValueError("camera fps must be positive")
        if self.buffer_size < 1:
            raise ValueError("camera buffer_size must be at least 1")


class Camera:
    """Small wrapper around OpenCV webcam capture."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        try:
            self._capture = cv2.VideoCapture(
                self.config.camera_index,
                cv2.CAP_DSHOW,
            )
        except cv2.error as exc:
            raise CameraError(
                f"Could not initialize webcam {self.config.camera_index}."
            ) from exc
        if self.is_opened:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        return self.is_opened

    def open_or_raise(self) -> None:
        if not self.open():
            self.release()
            raise CameraError(
                f"Could not open webcam {self.config.camera_index}. "
                "Check that it is connected and not used by another app."
            )

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

    def read_or_raise(self):
        if not self.is_opened:
            raise CameraError("The webcam is not open.")
        try:
            success, frame = self.read()
        except cv2.error as exc:
            raise CameraError("The webcam failed while reading a frame.") from exc
        if not success or frame is None:
            raise CameraError(
                "The webcam stopped returning frames. Reconnect it and try again."
            )
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        self.open_or_raise()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

from __future__ import annotations

from dataclasses import dataclass, field
import sys
import time

import cv2

from airgesture.errors import (
    CameraAccessError,
    CameraDisconnectedError,
    CameraError,
    CameraNotFoundError,
)


@dataclass(frozen=True)
class CameraBackend:
    name: str
    api_preference: int


@dataclass(frozen=True)
class CameraInfo:
    index: int
    backend: str
    width: int
    height: int
    fps: float

    @property
    def label(self) -> str:
        fps_text = f"{self.fps:.0f} FPS" if self.fps > 0.0 else "FPS unknown"
        return (
            f"Camera {self.index} | {self.backend} | "
            f"{self.width}x{self.height} @ {fps_text}"
        )


@dataclass
class CameraConfig:
    camera_index: int = 0
    mirror: bool = True
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1
    discovery_max_devices: int = 5
    read_failure_tolerance: int = 2
    reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 0.20
    available_indices: tuple[int, ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera_index cannot be negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.fps <= 0:
            raise ValueError("camera fps must be positive")
        if self.buffer_size < 1:
            raise ValueError("camera buffer_size must be at least 1")
        if self.discovery_max_devices < 1:
            raise ValueError("discovery_max_devices must be at least 1")
        if self.read_failure_tolerance < 0:
            raise ValueError("read_failure_tolerance cannot be negative")
        if self.reconnect_attempts < 0:
            raise ValueError("reconnect_attempts cannot be negative")
        if self.reconnect_delay_seconds < 0.0:
            raise ValueError("reconnect_delay_seconds cannot be negative")


class Camera:
    """Small wrapper around OpenCV webcam capture."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture: cv2.VideoCapture | None = None
        self._backend: CameraBackend | None = None
        self._info: CameraInfo | None = None
        self._reconnect_count = 0
        self._window_title_base: str | None = None

    @staticmethod
    def backend_candidates() -> tuple[CameraBackend, ...]:
        if sys.platform == "win32":
            return (
                CameraBackend("DirectShow", cv2.CAP_DSHOW),
                CameraBackend("Media Foundation", cv2.CAP_MSMF),
                CameraBackend("Default", cv2.CAP_ANY),
            )
        return (CameraBackend("Default", cv2.CAP_ANY),)

    @classmethod
    def discover_indices(cls, max_devices: int = 5) -> list[int]:
        discovered: list[int] = []
        for camera_index in range(max_devices):
            if cls._probe_index(camera_index):
                discovered.append(camera_index)
        return discovered

    @classmethod
    def _probe_index(cls, camera_index: int) -> bool:
        for backend in cls.backend_candidates():
            capture = None
            try:
                capture = cv2.VideoCapture(camera_index, backend.api_preference)
                if capture is not None and capture.isOpened():
                    return True
            except cv2.error:
                continue
            finally:
                if capture is not None:
                    capture.release()
        return False

    def open(self) -> bool:
        self.release()
        for backend in self.backend_candidates():
            capture = None
            try:
                capture = cv2.VideoCapture(
                    self.config.camera_index,
                    backend.api_preference,
                )
                if capture is None or not capture.isOpened():
                    continue

                self._capture = capture
                self._backend = backend
                self._configure_capture()
                self._info = self._read_camera_info()
                if self._window_title_base is not None:
                    self.apply_window_title(self._window_title_base)
                return True
            except cv2.error:
                if capture is self._capture:
                    self._capture = None
                    self._backend = None
                    self._info = None
                continue
            finally:
                if capture is not None and capture is not self._capture:
                    capture.release()
        return False

    def _configure_capture(self) -> None:
        assert self._capture is not None
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

    def _read_camera_info(self) -> CameraInfo:
        assert self._capture is not None
        width = self._safe_property(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        height = self._safe_property(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        fps = self._safe_property(cv2.CAP_PROP_FPS, float(self.config.fps))
        backend_name = self._backend.name if self._backend is not None else "Default"
        return CameraInfo(
            index=self.config.camera_index,
            backend=backend_name,
            width=max(1, int(round(width))),
            height=max(1, int(round(height))),
            fps=max(0.0, float(fps)),
        )

    def _safe_property(self, property_id: int, fallback: float) -> float:
        assert self._capture is not None
        try:
            value = float(self._capture.get(property_id))
        except (cv2.error, TypeError, ValueError):
            return float(fallback)
        return value if value > 0.0 else float(fallback)

    def open_or_raise(self) -> None:
        if not self.open():
            self.release()
            if self.config.camera_index in self.config.available_indices:
                raise CameraAccessError(
                    f"Webcam {self.config.camera_index} was detected but could not "
                    "be opened. Close other camera apps and check Windows camera "
                    "privacy permissions."
                )
            raise CameraNotFoundError(
                f"Webcam {self.config.camera_index} was not detected. Connect it "
                "and press R in the menu to rescan."
            )

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def info(self) -> CameraInfo | None:
        return self._info

    @property
    def status_label(self) -> str:
        if self._info is not None:
            return self._info.label
        return f"Camera {self.config.camera_index} | Not connected"

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    def apply_window_title(self, base_title: str) -> None:
        self._window_title_base = base_title
        try:
            cv2.setWindowTitle(base_title, f"{base_title} | {self.status_label}")
        except (AttributeError, cv2.error):
            pass

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

        for attempt in range(self.config.read_failure_tolerance + 1):
            try:
                success, frame = self.read()
            except cv2.error:
                success, frame = False, None
            if success and frame is not None:
                return frame
            if attempt < self.config.read_failure_tolerance:
                time.sleep(0.01)

        return self._reconnect_and_read()

    def _reconnect_and_read(self):
        for _ in range(self.config.reconnect_attempts):
            self.release()
            if self.config.reconnect_delay_seconds > 0.0:
                time.sleep(self.config.reconnect_delay_seconds)
            if not self.open():
                continue
            try:
                success, frame = self.read()
            except cv2.error:
                success, frame = False, None
            if success and frame is not None:
                self._reconnect_count += 1
                return frame

        raise CameraDisconnectedError(
            f"Webcam {self.config.camera_index} was disconnected and automatic "
            "reconnection failed. Reconnect it, then return to the menu and rescan."
        )

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._backend = None
        self._info = None

    def __enter__(self) -> "Camera":
        self.open_or_raise()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

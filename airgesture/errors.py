from __future__ import annotations


class AirGestureError(RuntimeError):
    """Base class for failures that can be explained to the user."""


class CameraError(AirGestureError):
    """Raised when a webcam cannot be opened or read."""


class CameraNotFoundError(CameraError):
    """Raised when the selected camera was not found during discovery."""


class CameraAccessError(CameraError):
    """Raised when a discovered camera cannot be opened."""


class CameraDisconnectedError(CameraError):
    """Raised when an active camera disconnects and recovery fails."""


class HandTrackingError(AirGestureError):
    """Raised when the hand-tracking model cannot start or process a frame."""


class DrawingSaveError(AirGestureError):
    """Raised when a drawing cannot be written to disk."""


class OutputDirectoryError(AirGestureError):
    """Raised when the drawings directory cannot be created or opened."""

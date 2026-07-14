from __future__ import annotations


class AirGestureError(RuntimeError):
    """Base class for failures that can be explained to the user."""


class CameraError(AirGestureError):
    """Raised when a webcam cannot be opened or read."""


class HandTrackingError(AirGestureError):
    """Raised when the hand-tracking model cannot start or process a frame."""


class DrawingSaveError(AirGestureError):
    """Raised when a drawing cannot be written to disk."""


class OutputDirectoryError(AirGestureError):
    """Raised when the drawings directory cannot be created or opened."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from airgesture.core.camera import CameraConfig
from airgesture.core.hand_tracker import HandTrackerConfig
from airgesture.core.smoothing import (
    AdaptiveSmoothingConfig,
    OneEuroConfig,
    SmoothingConfig,
)
from airgesture.puzzle.capture_gesture import CaptureGestureConfig
from airgesture.puzzle.gesture import PinchGestureConfig


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"


class SettingsError(ValueError):
    """Raised when settings.json is missing or contains invalid values."""


@dataclass(frozen=True)
class DrawingSettings:
    draw_grace_frames: int
    stroke_end_debounce_seconds: float
    max_bridge_distance: float
    detection_display_seconds: float
    default_brush_size: int
    thin_brush_size: int
    thick_brush_size: int
    eraser_size: int
    max_undo_steps: int

    def __post_init__(self) -> None:
        if self.draw_grace_frames < 0:
            raise ValueError("draw_grace_frames cannot be negative")
        if self.stroke_end_debounce_seconds < 0.0:
            raise ValueError("stroke_end_debounce_seconds cannot be negative")
        if self.max_bridge_distance <= 0.0:
            raise ValueError("max_bridge_distance must be positive")
        if self.detection_display_seconds < 0.0:
            raise ValueError("detection_display_seconds cannot be negative")
        brush_sizes = (
            self.default_brush_size,
            self.thin_brush_size,
            self.thick_brush_size,
            self.eraser_size,
        )
        if any(size < 2 for size in brush_sizes):
            raise ValueError("brush and eraser sizes must be at least 2")
        if self.max_undo_steps < 1:
            raise ValueError("max_undo_steps must be at least 1")


@dataclass(frozen=True)
class AirDrawingSettings:
    tracker: HandTrackerConfig
    adaptive_smoothing: AdaptiveSmoothingConfig
    pinch: PinchGestureConfig
    drawing: DrawingSettings


@dataclass(frozen=True)
class PuzzleGameSettings:
    countdown_seconds: float
    default_difficulty: int
    board_size: int

    def __post_init__(self) -> None:
        if self.countdown_seconds < 0.0:
            raise ValueError("countdown_seconds cannot be negative")
        if self.default_difficulty not in (3, 4):
            raise ValueError("default_difficulty must be 3 or 4")
        if self.board_size < 120:
            raise ValueError("board_size must be at least 120 pixels")


@dataclass(frozen=True)
class PuzzleSettings:
    tracker: HandTrackerConfig
    cursor_smoothing: SmoothingConfig
    pinch: PinchGestureConfig
    capture: CaptureGestureConfig
    game: PuzzleGameSettings


@dataclass(frozen=True)
class CalibrationSettings:
    tracker: HandTrackerConfig


@dataclass(frozen=True)
class AppSettings:
    camera: CameraConfig
    air_drawing: AirDrawingSettings
    puzzle: PuzzleSettings
    calibration: CalibrationSettings

    def calibration_tracker(self, required_hands: int) -> HandTrackerConfig:
        return replace(
            self.calibration.tracker,
            max_num_hands=max(1, required_hands),
        )


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> AppSettings:
    settings_path = Path(path)
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SettingsError(f"Settings file not found: {settings_path}") from exc
    except json.JSONDecodeError as exc:
        raise SettingsError(
            f"Invalid JSON in {settings_path} at line {exc.lineno}, column {exc.colno}"
        ) from exc

    try:
        root = _section(raw, "root", {"camera", "air_drawing", "puzzle", "calibration"})
        air = _section(
            root["air_drawing"],
            "air_drawing",
            {"tracker", "adaptive_smoothing", "pinch", "drawing"},
        )
        puzzle = _section(
            root["puzzle"],
            "puzzle",
            {"tracker", "cursor_smoothing", "pinch", "capture", "game"},
        )
        calibration = _section(
            root["calibration"],
            "calibration",
            {"tracker"},
        )
        return AppSettings(
            camera=CameraConfig(**_mapping(root["camera"], "camera")),
            air_drawing=AirDrawingSettings(
                tracker=_tracker_config(air["tracker"]),
                adaptive_smoothing=AdaptiveSmoothingConfig(
                    **_mapping(
                        air["adaptive_smoothing"],
                        "air_drawing.adaptive_smoothing",
                    )
                ),
                pinch=PinchGestureConfig(
                    **_mapping(air["pinch"], "air_drawing.pinch")
                ),
                drawing=DrawingSettings(
                    **_mapping(air["drawing"], "air_drawing.drawing")
                ),
            ),
            puzzle=PuzzleSettings(
                tracker=_tracker_config(puzzle["tracker"]),
                cursor_smoothing=SmoothingConfig(
                    **_mapping(puzzle["cursor_smoothing"], "puzzle.cursor_smoothing")
                ),
                pinch=PinchGestureConfig(
                    **_mapping(puzzle["pinch"], "puzzle.pinch")
                ),
                capture=CaptureGestureConfig(
                    **_mapping(puzzle["capture"], "puzzle.capture")
                ),
                game=PuzzleGameSettings(
                    **_mapping(puzzle["game"], "puzzle.game")
                ),
            ),
            calibration=CalibrationSettings(
                tracker=_tracker_config(calibration["tracker"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SettingsError(f"Invalid settings in {settings_path}: {exc}") from exc


def _tracker_config(value: Any) -> HandTrackerConfig:
    raw = dict(_mapping(value, "tracker"))
    raw["landmark_filter"] = OneEuroConfig(
        **_mapping(raw["landmark_filter"], "tracker.landmark_filter")
    )
    return HandTrackerConfig(**raw)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _section(value: Any, name: str, expected_keys: set[str]) -> dict[str, Any]:
    section = _mapping(value, name)
    missing_keys = expected_keys - section.keys()
    unknown_keys = section.keys() - expected_keys
    if missing_keys:
        raise KeyError(f"{name} is missing: {', '.join(sorted(missing_keys))}")
    if unknown_keys:
        raise KeyError(f"{name} has unknown fields: {', '.join(sorted(unknown_keys))}")
    return section


SETTINGS = load_settings()

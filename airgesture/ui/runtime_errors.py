from __future__ import annotations

import ctypes
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Callable

from airgesture.errors import AirGestureError
from airgesture.paths import LOGS_DIR


LOGGER_NAME = "airgesture"
LOG_FILE_NAME = "airgesture.log"
_logger: logging.Logger | None = None


def get_runtime_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if not logger.handlers:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = RotatingFileHandler(
                LOGS_DIR / LOG_FILE_NAME,
                maxBytes=1_000_000,
                backupCount=2,
                encoding="utf-8",
            )
        except OSError:
            handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _logger = logger
    return logger


def show_error_dialog(title: str, message: str) -> None:
    """Show a native Windows error dialog, with a console fallback."""
    print(f"{title}: {message}", file=sys.stderr)
    if os.environ.get("AIRGESTURE_NO_DIALOGS") == "1":
        return

    if sys.platform == "win32":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
            return
        except (AttributeError, OSError):
            pass


def show_yes_no_dialog(
    title: str,
    message: str,
    *,
    default_yes: bool = False,
) -> bool:
    """Show a native consent question; safely decline without a GUI."""
    if os.environ.get("AIRGESTURE_NO_DIALOGS") == "1":
        return False

    if sys.platform == "win32":
        try:
            yes_no = 0x04
            information = 0x40
            default_button = 0x00 if default_yes else 0x100
            result = ctypes.windll.user32.MessageBoxW(
                None,
                message,
                title,
                yes_no | information | default_button,
            )
            return result == 6
        except (AttributeError, OSError):
            pass
    return False


def run_with_error_dialog(title: str, operation: Callable[[], int | None]) -> int:
    """Run an application surface and convert failures into user feedback."""
    logger = get_runtime_logger()
    try:
        result = operation()
        return 0 if result is None else int(result)
    except AirGestureError as exc:
        logger.error("%s failed: %s", title, exc)
        show_error_dialog(title, str(exc))
        return 1
    except Exception:
        logger.exception("Unexpected failure in %s", title)
        show_error_dialog(
            title,
            "An unexpected error occurred. See the AirGesture log for details.",
        )
        return 1


def log_user_error(context: str, error: Exception) -> None:
    get_runtime_logger().error("%s: %s", context, error)


def runtime_log_path() -> Path:
    return LOGS_DIR / LOG_FILE_NAME

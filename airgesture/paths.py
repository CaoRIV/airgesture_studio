from __future__ import annotations

import os
from pathlib import Path


APP_DIRECTORY_NAME = "AirGesture"
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
RESOURCE_ROOT = PACKAGE_ROOT / "resources"
BUNDLED_MODELS_DIR = RESOURCE_ROOT / "models"


def get_app_data_dir() -> Path:
    """Return the writable per-user application data directory."""
    override = os.environ.get("AIRGESTURE_DATA_DIR")
    if override:
        return Path(override).expanduser()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_DIRECTORY_NAME

    return Path.home() / ".airgesture"


def get_documents_dir() -> Path:
    """Return the user's documents directory, with an override for tests."""
    override = os.environ.get("AIRGESTURE_DOCUMENTS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Documents"


def get_config_dir() -> Path:
    return get_app_data_dir() / "config"


def get_user_models_dir() -> Path:
    return get_app_data_dir() / "models"


def get_drawings_dir() -> Path:
    override = os.environ.get("AIRGESTURE_DRAWINGS_DIR")
    if override:
        return Path(override).expanduser()
    return get_documents_dir() / APP_DIRECTORY_NAME / "Drawings"


APP_DATA_DIR = get_app_data_dir()
CONFIG_DIR = get_config_dir()
CACHE_DIR = APP_DATA_DIR / "cache"
LOGS_DIR = APP_DATA_DIR / "logs"
USER_MODELS_DIR = get_user_models_dir()
DRAWINGS_DIR = get_drawings_dir()

# Backward-compatible aliases for integrations using the previous names.
MODELS_DIR = BUNDLED_MODELS_DIR
OUTPUTS_DIR = DRAWINGS_DIR.parent

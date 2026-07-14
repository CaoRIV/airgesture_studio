from airgesture.config.settings import (
    AppSettings,
    BUNDLED_SETTINGS_PATH,
    SettingsError,
    load_settings,
    resolve_settings_path,
)


SETTINGS_ERROR: SettingsError | None = None
try:
    SETTINGS = load_settings()
except SettingsError as exc:
    SETTINGS_ERROR = exc
    SETTINGS = load_settings(BUNDLED_SETTINGS_PATH)


def require_valid_settings() -> AppSettings:
    if SETTINGS_ERROR is not None:
        raise SETTINGS_ERROR
    return SETTINGS

__all__ = [
    "SETTINGS",
    "AppSettings",
    "SettingsError",
    "load_settings",
    "require_valid_settings",
    "resolve_settings_path",
]

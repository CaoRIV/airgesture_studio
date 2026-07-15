from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

from airgesture.errors import PrivacyConsentRequiredError
from airgesture.paths import get_app_data_dir
from airgesture.ui.runtime_errors import get_runtime_logger, show_yes_no_dialog


POLICY_VERSION = 1
MEDIAPIPE_VERSION = "0.10.35"
CONSENT_ENVIRONMENT_VARIABLE = "AIRGESTURE_METRICS_CONSENT"
CONSENT_FILE_NAME = "privacy.json"
_ACCEPTED_VALUES = frozenset({"1", "accept", "accepted", "true", "yes"})
_DECLINED_VALUES = frozenset({"0", "decline", "declined", "false", "no"})
_session_consent = False


@dataclass(frozen=True)
class PrivacyConsentRecord:
    policy_version: int
    metrics_consent: bool
    recorded_at: str
    mediapipe_version: str


def consent_file_path() -> Path:
    return get_app_data_dir() / CONSENT_FILE_NAME


def read_metrics_consent(path: Path | None = None) -> bool | None:
    """Return the current-policy consent decision, or None when unavailable."""
    target = path or consent_file_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None

    if not isinstance(raw, dict) or raw.get("policy_version") != POLICY_VERSION:
        return None
    decision = raw.get("metrics_consent")
    return decision if isinstance(decision, bool) else None


def write_metrics_consent(
    accepted: bool,
    path: Path | None = None,
) -> PrivacyConsentRecord:
    """Atomically persist a consent decision for the current privacy notice."""
    target = path or consent_file_path()
    record = PrivacyConsentRecord(
        policy_version=POLICY_VERSION,
        metrics_consent=accepted,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        mediapipe_version=MEDIAPIPE_VERSION,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(asdict(record), indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return record


def require_metrics_consent() -> None:
    """Require informed consent before a MediaPipe Task is initialized."""
    global _session_consent
    if _session_consent:
        return

    environment_decision = _environment_consent()
    if environment_decision is True or read_metrics_consent() is True:
        _session_consent = True
        return
    if environment_decision is False:
        raise _consent_error()

    accepted = show_yes_no_dialog(
        "AirGesture Privacy Notice",
        (
            "AirGesture processes camera frames and hand landmarks on this device. "
            "AirGesture does not upload your camera frames.\n\n"
            "MediaPipe Tasks may contact Google to receive compatibility updates "
            "and send operational metrics, including API usage, performance, "
            "general input metadata, and system information.\n\n"
            "Do you consent to this MediaPipe metrics processing and want to "
            "start hand tracking?"
        ),
        default_yes=False,
    )
    if not accepted:
        raise _consent_error()

    _session_consent = True
    try:
        write_metrics_consent(True)
    except OSError as exc:
        get_runtime_logger().warning(
            "Metrics consent was granted for this session but could not be saved: %s",
            exc,
        )


def _environment_consent() -> bool | None:
    value = os.environ.get(CONSENT_ENVIRONMENT_VARIABLE)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _ACCEPTED_VALUES:
        return True
    if normalized in _DECLINED_VALUES:
        return False
    return None


def _consent_error() -> PrivacyConsentRequiredError:
    return PrivacyConsentRequiredError(
        "Hand tracking was not started because MediaPipe metrics consent was "
        "not granted. Camera frames were not sent to Google. Start a tracking "
        "mode again if you want to review the notice."
    )

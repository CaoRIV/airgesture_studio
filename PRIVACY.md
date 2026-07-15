# AirGesture Studio Privacy Notice

Effective date: July 14, 2026

AirGesture Studio is a local Windows desktop application. It does not require
an AirGesture account and the application code does not implement its own
analytics, advertising, cloud synchronization, or upload service.

## Camera and hand data

Camera frames, detected hand landmarks, gestures, and puzzle images are
processed in memory on the device. AirGesture does not upload camera frames or
hand landmarks. Camera frames are discarded as the application runs unless the
user explicitly saves a drawing or uses a frame inside the local puzzle.

Saved drawings are created only after a user action and are stored by default
under `%USERPROFILE%\Documents\AirGesture\Drawings`.

## MediaPipe network activity

AirGesture uses MediaPipe Tasks 0.10.35 for hand tracking. Google's current
MediaPipe privacy notice and API terms state that input images and video are
processed on-device and are not sent to Google. They also state that MediaPipe
may contact Google for compatibility information, model updates, or bug fixes,
and may send operational metrics. Those metrics can include SDK engagement,
session and inference counts, performance, application and general input
metadata, and host system information.

AirGesture requires an explicit Yes decision before it initializes a MediaPipe
Task. No decision is preselected: the consent dialog defaults to No. The
decision is stored locally in `%LOCALAPPDATA%\AirGesture\privacy.json` and is
requested again whenever this notice's policy version changes.

To revoke consent, close AirGesture and delete that file. For a managed or
automated environment, set `AIRGESTURE_METRICS_CONSENT=declined` to block hand
tracking, even when a saved consent record exists. Setting it to `accepted`
allows hand tracking for that process without writing a consent record.

MediaPipe 0.10.35 does not expose a verified, supported Python switch in its
public API for disabling this network behavior. An organization that requires
strict zero-egress operation should enforce that policy outside the process,
for example with an approved firewall rule or isolated network, and validate
the packaged build before deployment.

## Other local data

AirGesture stores only the data needed to run and diagnose the application:

- Settings in `%LOCALAPPDATA%\AirGesture\config\settings.json`.
- The consent record described above.
- A rotating runtime log in `%LOCALAPPDATA%\AirGesture\logs`. It records
  operational errors and does not intentionally record camera frames or hand
  landmarks. The current log and up to two backups are retained; each file is
  limited to approximately 1 MB.
- Cache files in `%LOCALAPPDATA%\AirGesture\cache`.
- Optional user-supplied recognition models in
  `%LOCALAPPDATA%\AirGesture\models`.

These files remain on the device until the user or device administrator removes
them. Uninstalling a Python package may not remove per-user data; delete the
`%LOCALAPPDATA%\AirGesture` directory and the saved Drawings directory to erase
AirGesture data.

## Third-party terms

MediaPipe's processing is governed by Google's applicable terms and privacy
policy. Dependency and model provenance is documented in
`THIRD_PARTY_NOTICES.md`. This notice must be reviewed whenever MediaPipe or the
bundled model is upgraded.

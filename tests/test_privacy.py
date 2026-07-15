from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from airgesture.errors import PrivacyConsentRequiredError
from airgesture import privacy


class PrivacyConsentTests(unittest.TestCase):
    def setUp(self) -> None:
        privacy._session_consent = False

    def tearDown(self) -> None:
        privacy._session_consent = False

    def test_consent_record_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "privacy.json"
            record = privacy.write_metrics_consent(True, path)

            self.assertTrue(privacy.read_metrics_consent(path))
            self.assertTrue(record.metrics_consent)
            self.assertEqual(record.policy_version, privacy.POLICY_VERSION)
            self.assertEqual(record.mediapipe_version, "0.10.35")

    def test_old_policy_requires_fresh_consent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "privacy.json"
            path.write_text(
                json.dumps({"policy_version": 0, "metrics_consent": True}),
                encoding="utf-8",
            )

            self.assertIsNone(privacy.read_metrics_consent(path))

    def test_declined_environment_decision_blocks_tracking(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {privacy.CONSENT_ENVIRONMENT_VARIABLE: "declined"},
                clear=False,
            ),
            patch("airgesture.privacy.read_metrics_consent", return_value=None),
            patch("airgesture.privacy.show_yes_no_dialog") as dialog,
        ):
            with self.assertRaises(PrivacyConsentRequiredError):
                privacy.require_metrics_consent()

        dialog.assert_not_called()

    def test_prompt_acceptance_is_saved_before_tracking(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {privacy.CONSENT_ENVIRONMENT_VARIABLE: ""},
                clear=False,
            ),
            patch("airgesture.privacy.read_metrics_consent", return_value=None),
            patch("airgesture.privacy.show_yes_no_dialog", return_value=True),
            patch("airgesture.privacy.write_metrics_consent") as write,
        ):
            privacy.require_metrics_consent()

        write.assert_called_once_with(True)

    def test_prompt_decline_does_not_save_consent(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {privacy.CONSENT_ENVIRONMENT_VARIABLE: ""},
                clear=False,
            ),
            patch("airgesture.privacy.read_metrics_consent", return_value=None),
            patch("airgesture.privacy.show_yes_no_dialog", return_value=False),
            patch("airgesture.privacy.write_metrics_consent") as write,
        ):
            with self.assertRaises(PrivacyConsentRequiredError):
                privacy.require_metrics_consent()

        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from airgesture.paths import (
    BUNDLED_MODELS_DIR,
    get_app_data_dir,
    get_config_dir,
    get_drawings_dir,
    get_user_models_dir,
)


class RuntimePathTests(unittest.TestCase):
    def test_bundled_hand_model_is_part_of_package_resources(self) -> None:
        model_path = BUNDLED_MODELS_DIR / "hand_landmarker.task"

        self.assertTrue(model_path.is_file())

    def test_app_data_override_controls_writable_runtime_directories(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with patch.dict(
                "os.environ",
                {"AIRGESTURE_DATA_DIR": temporary_directory},
                clear=True,
            ):
                root = Path(temporary_directory)
                self.assertEqual(get_app_data_dir(), root)
                self.assertEqual(get_config_dir(), root / "config")
                self.assertEqual(get_user_models_dir(), root / "models")

    def test_drawings_use_documents_instead_of_project_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with patch.dict(
                "os.environ",
                {"AIRGESTURE_DOCUMENTS_DIR": temporary_directory},
                clear=True,
            ):
                self.assertEqual(
                    get_drawings_dir(),
                    Path(temporary_directory) / "AirGesture" / "Drawings",
                )

    def test_explicit_drawings_override_has_priority(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            custom_drawings = Path(temporary_directory) / "custom"
            with patch.dict(
                "os.environ",
                {"AIRGESTURE_DRAWINGS_DIR": str(custom_drawings)},
                clear=True,
            ):
                self.assertEqual(get_drawings_dir(), custom_drawings)


if __name__ == "__main__":
    unittest.main()

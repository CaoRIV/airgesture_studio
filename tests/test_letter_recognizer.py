from __future__ import annotations

import unittest

import numpy as np

from airgesture.drawing.letter_recognizer import LetterRecognizer, RecognitionConfig


def vertical_i_points() -> list[tuple[int, int]]:
    return [(100, y) for y in range(30, 231, 8)]


def hooked_one_points() -> list[tuple[int, int]]:
    hook = [(80 + index * 4, 55 - index * 3) for index in range(6)]
    stem = [(100, y) for y in range(40, 241, 8)]
    return hook + stem


def based_one_points() -> list[tuple[int, int]]:
    stem = [(100, y) for y in range(30, 231, 8)]
    base = [(100 + index * 5, 230) for index in range(1, 11)]
    return stem + base


class LetterRecognizerTests(unittest.TestCase):
    def test_clean_vertical_stroke_is_uppercase_i_not_digit_one(self) -> None:
        analysis = LetterRecognizer().analyze(vertical_i_points())

        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis.suggestions[0].symbol, "I")
        self.assertIsNotNone(analysis.accepted)
        assert analysis.accepted is not None
        self.assertEqual(analysis.accepted.letter, "I")

    def test_top_hook_distinguishes_digit_one_from_uppercase_i(self) -> None:
        analysis = LetterRecognizer().analyze(hooked_one_points())

        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis.suggestions[0].symbol, "1")
        self.assertIsNotNone(analysis.accepted)
        assert analysis.accepted is not None
        self.assertEqual(analysis.accepted.letter, "1")

    def test_bottom_base_distinguishes_digit_one_from_uppercase_i(self) -> None:
        analysis = LetterRecognizer().analyze(based_one_points())

        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis.suggestions[0].symbol, "1")
        self.assertIsNotNone(analysis.accepted)
        assert analysis.accepted is not None
        self.assertEqual(analysis.accepted.letter, "1")

    def test_high_confidence_gate_keeps_top_three_without_snapping(self) -> None:
        recognizer = LetterRecognizer(
            config=RecognitionConfig(snap_confidence_threshold=0.995)
        )

        analysis = recognizer.analyze(vertical_i_points())

        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertIsNone(analysis.accepted)
        self.assertEqual(len(analysis.suggestions), 3)
        self.assertEqual(len({item.symbol for item in analysis.suggestions}), 3)

    def test_renders_centered_grayscale_stroke_for_onnx(self) -> None:
        recognizer = LetterRecognizer()

        image = recognizer.render_stroke_image(hooked_one_points())

        self.assertEqual(image.shape, (64, 64))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(image)), 0)
        self.assertGreater(float(np.mean(image)), 0.0)


if __name__ == "__main__":
    unittest.main()

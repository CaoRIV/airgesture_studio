from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, pi, sin
from pathlib import Path

import cv2
import numpy as np

from airgesture.paths import MODELS_DIR


Point = tuple[float, float]


@dataclass(frozen=True)
class RecognizedLetter:
    letter: str
    bounds: tuple[int, int, int, int]
    confidence: float
    cleaned_points: list[tuple[int, int]]


@dataclass(frozen=True)
class RecognitionCandidate:
    symbol: str
    confidence: float


@dataclass(frozen=True)
class RecognitionAnalysis:
    accepted: RecognizedLetter | None
    suggestions: tuple[RecognitionCandidate, ...]
    bounds: tuple[int, int, int, int]
    cleaned_points: list[tuple[int, int]]


@dataclass(frozen=True)
class RecognitionConfig:
    sample_count: int = 64
    snap_confidence_threshold: float = 0.74
    min_confidence_margin: float = 0.07
    ambiguous_confidence_margin: float = 0.12
    suggestion_count: int = 3
    suggestion_min_confidence: float = 0.30
    image_size: int = 64
    image_padding: int = 8
    image_line_thickness: int = 4
    onnx_model_path: str | None = None
    onnx_weight: float = 0.70
    onnx_labels: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __post_init__(self) -> None:
        if self.sample_count < 16:
            raise ValueError("sample_count must be at least 16")
        confidence_values = (
            self.snap_confidence_threshold,
            self.min_confidence_margin,
            self.ambiguous_confidence_margin,
            self.suggestion_min_confidence,
            self.onnx_weight,
        )
        if any(not 0.0 <= value <= 1.0 for value in confidence_values):
            raise ValueError("recognition confidence values must be in the range [0, 1]")
        if self.suggestion_count < 1:
            raise ValueError("suggestion_count must be at least 1")
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if not 0 <= self.image_padding < self.image_size // 2:
            raise ValueError("image_padding must fit inside image_size")
        if self.image_line_thickness < 1:
            raise ValueError("image_line_thickness must be positive")
        if not self.onnx_labels:
            raise ValueError("onnx_labels cannot be empty")


class OnnxStrokeClassifier:
    """Optional OpenCV-DNN classifier for a normalized one-channel stroke image."""

    def __init__(self, model_path: Path, labels: str, input_size: int) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX handwriting model not found: {model_path}")
        self.labels = labels
        self.input_size = input_size
        self._network = cv2.dnn.readNetFromONNX(str(model_path))

    def predict(self, image: np.ndarray) -> dict[str, float]:
        blob = cv2.dnn.blobFromImage(
            image.astype(np.float32) / 255.0,
            scalefactor=1.0,
            size=(self.input_size, self.input_size),
            mean=(0.0,),
            swapRB=False,
            crop=False,
        )
        self._network.setInput(blob)
        output = np.asarray(self._network.forward(), dtype=np.float32).reshape(-1)
        if output.size != len(self.labels):
            raise ValueError(
                "ONNX output size does not match onnx_labels: "
                f"{output.size} != {len(self.labels)}"
            )

        probabilities = self._probabilities(output)
        return {
            label: float(probability)
            for label, probability in zip(self.labels, probabilities)
        }

    @staticmethod
    def _probabilities(output: np.ndarray) -> np.ndarray:
        if np.all(output >= 0.0) and abs(float(np.sum(output)) - 1.0) <= 0.02:
            return output / max(float(np.sum(output)), 1e-9)
        shifted = output - float(np.max(output))
        exponentials = np.exp(shifted)
        return exponentials / max(float(np.sum(exponentials)), 1e-9)


class LetterRecognizer:
    """Hybrid template/ONNX recognizer with confidence-gated snapping."""

    AMBIGUOUS_GROUPS = (
        frozenset(("I", "1")),
        frozenset(("O", "0")),
        frozenset(("S", "5")),
        frozenset(("Z", "2")),
        frozenset(("B", "8")),
        frozenset(("G", "6")),
    )

    def __init__(
        self,
        config: RecognitionConfig | None = None,
        sample_count: int | None = None,
        match_threshold: float | None = None,
    ) -> None:
        resolved_config = config or RecognitionConfig(
            sample_count=sample_count or 64,
            snap_confidence_threshold=(
                0.74 if match_threshold is None else match_threshold
            ),
        )
        self.config = resolved_config
        self.sample_count = resolved_config.sample_count
        self.match_threshold = resolved_config.snap_confidence_threshold
        self._templates = self._build_templates()
        self._onnx_classifier = self._load_onnx_classifier()

    def recognize(self, points: list[tuple[int, int]]) -> RecognizedLetter | None:
        analysis = self.analyze(points)
        return None if analysis is None else analysis.accepted

    def analyze(self, points: list[tuple[int, int]]) -> RecognitionAnalysis | None:
        cleaned_points = self.clean_points(points)
        if len(cleaned_points) < 12:
            return None

        candidate = self._normalize(cleaned_points)
        if candidate is None:
            return None

        distances: dict[str, float] = {}
        for letter, templates in self._templates.items():
            best_distance = float("inf")
            for template in templates:
                distance = self._path_distance(candidate, template)
                if distance < best_distance:
                    best_distance = distance
            distances[letter] = best_distance

        self._resolve_i_one_ambiguity(distances, cleaned_points)
        template_scores = {
            symbol: self._distance_confidence(distance)
            for symbol, distance in distances.items()
        }
        scores = self._combine_onnx_scores(template_scores, cleaned_points)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked_candidates = tuple(
            RecognitionCandidate(symbol, confidence)
            for symbol, confidence in ranked
        )
        suggestions = tuple(
            candidate
            for candidate in ranked_candidates[: self.config.suggestion_count]
            if candidate.confidence >= self.config.suggestion_min_confidence
        )
        if not suggestions:
            return None

        bounds = self.bounds(cleaned_points)
        accepted = self._accept_candidate(
            ranked_candidates[:2],
            bounds,
            cleaned_points,
        )
        return RecognitionAnalysis(accepted, suggestions, bounds, cleaned_points)

    def render_stroke_image(self, points: list[tuple[int, int]]) -> np.ndarray:
        size = self.config.image_size
        padding = self.config.image_padding
        image = np.zeros((size, size), dtype=np.uint8)
        if len(points) < 2:
            return image

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(max(xs) - min(xs), 1)
        height = max(max(ys) - min(ys), 1)
        min_x = min(xs)
        min_y = min(ys)
        available = size - padding * 2 - 1
        scale = min(available / width, available / height)
        scaled_width = width * scale
        scaled_height = height * scale
        offset_x = padding + (available - scaled_width) / 2.0
        offset_y = padding + (available - scaled_height) / 2.0
        rendered_points = [
            (
                int(round((x - min_x) * scale + offset_x)),
                int(round((y - min_y) * scale + offset_y)),
            )
            for x, y in points
        ]
        cv2.polylines(
            image,
            [np.asarray(rendered_points, dtype=np.int32)],
            False,
            255,
            self.config.image_line_thickness,
            cv2.LINE_AA,
        )
        return image

    def _accept_candidate(
        self,
        suggestions: tuple[RecognitionCandidate, ...],
        bounds: tuple[int, int, int, int],
        cleaned_points: list[tuple[int, int]],
    ) -> RecognizedLetter | None:
        best = suggestions[0]
        if best.confidence < self.config.snap_confidence_threshold:
            return None

        second = suggestions[1] if len(suggestions) > 1 else None
        required_margin = self.config.min_confidence_margin
        if second is not None and self._is_ambiguous_pair(best.symbol, second.symbol):
            required_margin = max(
                required_margin,
                self.config.ambiguous_confidence_margin,
            )
        if second is not None and best.confidence - second.confidence < required_margin:
            return None

        return RecognizedLetter(
            best.symbol,
            bounds,
            best.confidence,
            cleaned_points,
        )

    def _combine_onnx_scores(
        self,
        template_scores: dict[str, float],
        cleaned_points: list[tuple[int, int]],
    ) -> dict[str, float]:
        if self._onnx_classifier is None:
            return template_scores

        image = self.render_stroke_image(cleaned_points)
        onnx_scores = self._onnx_classifier.predict(image)
        weight = self.config.onnx_weight
        symbols = set(template_scores) | set(onnx_scores)
        return {
            symbol: template_scores.get(symbol, 0.0) * (1.0 - weight)
            + onnx_scores.get(symbol, 0.0) * weight
            for symbol in symbols
        }

    def _load_onnx_classifier(self) -> OnnxStrokeClassifier | None:
        if not self.config.onnx_model_path:
            return None
        model_path = Path(self.config.onnx_model_path)
        if not model_path.is_absolute():
            model_path = MODELS_DIR / model_path
        return OnnxStrokeClassifier(
            model_path=model_path,
            labels=self.config.onnx_labels,
            input_size=self.config.image_size,
        )

    def _resolve_i_one_ambiguity(
        self,
        distances: dict[str, float],
        points: list[tuple[int, int]],
    ) -> None:
        if "I" not in distances or "1" not in distances or len(points) < 2:
            return

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(xs) - min(xs)
        height = max(max(ys) - min(ys), 1)
        path_length = max(self._path_length(points), 1e-9)
        endpoint_verticality = abs(points[-1][1] - points[0][1]) / path_length
        width_ratio = width / height
        top_limit = min(ys) + height * 0.25
        bottom_limit = max(ys) - height * 0.25
        top_xs = [point[0] for point in points if point[1] <= top_limit]
        bottom_xs = [point[0] for point in points if point[1] >= bottom_limit]
        top_span_ratio = (
            (max(top_xs) - min(top_xs)) / height if len(top_xs) >= 2 else 0.0
        )
        bottom_span_ratio = (
            (max(bottom_xs) - min(bottom_xs)) / height
            if len(bottom_xs) >= 2
            else 0.0
        )

        # A clean single vertical stroke is treated as uppercase I. Digit 1 needs a
        # visible top hook or base because the two symbols are otherwise identical.
        if width_ratio <= 0.07 and endpoint_verticality >= 0.82:
            distances["I"] = min(distances["I"], 0.055)
            distances["1"] += 0.10
        elif (
            endpoint_verticality >= 0.75
            and top_span_ratio >= 0.07
            and bottom_span_ratio <= 0.05
        ):
            distances["1"] = min(distances["1"], 0.055)
            distances["I"] += 0.10
        elif (
            endpoint_verticality >= 0.70
            and top_span_ratio <= 0.05
            and bottom_span_ratio >= 0.07
        ):
            distances["1"] = min(distances["1"], 0.065)
            distances["I"] += 0.08

    def _distance_confidence(self, distance: float) -> float:
        return max(0.0, min(0.99, 1.0 - distance / 0.42))

    def _is_ambiguous_pair(self, first: str, second: str) -> bool:
        pair = frozenset((first, second))
        return pair in self.AMBIGUOUS_GROUPS

    def clean_points(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        deduped = self._remove_near_duplicates(points, min_distance=3.0)
        if len(deduped) < 2:
            return deduped

        resampled = self._resample(deduped, self.sample_count)
        smoothed = self._smooth(resampled, passes=2)
        return [(int(round(x)), int(round(y))) for x, y in smoothed]

    def bounds(self, points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        padding = max(18, int(max(width, height) * 0.14))
        return (
            max(0, min_x - padding),
            max(0, min_y - padding),
            max(1, width + padding * 2),
            max(1, height + padding * 2),
        )

    def _build_templates(self) -> dict[str, list[list[Point]]]:
        raw_templates = {
            "A": [[(0.18, 0.86), (0.50, 0.14), (0.82, 0.86), (0.66, 0.55), (0.34, 0.55)]],
            "B": [[(0.24, 0.86), (0.24, 0.15), (0.62, 0.16), (0.80, 0.32), (0.62, 0.49), (0.24, 0.50), (0.66, 0.52), (0.82, 0.70), (0.62, 0.86), (0.24, 0.86)]],
            "C": [self._arc(55, 305, center=(0.5, 0.5), radius_x=0.42, radius_y=0.46)],
            "D": [[(0.25, 0.86), (0.25, 0.15), (0.58, 0.16), (0.82, 0.38), (0.80, 0.64), (0.58, 0.84), (0.25, 0.86)]],
            "E": [[(0.82, 0.15), (0.25, 0.15), (0.25, 0.50), (0.70, 0.50), (0.25, 0.50), (0.25, 0.86), (0.82, 0.86)]],
            "F": [[(0.82, 0.15), (0.25, 0.15), (0.25, 0.86), (0.25, 0.50), (0.70, 0.50)]],
            "G": [self._arc(40, 325, center=(0.5, 0.5), radius_x=0.42, radius_y=0.46) + [(0.62, 0.57), (0.84, 0.57)]],
            "H": [[(0.22, 0.15), (0.22, 0.86), (0.22, 0.50), (0.78, 0.50), (0.78, 0.15), (0.78, 0.86)]],
            "I": [
                [(0.25, 0.15), (0.75, 0.15), (0.50, 0.15), (0.50, 0.86), (0.25, 0.86), (0.75, 0.86)],
                [(0.50, 0.15), (0.50, 0.86)],
            ],
            "J": [[(0.20, 0.15), (0.80, 0.15), (0.62, 0.15), (0.62, 0.72), (0.50, 0.88), (0.30, 0.82), (0.22, 0.66)]],
            "K": [[(0.25, 0.15), (0.25, 0.86), (0.25, 0.52), (0.80, 0.15), (0.25, 0.52), (0.82, 0.86)]],
            "O": [self._arc(0, 350, center=(0.5, 0.5), radius_x=0.42, radius_y=0.46)],
            "L": [[(0.25, 0.15), (0.25, 0.86), (0.78, 0.86)]],
            "M": [[(0.16, 0.86), (0.16, 0.16), (0.50, 0.58), (0.84, 0.16), (0.84, 0.86)]],
            "N": [[(0.20, 0.86), (0.20, 0.16), (0.80, 0.86), (0.80, 0.16)]],
            "P": [[(0.24, 0.86), (0.24, 0.15), (0.64, 0.16), (0.82, 0.34), (0.62, 0.52), (0.24, 0.52)]],
            "Q": [self._arc(0, 350, center=(0.5, 0.5), radius_x=0.40, radius_y=0.44) + [(0.60, 0.62), (0.82, 0.88)]],
            "R": [[(0.24, 0.86), (0.24, 0.15), (0.64, 0.16), (0.82, 0.34), (0.62, 0.52), (0.24, 0.52), (0.82, 0.86)]],
            "V": [[(0.18, 0.18), (0.50, 0.86), (0.82, 0.18)]],
            "W": [[(0.12, 0.18), (0.28, 0.86), (0.50, 0.42), (0.72, 0.86), (0.88, 0.18)]],
            "X": [[(0.18, 0.18), (0.82, 0.86), (0.50, 0.52), (0.82, 0.18), (0.18, 0.86)]],
            "Y": [[(0.16, 0.16), (0.50, 0.50), (0.84, 0.16), (0.50, 0.50), (0.50, 0.86)]],
            "Z": [[(0.20, 0.18), (0.82, 0.18), (0.20, 0.86), (0.82, 0.86)]],
            "S": [self._s_curve()],
            "T": [[(0.18, 0.16), (0.82, 0.16), (0.50, 0.16), (0.50, 0.86)]],
            "U": [self._u_curve()],
            "0": [self._arc(0, 350, center=(0.5, 0.5), radius_x=0.36, radius_y=0.46)],
            "1": [[(0.48, 0.18), (0.58, 0.12), (0.58, 0.88)]],
            "2": [[(0.24, 0.25), (0.48, 0.12), (0.78, 0.22), (0.22, 0.86), (0.82, 0.86)]],
            "3": [self._three_curve()],
            "4": [[(0.76, 0.86), (0.76, 0.16), (0.20, 0.58), (0.86, 0.58)]],
            "5": [[(0.82, 0.16), (0.28, 0.16), (0.24, 0.48), (0.66, 0.50), (0.82, 0.70), (0.62, 0.86), (0.26, 0.82)]],
            "6": [self._six_curve()],
            "7": [[(0.18, 0.16), (0.84, 0.16), (0.46, 0.86)]],
            "8": [self._eight_curve()],
            "9": [self._nine_curve()],
        }

        templates: dict[str, list[list[Point]]] = {}
        for letter, strokes in raw_templates.items():
            templates[letter] = []
            for stroke in strokes:
                normalized = self._normalize([(int(x * 1000), int(y * 1000)) for x, y in stroke])
                if normalized is None:
                    continue
                templates[letter].append(normalized)
                templates[letter].append(list(reversed(normalized)))
        return templates

    def _arc(
        self,
        start_degrees: int,
        end_degrees: int,
        center: Point,
        radius_x: float,
        radius_y: float,
    ) -> list[Point]:
        step = 6 if end_degrees >= start_degrees else -6
        degrees = list(range(start_degrees, end_degrees + step, step))
        return [
            (
                center[0] + radius_x * cos(degree * pi / 180.0),
                center[1] + radius_y * sin(degree * pi / 180.0),
            )
            for degree in degrees
        ]

    def _s_curve(self) -> list[Point]:
        top = self._arc(35, 320, center=(0.52, 0.32), radius_x=0.30, radius_y=0.20)
        bottom = self._arc(215, -40, center=(0.48, 0.68), radius_x=0.30, radius_y=0.20)
        return top + bottom

    def _three_curve(self) -> list[Point]:
        top = self._arc(215, -35, center=(0.45, 0.32), radius_x=0.30, radius_y=0.20)
        bottom = self._arc(215, -35, center=(0.45, 0.68), radius_x=0.30, radius_y=0.20)
        return top + bottom

    def _u_curve(self) -> list[Point]:
        left = [(0.24, 0.15), (0.24, 0.62)]
        bottom = self._arc(180, 360, center=(0.50, 0.62), radius_x=0.26, radius_y=0.24)
        right = [(0.76, 0.62), (0.76, 0.15)]
        return left + bottom + right

    def _six_curve(self) -> list[Point]:
        loop = self._arc(320, 700, center=(0.48, 0.58), radius_x=0.30, radius_y=0.30)
        return [(0.72, 0.16), (0.42, 0.24), (0.24, 0.50)] + loop

    def _nine_curve(self) -> list[Point]:
        loop = self._arc(140, 500, center=(0.52, 0.38), radius_x=0.30, radius_y=0.28)
        return loop + [(0.72, 0.52), (0.58, 0.86)]

    def _eight_curve(self) -> list[Point]:
        points = []
        for index in range(84):
            t = 2.0 * pi * index / 83
            points.append(
                (
                    0.50 + 0.28 * sin(t),
                    0.50 + 0.38 * sin(t) * cos(t),
                )
            )
        return points

    def _normalize(self, points: list[tuple[int, int]] | list[Point]) -> list[Point] | None:
        if len(points) < 2:
            return None

        resampled = self._resample([(float(x), float(y)) for x, y in points], self.sample_count)
        xs = [point[0] for point in resampled]
        ys = [point[1] for point in resampled]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        scale = max(width, height)
        if scale < 1e-6:
            return None

        normalized = []
        for x, y in resampled:
            normalized_x = (x - min_x) / scale
            normalized_y = (y - min_y) / scale
            normalized.append((normalized_x, normalized_y))

        centroid_x = sum(point[0] for point in normalized) / len(normalized)
        centroid_y = sum(point[1] for point in normalized) / len(normalized)
        return [(x - centroid_x, y - centroid_y) for x, y in normalized]

    def _resample(self, points: list[Point], sample_count: int) -> list[Point]:
        path_length = self._path_length(points)
        if path_length <= 0:
            return points[:]

        cumulative = [0.0]
        for previous, current in zip(points, points[1:]):
            cumulative.append(
                cumulative[-1] + hypot(current[0] - previous[0], current[1] - previous[1])
            )

        targets = [
            index * path_length / (sample_count - 1)
            for index in range(sample_count)
        ]
        new_points = []
        segment_index = 1

        for target in targets:
            while segment_index < len(cumulative) - 1 and cumulative[segment_index] < target:
                segment_index += 1

            previous_distance = cumulative[segment_index - 1]
            current_distance = cumulative[segment_index]
            previous = points[segment_index - 1]
            current = points[segment_index]
            segment_length = max(current_distance - previous_distance, 1e-9)
            ratio = (target - previous_distance) / segment_length
            new_points.append(
                (
                    previous[0] + ratio * (current[0] - previous[0]),
                    previous[1] + ratio * (current[1] - previous[1]),
                )
            )

        return new_points

    def _remove_near_duplicates(
        self,
        points: list[tuple[int, int]],
        min_distance: float,
    ) -> list[tuple[int, int]]:
        if not points:
            return []

        deduped = [points[0]]
        for point in points[1:]:
            if hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) >= min_distance:
                deduped.append(point)
        return deduped

    def _smooth(self, points: list[Point], passes: int) -> list[Point]:
        smoothed = points[:]
        for _ in range(passes):
            if len(smoothed) < 3:
                return smoothed
            next_points = [smoothed[0]]
            for previous, current, following in zip(smoothed, smoothed[1:], smoothed[2:]):
                next_points.append(
                    (
                        previous[0] * 0.25 + current[0] * 0.50 + following[0] * 0.25,
                        previous[1] * 0.25 + current[1] * 0.50 + following[1] * 0.25,
                    )
                )
            next_points.append(smoothed[-1])
            smoothed = next_points
        return smoothed

    def _path_distance(self, candidate: list[Point], template: list[Point]) -> float:
        return sum(
            hypot(candidate_point[0] - template_point[0], candidate_point[1] - template_point[1])
            for candidate_point, template_point in zip(candidate, template)
        ) / len(candidate)

    def _path_length(self, points: list[Point]) -> float:
        return sum(
            hypot(current[0] - previous[0], current[1] - previous[1])
            for previous, current in zip(points, points[1:])
        )

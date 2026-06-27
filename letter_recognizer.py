from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, pi, sin


Point = tuple[float, float]


@dataclass(frozen=True)
class RecognizedLetter:
    letter: str
    bounds: tuple[int, int, int, int]
    confidence: float
    cleaned_points: list[tuple[int, int]]


class LetterRecognizer:
    """Template recognizer for simple one-stroke letters and digits."""

    def __init__(self, sample_count: int = 64, match_threshold: float = 0.70) -> None:
        self.sample_count = sample_count
        self.match_threshold = match_threshold
        self._templates = self._build_templates()

    def recognize(self, points: list[tuple[int, int]]) -> RecognizedLetter | None:
        cleaned_points = self.clean_points(points)
        if len(cleaned_points) < 12:
            return None

        candidate = self._normalize(cleaned_points)
        if candidate is None:
            return None

        best_letter = None
        best_distance = float("inf")
        for letter, templates in self._templates.items():
            for template in templates:
                distance = self._path_distance(candidate, template)
                if distance < best_distance:
                    best_distance = distance
                    best_letter = letter

        confidence = max(0.0, min(0.99, 1.0 - best_distance / 0.42))
        if best_letter is None or confidence < self.match_threshold:
            return None

        return RecognizedLetter(
            best_letter,
            self.bounds(cleaned_points),
            confidence,
            cleaned_points,
        )

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
            "C": [self._arc(55, 305, center=(0.5, 0.5), radius_x=0.42, radius_y=0.46)],
            "O": [self._arc(0, 350, center=(0.5, 0.5), radius_x=0.42, radius_y=0.46)],
            "L": [[(0.25, 0.15), (0.25, 0.86), (0.78, 0.86)]],
            "V": [[(0.18, 0.18), (0.50, 0.86), (0.82, 0.18)]],
            "Z": [[(0.20, 0.18), (0.82, 0.18), (0.20, 0.86), (0.82, 0.86)]],
            "S": [self._s_curve()],
            "1": [[(0.48, 0.18), (0.58, 0.12), (0.58, 0.88)]],
            "2": [[(0.24, 0.25), (0.48, 0.12), (0.78, 0.22), (0.22, 0.86), (0.82, 0.86)]],
            "3": [self._three_curve()],
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
        if width < 1e-6 and height < 1e-6:
            return None

        normalized = []
        for x, y in resampled:
            normalized_x = 0.5 if width < 1e-6 else (x - min_x) / width
            normalized_y = 0.5 if height < 1e-6 else (y - min_y) / height
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

from __future__ import annotations

import cv2


def draw_capture_hud(
    frame,
    hand_count: int,
    capture_message: str,
    capture_progress: float,
    difficulty: int,
) -> None:
    _draw_top_bar(frame)
    _put(frame, "Gesture Puzzle", (28, 46), 0.95, (245, 247, 250), 2)
    status = f"{capture_message}   3/4: difficulty   Space/Enter/C: fallback capture"
    color = (0, 230, 140) if hand_count >= 2 else (0, 180, 255)
    _put(frame, status, (28, frame.shape[0] - 30), 0.72, color, 2)
    _put(frame, f"Hands: {hand_count}/2", (frame.shape[1] - 500, 46), 0.72, color, 2)
    _put(frame, f"Grid: {difficulty}x{difficulty}", (frame.shape[1] - 310, 46), 0.72, (245, 247, 250), 2)
    _draw_progress(frame, capture_progress)


def draw_countdown_hud(frame, remaining_seconds: float, difficulty: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (8, 10, 14), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    number = max(1, int(remaining_seconds) + 1)
    _put_center(frame, str(number), (center_x, center_y - 30), 3.2, (0, 230, 140), 7)
    _put_center(
        frame,
        f"Creating {difficulty}x{difficulty} puzzle...",
        (center_x, center_y + 92),
        0.86,
        (245, 247, 250),
        2,
    )


def draw_capture_gesture(frame, points, bounds) -> None:
    for point in points:
        cv2.circle(frame, point, 10, (0, 230, 140), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 14, (255, 255, 255), 2, cv2.LINE_AA)

    if bounds is None:
        return

    x, y, width, height = bounds
    padding = 16
    cv2.rectangle(
        frame,
        (max(0, x - padding), max(0, y - padding)),
        (min(frame.shape[1] - 1, x + width + padding), min(frame.shape[0] - 1, y + height + padding)),
        (0, 230, 140),
        3,
        cv2.LINE_AA,
    )


def draw_play_hud(
    frame,
    elapsed_seconds: float,
    moves: int,
    pinch_active: bool,
    selected_tile: int | None,
    difficulty: int,
) -> None:
    _draw_top_bar(frame)
    _put(frame, "Gesture Puzzle", (28, 46), 0.95, (245, 247, 250), 2)
    _put(frame, f"Time: {elapsed_seconds:05.1f}s", (410, 45), 0.68, (245, 247, 250), 2)
    _put(frame, f"Moves: {moves}", (610, 45), 0.68, (245, 247, 250), 2)
    _put(frame, f"Grid: {difficulty}x{difficulty}", (770, 45), 0.68, (245, 247, 250), 2)
    pinch_text = "Pinch: Grab" if pinch_active else "Pinch: Open"
    pinch_color = (0, 230, 140) if pinch_active else (0, 180, 255)
    _put(frame, pinch_text, (940, 45), 0.68, pinch_color, 2)

    selected_text = "Selected: None" if selected_tile is None else f"Selected: {selected_tile + 1}"
    _put(frame, selected_text, (28, frame.shape[0] - 30), 0.64, (225, 230, 238), 1)
    _put(frame, "Pinch tile -> drag to another tile -> release to swap   R: Restart", (300, frame.shape[0] - 30), 0.64, (225, 230, 238), 1)


def draw_victory_hud(frame, elapsed_seconds: float, moves: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (8, 10, 14), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    _put_center(frame, "VICTORY", (center_x, center_y - 70), 1.8, (0, 230, 140), 4)
    _put_center(frame, f"Time: {elapsed_seconds:05.1f}s", (center_x, center_y), 0.9, (245, 247, 250), 2)
    _put_center(frame, f"Moves: {moves}", (center_x, center_y + 46), 0.9, (245, 247, 250), 2)
    _put_center(frame, "R: Restart   Q / Esc: Exit", (center_x, center_y + 112), 0.72, (225, 230, 238), 2)


def draw_cursor(frame, point: tuple[int, int] | None, pinch_active: bool) -> None:
    if point is None:
        return

    color = (0, 230, 140) if pinch_active else (0, 180, 255)
    radius = 17 if pinch_active else 13
    cv2.circle(frame, point, radius, color, 3, cv2.LINE_AA)
    cv2.circle(frame, point, 4, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_top_bar(frame) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 74), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)


def _draw_progress(frame, progress: float) -> None:
    x = frame.shape[1] - 290
    y = 31
    width = 230
    height = 14
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 66, 78), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * progress), y + height), (0, 230, 140), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (220, 226, 236), 1)


def _put(frame, text: str, origin: tuple[int, int], scale: float, color, thickness: int) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _put_center(frame, text: str, center: tuple[int, int], scale: float, color, thickness: int) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = (center[0] - size[0] // 2, center[1] + size[1] // 2)
    _put(frame, text, origin, scale, color, thickness)

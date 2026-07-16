from __future__ import annotations

import cv2

from airgesture.ui import theme as ui


def draw_capture_hud(
    frame,
    hand_count: int,
    capture_message: str,
    capture_progress: float,
    difficulty: int,
) -> None:
    _draw_top_bar(frame, "Gesture Puzzle", "Open both hands to auto capture a snapshot.")
    hand_color = ui.GREEN if hand_count >= 2 else ui.YELLOW
    width = frame.shape[1]
    ui.chip(frame, (width - 512, 22, 132, 34), f"HANDS {hand_count}/2", color=hand_color, active=hand_count >= 2)
    ui.chip(frame, (width - 370, 22, 126, 34), f"GRID {difficulty}x{difficulty}", color=ui.CYAN, active=True)
    ui.progress_bar(frame, (width - 224, 34, 176, 12), capture_progress, color=ui.GREEN)

    _draw_bottom_panel(
        frame,
        capture_message,
        "Open hands wide and hold still    3/4 Difficulty    F11 Fullscreen    Space/Enter/C Capture",
        hand_color,
    )


def draw_countdown_hud(frame, remaining_seconds: float, difficulty: int) -> None:
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    ui.blend_rect(frame, (0, 0), (frame.shape[1], frame.shape[0]), (8, 10, 14), 0.58)
    ui.panel(frame, (center_x - 260, center_y - 155, 520, 292), fill=(14, 18, 27), border=ui.CYAN, alpha=0.92)
    number = max(1, int(remaining_seconds) + 1)
    _put_center(frame, str(number), (center_x, center_y - 48), 3.3, ui.GREEN, 7)
    _put_center(
        frame,
        "Puzzle starts in",
        (center_x, center_y - 122),
        0.64,
        ui.TEXT_MUTED,
        1,
    )
    _put_center(
        frame,
        f"Creating {difficulty}x{difficulty} puzzle",
        (center_x, center_y + 72),
        0.86,
        ui.TEXT,
        2,
    )
    _put_center(frame, "Hold your hand ready over the board.", (center_x, center_y + 114), 0.54, ui.TEXT_MUTED, 1)


def draw_capture_gesture(frame, points, bounds) -> None:
    height, width = frame.shape[:2]
    guide_width = int(width * 0.62)
    guide_height = int(height * 0.54)
    guide_x = (width - guide_width) // 2
    guide_y = (height - guide_height) // 2 + 24
    _draw_corner_guide(frame, (guide_x, guide_y, guide_width, guide_height), ui.CYAN)

    for point in points:
        cv2.circle(frame, point, 12, ui.GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 17, ui.WHITE, 2, cv2.LINE_AA)

    if bounds is None:
        return

    x, y, width, height = bounds
    padding = 16
    cv2.rectangle(
        frame,
        (max(0, x - padding), max(0, y - padding)),
        (min(frame.shape[1] - 1, x + width + padding), min(frame.shape[0] - 1, y + height + padding)),
        ui.GREEN,
        3,
        cv2.LINE_AA,
    )


def _draw_corner_guide(
    frame,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x, y, width, height = rect
    corner = 72
    for start, end in [
        ((x, y), (x + corner, y)),
        ((x, y), (x, y + corner)),
        ((x + width - corner, y), (x + width, y)),
        ((x + width, y), (x + width, y + corner)),
        ((x, y + height), (x + corner, y + height)),
        ((x, y + height - corner), (x, y + height)),
        ((x + width - corner, y + height), (x + width, y + height)),
        ((x + width, y + height - corner), (x + width, y + height)),
    ]:
        cv2.line(frame, start, end, color, 3, cv2.LINE_AA)


def draw_play_hud(
    frame,
    elapsed_seconds: float,
    moves: int,
    pinch_active: bool,
    selected_tile: int | None,
    difficulty: int,
) -> None:
    _draw_top_bar(frame, "Gesture Puzzle", "Pinch a tile, drag, and release to swap.")
    width = frame.shape[1]
    ui.chip(frame, (width - 690, 22, 154, 34), f"TIME {elapsed_seconds:05.1f}s", color=ui.CYAN, active=False)
    ui.chip(frame, (width - 526, 22, 118, 34), f"MOVES {moves}", color=ui.GREEN, active=False)
    ui.chip(frame, (width - 398, 22, 120, 34), f"GRID {difficulty}x{difficulty}", color=ui.BLUE, active=False)
    pinch_text = "Pinch: Grab" if pinch_active else "Pinch: Open"
    pinch_color = ui.GREEN if pinch_active else ui.CYAN
    ui.chip(frame, (width - 268, 22, 156, 34), pinch_text.upper(), color=pinch_color, active=pinch_active)

    selected_text = "Selected: None" if selected_tile is None else f"Selected: {selected_tile + 1}"
    _draw_bottom_panel(
        frame,
        selected_text,
        "Pinch, drag, and release to swap    R Restart    F11 Fullscreen    Q/Esc Exit",
        ui.GREEN if selected_tile is not None else ui.CYAN,
    )


def draw_victory_hud(frame, elapsed_seconds: float, moves: int) -> None:
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    ui.blend_rect(frame, (0, 0), (frame.shape[1], frame.shape[0]), (8, 10, 14), 0.70)
    ui.panel(frame, (center_x - 290, center_y - 160, 580, 300), fill=(14, 18, 27), border=ui.GREEN, alpha=0.94)
    _put_center(frame, "PUZZLE COMPLETE", (center_x, center_y - 92), 1.28, ui.GREEN, 3)
    ui.chip(frame, (center_x - 210, center_y - 18, 188, 42), f"TIME {elapsed_seconds:05.1f}s", color=ui.CYAN, active=True)
    ui.chip(frame, (center_x + 22, center_y - 18, 188, 42), f"MOVES {moves}", color=ui.GREEN, active=True)
    _put_center(frame, "R Restart    F11 Fullscreen    Q/Esc Exit", (center_x, center_y + 94), 0.62, ui.TEXT_MUTED, 1)


def draw_cursor(frame, point: tuple[int, int] | None, pinch_active: bool) -> None:
    if point is None:
        return

    color = ui.GREEN if pinch_active else ui.CYAN
    radius = 19 if pinch_active else 14
    cv2.circle(frame, point, radius + 4, (8, 10, 14), 2, cv2.LINE_AA)
    cv2.circle(frame, point, radius, color, 3, cv2.LINE_AA)
    cv2.circle(frame, point, 4, ui.WHITE, -1, cv2.LINE_AA)


def _draw_top_bar(frame, title: str, subtitle: str) -> None:
    ui.blend_rect(frame, (0, 0), (frame.shape[1], 78), (10, 13, 20), 0.82)
    cv2.line(frame, (0, 78), (frame.shape[1], 78), ui.BORDER_SOFT, 1, cv2.LINE_AA)
    _put(frame, title, (28, 44), 0.92, ui.TEXT, 2)
    _put(frame, subtitle, (30, 67), 0.42, ui.TEXT_MUTED, 1)


def _draw_bottom_panel(
    frame,
    primary: str,
    secondary: str,
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    ui.blend_rect(frame, (0, height - 62), (width, height), (10, 13, 20), 0.80)
    cv2.line(frame, (0, height - 62), (width, height - 62), ui.BORDER_SOFT, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (26, height - 49), (30, height - 16), color, -1)
    _put(frame, primary, (42, height - 34), 0.58, color, 2)
    _put(frame, secondary, (42, height - 13), 0.46, ui.TEXT_MUTED, 1)


def _draw_progress(frame, progress: float) -> None:
    x = frame.shape[1] - 290
    y = 31
    width = 230
    height = 14
    ui.progress_bar(frame, (x, y, width, height), progress, color=ui.GREEN)


def _put(frame, text: str, origin: tuple[int, int], scale: float, color, thickness: int) -> None:
    ui.put_text(frame, text, origin, scale, color, thickness)


def _put_center(frame, text: str, center: tuple[int, int], scale: float, color, thickness: int) -> None:
    ui.put_center(frame, text, center, scale, color, thickness)

from __future__ import annotations

import cv2

from airgesture.ui import theme as ui


PUZZLE_PAPER = (250, 250, 247)
PUZZLE_INK = (8, 9, 10)
PUZZLE_BODY = (46, 48, 52)
PUZZLE_CYAN = (220, 222, 24)
PUZZLE_YELLOW = (0, 225, 255)
PUZZLE_PINK = (168, 70, 242)
PUZZLE_GREEN = (57, 201, 49)
PUZZLE_FONT = cv2.FONT_HERSHEY_DUPLEX
PUZZLE_BODY_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_capture_hud(
    frame,
    hand_count: int,
    capture_message: str,
    capture_progress: float,
    difficulty: int,
    fps: float = 0.0,
) -> None:
    camera_frame = frame.copy()
    height, width = frame.shape[:2]
    layout = ui.Layout(width, height)

    frame[:] = PUZZLE_YELLOW
    _draw_halftone(frame, layout)
    cv2.rectangle(frame, layout.point(0, 0), layout.point(1280, 126), PUZZLE_PAPER, -1)
    cv2.line(frame, layout.point(0, 126), layout.point(1280, 126), PUZZLE_INK, layout.px(2), cv2.LINE_AA)

    _puzzle_text(frame, "GESTURE PUZZLE", layout.point(36, 58), layout.font(1.20), PUZZLE_INK, layout.px(4))
    _puzzle_text(
        frame,
        "Open both hands to auto capture a snapshot.",
        layout.point(38, 83),
        layout.font(0.45),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    _draw_capture_status_cards(
        frame,
        layout,
        hand_count=hand_count,
        difficulty=difficulty,
        capture_progress=capture_progress,
        fps=fps,
    )

    camera_rect = layout.rect(172, 138, 936, 443)
    _draw_camera_panel(frame, camera_frame, camera_rect, layout)
    _draw_capture_footer(
        frame,
        layout,
        capture_message=capture_message,
        difficulty=difficulty,
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
    layout = ui.Layout(frame.shape[1], frame.shape[0])
    _draw_play_top_bar(
        frame,
        layout,
        elapsed_seconds=elapsed_seconds,
        moves=moves,
        pinch_active=pinch_active,
        difficulty=difficulty,
    )
    _draw_play_bottom_bar(
        frame,
        layout,
        selected_tile=selected_tile,
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


def _draw_capture_status_cards(
    frame,
    layout: ui.Layout,
    *,
    hand_count: int,
    difficulty: int,
    capture_progress: float,
    fps: float,
) -> None:
    _puzzle_panel(frame, layout.rect(720, 25, 176, 52), layout, PUZZLE_YELLOW)
    _draw_hand_icon(frame, layout.point(740, 37), layout.px(24), PUZZLE_INK)
    _puzzle_text(
        frame,
        "HANDS",
        layout.point(778, 58),
        layout.font(0.40),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    _puzzle_text(
        frame,
        f"{hand_count}/2",
        layout.point(844, 58),
        layout.font(0.44),
        PUZZLE_PINK if hand_count < 2 else PUZZLE_GREEN,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    progress_width = layout.px(153)
    progress_x, progress_y = layout.point(732, 69)
    cv2.rectangle(frame, (progress_x, progress_y), (progress_x + progress_width, progress_y + layout.px(3)), PUZZLE_INK, -1)
    cv2.rectangle(
        frame,
        (progress_x, progress_y),
        (progress_x + int(progress_width * max(0.0, min(1.0, capture_progress))), progress_y + layout.px(3)),
        PUZZLE_PINK,
        -1,
    )

    _puzzle_panel(frame, layout.rect(912, 25, 146, 52), layout, PUZZLE_CYAN)
    grid_x, grid_y = layout.point(929, 39)
    cell = layout.px(6)
    for row in range(3):
        for column in range(3):
            cv2.rectangle(
                frame,
                (grid_x + column * (cell + layout.px(2)), grid_y + row * (cell + layout.px(2))),
                (grid_x + column * (cell + layout.px(2)) + cell, grid_y + row * (cell + layout.px(2)) + cell),
                PUZZLE_INK,
                -1,
            )
    _puzzle_text(
        frame,
        "GRID",
        layout.point(970, 58),
        layout.font(0.40),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    _puzzle_text(
        frame,
        f"{difficulty}x{difficulty}",
        layout.point(1020, 58),
        layout.font(0.39),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )

    _puzzle_panel(frame, layout.rect(1074, 25, 170, 52), layout, PUZZLE_PAPER)
    _puzzle_text(
        frame,
        "FPS",
        layout.point(1090, 58),
        layout.font(0.40),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    _puzzle_text(
        frame,
        f"{fps:04.1f}",
        layout.point(1131, 58),
        layout.font(0.40),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    for index, bar_height in enumerate((8, 14, 20)):
        x = layout.x(1202 + index * 10)
        bottom = layout.y(62)
        cv2.rectangle(
            frame,
            (x, bottom - layout.px(bar_height)),
            (x + layout.px(6), bottom),
            PUZZLE_GREEN,
            -1,
        )


def _draw_play_top_bar(
    frame,
    layout: ui.Layout,
    *,
    elapsed_seconds: float,
    moves: int,
    pinch_active: bool,
    difficulty: int,
) -> None:
    cv2.rectangle(frame, layout.point(0, 0), layout.point(1280, 92), PUZZLE_PAPER, -1)
    cv2.line(frame, layout.point(0, 92), layout.point(1280, 92), PUZZLE_INK, layout.px(3), cv2.LINE_AA)

    _puzzle_text(
        frame,
        "GESTURE PUZZLE",
        layout.point(30, 53),
        layout.font(0.90),
        PUZZLE_INK,
        layout.px(3),
    )
    _puzzle_text(
        frame,
        "Pinch a tile, drag, and release to swap.",
        layout.point(32, 77),
        layout.font(0.40),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )

    cards = [
        ((620, 20, 150, 52), "TIME", f"{elapsed_seconds:05.1f}s", PUZZLE_CYAN),
        ((784, 20, 125, 52), "MOVES", str(moves), PUZZLE_GREEN),
        ((923, 20, 135, 52), "GRID", f"{difficulty} x {difficulty}", (255, 135, 24)),
        ((1072, 20, 176, 52), "PINCH", "GRAB" if pinch_active else "OPEN", PUZZLE_GREEN if pinch_active else PUZZLE_CYAN),
    ]
    for rect, label, value, accent in cards:
        _draw_play_status_card(frame, layout, layout.rect(*rect), label, value, accent)


def _draw_play_status_card(
    frame,
    layout: ui.Layout,
    rect: tuple[int, int, int, int],
    label: str,
    value: str,
    accent: tuple[int, int, int],
) -> None:
    _puzzle_panel(frame, rect, layout, PUZZLE_PAPER, thickness=2)
    x, y, width, _ = rect
    cv2.rectangle(
        frame,
        (x + layout.px(7), y + layout.px(7)),
        (x + layout.px(12), y + layout.px(45)),
        accent,
        -1,
    )
    _puzzle_text(
        frame,
        label,
        (x + layout.px(22), y + layout.px(20)),
        layout.font(0.25),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    _puzzle_text(
        frame,
        value,
        (x + layout.px(22), y + layout.px(42)),
        layout.font(0.42),
        PUZZLE_INK,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )


def _draw_play_bottom_bar(
    frame,
    layout: ui.Layout,
    *,
    selected_tile: int | None,
) -> None:
    cv2.rectangle(frame, layout.point(0, 650), layout.point(1280, 720), PUZZLE_PAPER, -1)
    cv2.line(frame, layout.point(0, 650), layout.point(1280, 650), PUZZLE_INK, layout.px(3), cv2.LINE_AA)
    accent = PUZZLE_GREEN if selected_tile is not None else PUZZLE_CYAN
    cv2.rectangle(frame, layout.point(28, 664), layout.point(34, 708), accent, -1)

    _puzzle_text(
        frame,
        "SELECTED",
        layout.point(48, 677),
        layout.font(0.25),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    _puzzle_text(
        frame,
        "NONE" if selected_tile is None else str(selected_tile + 1),
        layout.point(48, 699),
        layout.font(0.43),
        PUZZLE_INK,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    _puzzle_text(
        frame,
        "Pinch, drag, and release to swap",
        layout.point(158, 690),
        layout.font(0.38),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )

    controls = [
        (760, 660, 136, 48, "R", "RESTART"),
        (914, 660, 150, 48, "F11", "FULLSCREEN"),
        (1082, 660, 166, 48, "Q / ESC", "EXIT"),
    ]
    for x, y, width, height, primary, secondary in controls:
        control_rect = layout.rect(x, y, width, height)
        _puzzle_panel(frame, control_rect, layout, PUZZLE_PAPER, shadow=False, thickness=2)
        left, top, control_width, _ = control_rect
        secondary_scale = layout.font(0.25)
        secondary_size, _ = cv2.getTextSize(
            secondary,
            PUZZLE_BODY_FONT,
            secondary_scale,
            layout.px(1),
        )
        _puzzle_text(
            frame,
            primary,
            (left + layout.px(12), top + layout.px(30)),
            layout.font(0.40 if len(primary) <= 3 else 0.31),
            PUZZLE_INK,
            layout.px(1),
            font=PUZZLE_BODY_FONT,
        )
        _puzzle_text(
            frame,
            secondary,
            (left + control_width - layout.px(12) - secondary_size[0], top + layout.px(29)),
            secondary_scale,
            PUZZLE_BODY,
            layout.px(1),
            font=PUZZLE_BODY_FONT,
        )


def _draw_camera_panel(
    frame,
    camera_frame,
    rect: tuple[int, int, int, int],
    layout: ui.Layout,
) -> None:
    x, y, width, height = rect
    offset = layout.px(7)
    cv2.rectangle(
        frame,
        (x + offset, y + offset),
        (x + width + offset, y + height + offset),
        PUZZLE_INK,
        -1,
    )
    source_height, source_width = camera_frame.shape[:2]
    target_aspect = width / max(height, 1)
    source_aspect = source_width / max(source_height, 1)
    crop_x = 0
    crop_y = 0
    crop_width = source_width
    crop_height = source_height
    if source_aspect < target_aspect:
        crop_height = max(1, int(round(source_width / target_aspect)))
        crop_y = max(0, (source_height - crop_height) // 2)
    elif source_aspect > target_aspect:
        crop_width = max(1, int(round(source_height * target_aspect)))
        crop_x = max(0, (source_width - crop_width) // 2)
    visible = camera_frame[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    resized = cv2.resize(visible, (width, height), interpolation=cv2.INTER_AREA)
    frame[y : y + height, x : x + width] = resized
    cv2.rectangle(frame, (x, y), (x + width, y + height), PUZZLE_INK, layout.px(3), cv2.LINE_AA)


def _draw_capture_footer(
    frame,
    layout: ui.Layout,
    *,
    capture_message: str,
    difficulty: int,
) -> None:
    rect = layout.rect(30, 596, 1218, 98)
    _puzzle_panel(frame, rect, layout, PUZZLE_PAPER)
    _draw_two_hands_badge(frame, layout, layout.rect(42, 610, 130, 66))

    scale = layout.font(0.46)
    message = capture_message.upper()
    max_width = layout.px(470)
    while cv2.getTextSize(message, PUZZLE_BODY_FONT, scale, layout.px(1))[0][0] > max_width and scale > layout.font(0.30):
        scale -= 0.02
    _puzzle_text(
        frame,
        message,
        layout.point(190, 637),
        scale,
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )
    _puzzle_text(
        frame,
        "Open hands wide and hold still",
        layout.point(190, 665),
        layout.font(0.39),
        PUZZLE_BODY,
        layout.px(1),
        font=PUZZLE_BODY_FONT,
    )

    controls = [
        (690, 610, 92, 62, "3 / 4", f"GRID {difficulty}x{difficulty}", PUZZLE_PINK),
        (794, 610, 90, 62, "F11", "FULLSCREEN", PUZZLE_PAPER),
        (896, 610, 126, 62, "SPACE/ENTER", "CAPTURE", PUZZLE_PAPER),
        (1034, 610, 80, 62, "C", "CAPTURE", PUZZLE_PAPER),
        (1126, 610, 104, 62, "Q/ESC", "EXIT", PUZZLE_PAPER),
    ]
    for x, y, width, height, primary, secondary, fill in controls:
        control_rect = layout.rect(x, y, width, height)
        _puzzle_panel(frame, control_rect, layout, fill)
        left, top, control_width, _ = control_rect
        _puzzle_center(
            frame,
            primary,
            (left + control_width // 2, top + layout.px(25)),
            layout.font(0.38 if len(primary) <= 5 else 0.27),
            PUZZLE_BODY,
            layout.px(1),
            font=PUZZLE_BODY_FONT,
        )
        _puzzle_center(
            frame,
            secondary,
            (left + control_width // 2, top + layout.px(46)),
            layout.font(0.24),
            PUZZLE_BODY,
            layout.px(1),
            font=PUZZLE_BODY_FONT,
        )


def _draw_two_hands_badge(
    frame,
    layout: ui.Layout,
    rect: tuple[int, int, int, int],
) -> None:
    _puzzle_panel(frame, rect, layout, PUZZLE_PINK, shadow=False, thickness=2)
    x, y, width, height = rect
    for center_x in (x + width // 3, x + width * 2 // 3):
        palm_top = y + layout.px(28)
        cv2.ellipse(
            frame,
            (center_x, palm_top + layout.px(14)),
            (layout.px(15), layout.px(19)),
            0,
            0,
            180,
            PUZZLE_INK,
            layout.px(3),
            cv2.LINE_AA,
        )
        for finger in range(4):
            finger_x = center_x - layout.px(11) + finger * layout.px(7)
            finger_height = layout.px(18 + (finger % 2) * 5)
            cv2.line(
                frame,
                (finger_x, palm_top + layout.px(8)),
                (finger_x, palm_top + layout.px(8) - finger_height),
                PUZZLE_INK,
                layout.px(3),
                cv2.LINE_AA,
            )


def _draw_hand_icon(
    frame,
    origin: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    cv2.ellipse(frame, (x + size // 2, y + size * 2 // 3), (size // 3, size // 3), 0, 0, 180, color, 2, cv2.LINE_AA)
    for index in range(4):
        finger_x = x + size // 4 + index * max(2, size // 6)
        cv2.line(frame, (finger_x, y + size * 2 // 3), (finger_x, y + index % 2), color, 2, cv2.LINE_AA)


def _draw_halftone(frame, layout: ui.Layout) -> None:
    for side_x in (22, 1165):
        for row in range(12):
            for column in range(9):
                distance = abs(row - 5.5) + abs(column - 4)
                radius = max(1, layout.px(3.2 - distance * 0.20))
                center = layout.point(side_x + column * 12, 292 + row * 19)
                cv2.circle(frame, center, radius, PUZZLE_INK, -1, cv2.LINE_AA)


def _puzzle_panel(
    frame,
    rect: tuple[int, int, int, int],
    layout: ui.Layout,
    fill: tuple[int, int, int],
    *,
    shadow: bool = True,
    thickness: int = 3,
) -> None:
    x, y, width, height = rect
    if shadow:
        offset = layout.px(6)
        cv2.rectangle(frame, (x + offset, y + offset), (x + width + offset, y + height + offset), PUZZLE_INK, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), PUZZLE_INK, layout.px(thickness), cv2.LINE_AA)


def _puzzle_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    *,
    font: int = PUZZLE_FONT,
) -> None:
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def _puzzle_center(
    frame,
    text: str,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    *,
    font: int = PUZZLE_FONT,
) -> None:
    size, _ = cv2.getTextSize(text, font, scale, thickness)
    _puzzle_text(
        frame,
        text,
        (center[0] - size[0] // 2, center[1] + size[1] // 2),
        scale,
        color,
        thickness,
        font=font,
    )


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

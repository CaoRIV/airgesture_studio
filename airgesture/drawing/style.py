from __future__ import annotations

import cv2

from airgesture.ui import theme as ui


PAPER = (247, 247, 243)
PAPER_ALT = (252, 252, 249)
INK = (8, 9, 10)
GRID = (218, 220, 215)
GRID_STRONG = (184, 194, 191)
CYAN = (232, 190, 0)
LIME = (34, 239, 169)
YELLOW = (0, 220, 255)
RED = (66, 86, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX


def put_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = INK,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def put_center(
    frame,
    text: str,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = INK,
    thickness: int = 1,
) -> None:
    size, _ = cv2.getTextSize(text, FONT, scale, thickness)
    origin = (center[0] - size[0] // 2, center[1] + size[1] // 2)
    put_text(frame, text, origin, scale, color, thickness)


def rounded_rect(
    frame,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int,
    radius: int,
) -> None:
    x, y, width, height = rect
    radius = max(1, min(radius, width // 2, height // 2))
    if thickness < 0:
        cv2.rectangle(frame, (x + radius, y), (x + width - radius, y + height), color, -1)
        cv2.rectangle(frame, (x, y + radius), (x + width, y + height - radius), color, -1)
        for center in (
            (x + radius, y + radius),
            (x + width - radius, y + radius),
            (x + radius, y + height - radius),
            (x + width - radius, y + height - radius),
        ):
            cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
        return

    cv2.line(frame, (x + radius, y), (x + width - radius, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + radius, y + height), (x + width - radius, y + height), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + radius), (x, y + height - radius), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + width, y + radius), (x + width, y + height - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x + radius, y + radius), (radius, radius), 0, 180, 270, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x + width - radius, y + radius), (radius, radius), 0, 270, 360, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x + radius, y + height - radius), (radius, radius), 0, 90, 180, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x + width - radius, y + height - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def panel(
    frame,
    rect: tuple[int, int, int, int],
    layout: ui.Layout,
    *,
    fill: tuple[int, int, int] = PAPER_ALT,
    radius: int = 5,
    shadow: bool = True,
    thickness: int = 3,
) -> None:
    x, y, width, height = rect
    scaled_radius = layout.px(radius)
    if shadow:
        offset = layout.px(6)
        rounded_rect(
            frame,
            (x + offset, y + offset, width, height),
            INK,
            -1,
            scaled_radius,
        )
    rounded_rect(frame, rect, fill, -1, scaled_radius)
    rounded_rect(frame, rect, INK, layout.px(thickness), scaled_radius)


def dashed_line(
    frame,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int,
    gap: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = max(abs(x2 - x1), abs(y2 - y1))
    if length == 0:
        return
    step = max(1, dash + gap)
    for offset in range(0, length, step):
        finish = min(length, offset + dash)
        sx = int(round(x1 + (x2 - x1) * offset / length))
        sy = int(round(y1 + (y2 - y1) * offset / length))
        ex = int(round(x1 + (x2 - x1) * finish / length))
        ey = int(round(y1 + (y2 - y1) * finish / length))
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


def draw_background(frame, layout: ui.Layout) -> None:
    frame[:] = PAPER
    for x in range(32, 1280, 54):
        dashed_line(
            frame,
            layout.point(x, 0),
            layout.point(x, 720),
            GRID,
            layout.px(1),
            layout.px(4),
            layout.px(5),
        )
    for y in range(28, 720, 46):
        dashed_line(
            frame,
            layout.point(0, y),
            layout.point(1280, y),
            GRID,
            layout.px(1),
            layout.px(4),
            layout.px(5),
        )


def draw_target(
    frame,
    layout: ui.Layout,
    center: tuple[int, int],
    color: tuple[int, int, int] = CYAN,
    radius: int = 22,
) -> None:
    cx, cy = layout.point(*center)
    scaled_radius = layout.px(radius)
    cv2.circle(frame, (cx, cy), scaled_radius, color, layout.px(2), cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), layout.px(3), color, -1, cv2.LINE_AA)
    cv2.line(
        frame,
        (cx - scaled_radius - layout.px(8), cy),
        (cx + scaled_radius + layout.px(8), cy),
        color,
        layout.px(2),
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (cx, cy - scaled_radius - layout.px(8)),
        (cx, cy + scaled_radius + layout.px(8)),
        color,
        layout.px(2),
        cv2.LINE_AA,
    )


def corner_bracket(
    frame,
    layout: ui.Layout,
    x: int,
    y: int,
    dx: int,
    dy: int,
    color: tuple[int, int, int] = CYAN,
    thickness: int = 2,
) -> None:
    origin = layout.point(x, y)
    cv2.line(frame, origin, layout.point(x + dx, y), color, layout.px(thickness), cv2.LINE_AA)
    cv2.line(frame, origin, layout.point(x, y + dy), color, layout.px(thickness), cv2.LINE_AA)

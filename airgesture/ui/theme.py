from __future__ import annotations

import cv2


FONT = cv2.FONT_HERSHEY_SIMPLEX

BG = (10, 12, 17)
SURFACE = (20, 24, 32)
SURFACE_ALT = (29, 35, 45)
SURFACE_RAISED = (38, 46, 59)
BORDER = (74, 88, 108)
BORDER_SOFT = (43, 52, 66)
TEXT = (245, 247, 250)
TEXT_MUTED = (174, 187, 204)
TEXT_DIM = (118, 132, 150)
CYAN = (255, 205, 0)
BLUE = (255, 135, 24)
GREEN = (118, 238, 72)
YELLOW = (0, 216, 255)
RED = (72, 84, 255)
WHITE = (250, 250, 250)


def blend_rect(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_background(frame) -> None:
    height, width = frame.shape[:2]
    frame[:] = BG

    grid_color = (18, 23, 31)
    for x in range(0, width, 48):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1, cv2.LINE_AA)
    for y in range(0, height, 48):
        cv2.line(frame, (0, y), (width, y), grid_color, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (width, 86), (13, 16, 23), -1)
    cv2.line(frame, (0, 86), (width, 86), BORDER_SOFT, 1, cv2.LINE_AA)
    cv2.line(frame, (0, height - 76), (width, height - 76), BORDER_SOFT, 1, cv2.LINE_AA)

    for offset in range(-width, width, 170):
        cv2.line(
            frame,
            (offset, height),
            (offset + 360, 0),
            (14, 26, 34),
            1,
            cv2.LINE_AA,
        )


def panel(
    frame,
    rect: tuple[int, int, int, int],
    fill: tuple[int, int, int] = SURFACE,
    border: tuple[int, int, int] = BORDER_SOFT,
    alpha: float = 0.92,
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    x, y, width, height = rect
    if shadow:
        blend_rect(frame, (x + 6, y + 8), (x + width + 6, y + height + 8), (0, 0, 0), 0.22)
    blend_rect(frame, (x, y), (x + width, y + height), fill, alpha)
    cv2.rectangle(frame, (x, y), (x + width, y + height), border, thickness, cv2.LINE_AA)


def accent_bar(
    frame,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int] = CYAN,
) -> None:
    x, y, width, height = rect
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)


def chip(
    frame,
    rect: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int] = CYAN,
    active: bool = False,
) -> None:
    fill = SURFACE_RAISED if active else SURFACE_ALT
    border = color if active else BORDER_SOFT
    panel(frame, rect, fill=fill, border=border, alpha=0.88, thickness=1, shadow=False)
    x, y, width, height = rect
    cv2.rectangle(frame, (x, y), (x + 4, y + height), color, -1)
    put_text(frame, text, (x + 14, y + height // 2 + 7), 0.54, TEXT, 1)


def progress_bar(
    frame,
    rect: tuple[int, int, int, int],
    progress: float,
    color: tuple[int, int, int] = GREEN,
) -> None:
    x, y, width, height = rect
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (48, 57, 70), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * progress), y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), BORDER, 1, cv2.LINE_AA)


def put_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def put_center(
    frame,
    text: str,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    size, _ = cv2.getTextSize(text, FONT, scale, thickness)
    origin = (center[0] - size[0] // 2, center[1] + size[1] // 2)
    put_text(frame, text, origin, scale, color, thickness)

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from airgesture.calibration import CalibrationConfig, run_calibration
from airgesture.drawing import main as drawing_main
from airgesture.puzzle import main as puzzle_main
from airgesture.ui import theme as ui


WINDOW_NAME = "AirGesture Studio"
MENU_WIDTH = 1280
MENU_HEIGHT = 720


class MenuAction(Enum):
    DRAWING = "Air Drawing"
    PUZZLE = "Gesture Puzzle"
    QUIT = "Quit"


@dataclass(frozen=True)
class MenuItem:
    action: MenuAction
    title: str
    subtitle: str
    shortcut: str


MENU_ITEMS = [
    MenuItem(
        MenuAction.DRAWING,
        "Air Drawing",
        "Draw symbols, use toolbar, snap detected letters.",
        "1",
    ),
    MenuItem(
        MenuAction.PUZZLE,
        "Gesture Puzzle",
        "Capture a webcam shot, solve with pinch swaps.",
        "2",
    ),
    MenuItem(
        MenuAction.QUIT,
        "Quit",
        "Close the studio.",
        "Q",
    ),
]


def main() -> int:
    selected_index = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    while True:
        frame = render_menu(selected_index)
        cv2.imshow(WINDOW_NAME, frame)
        key_code = cv2.waitKey(30) & 0xFF

        if key_code in (27, ord("q"), ord("Q")):
            break
        if key_code in (ord("1"),):
            run_action(MenuAction.DRAWING)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        elif key_code in (ord("2"),):
            run_action(MenuAction.PUZZLE)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        elif key_code in (13, 10):
            action = MENU_ITEMS[selected_index].action
            if action == MenuAction.QUIT:
                break
            run_action(action)
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        elif key_code in (ord("w"), ord("W")):
            selected_index = (selected_index - 1) % len(MENU_ITEMS)
        elif key_code in (ord("s"), ord("S")):
            selected_index = (selected_index + 1) % len(MENU_ITEMS)
        elif key_code == 0:
            # Some OpenCV builds report arrow keys through a second waitKey call.
            selected_index = selected_index

    cv2.destroyAllWindows()
    return 0


def run_action(action: MenuAction) -> None:
    cv2.destroyWindow(WINDOW_NAME)
    if action == MenuAction.DRAWING:
        if run_calibration(CalibrationConfig("Air Drawing needs one clearly visible hand.", 1)):
            drawing_main.main()
    elif action == MenuAction.PUZZLE:
        if run_calibration(CalibrationConfig("Gesture Puzzle works best with two visible hands.", 2)):
            puzzle_main.main()


def render_menu(selected_index: int):
    frame = np.zeros((MENU_HEIGHT, MENU_WIDTH, 3), dtype=np.uint8)
    draw_background(frame)

    ui.put_text(frame, "COMPUTER VISION PLAYGROUND", (70, 62), 0.50, ui.CYAN, 1)
    ui.put_text(frame, "AirGesture Studio", (70, 132), 1.62, ui.TEXT, 4)
    ui.put_text(
        frame,
        "Draw and play with hand gestures.",
        (74, 178),
        0.68,
        ui.TEXT_MUTED,
        1,
    )

    draw_mode_visual(frame)
    draw_system_strip(frame)

    ui.panel(frame, (612, 154, 594, 420), fill=(16, 20, 29), border=ui.BORDER_SOFT, alpha=0.90)
    ui.put_text(frame, "Select Experience", (640, 202), 0.82, ui.TEXT, 2)
    ui.put_text(frame, "Use number keys or W/S + Enter.", (642, 232), 0.52, ui.TEXT_MUTED, 1)

    start_y = 266
    for index, item in enumerate(MENU_ITEMS):
        draw_menu_item(
            frame,
            item,
            index=index,
            selected=index == selected_index,
            origin=(640, start_y + index * 102),
        )

    ui.panel(frame, (58, MENU_HEIGHT - 68, MENU_WIDTH - 116, 42), fill=(16, 20, 28), border=ui.BORDER_SOFT, alpha=0.82, shadow=False)
    ui.put_text(
        frame,
        "1 Drawing    2 Puzzle    W/S Select    Enter Open    Q/Esc Quit",
        (78, MENU_HEIGHT - 41),
        0.62,
        ui.TEXT_MUTED,
        1,
    )
    return frame


def draw_background(frame) -> None:
    ui.draw_background(frame)
    cv2.rectangle(frame, (45, 34), (MENU_WIDTH - 45, MENU_HEIGHT - 34), ui.BORDER_SOFT, 1)
    cv2.line(frame, (58, 86), (470, 86), ui.CYAN, 2, cv2.LINE_AA)
    cv2.line(frame, (470, 86), (570, 86), ui.GREEN, 2, cv2.LINE_AA)


def draw_mode_visual(frame) -> None:
    ui.panel(frame, (70, 232, 490, 260), fill=(17, 22, 31), border=ui.BORDER_SOFT, alpha=0.88)
    ui.put_text(frame, "LIVE GESTURE SURFACE", (98, 278), 0.66, ui.TEXT, 2)
    ui.put_text(frame, "Clear status, big targets, low-latency feedback.", (100, 310), 0.48, ui.TEXT_MUTED, 1)

    trail = np.array(
        [(124, 402), (180, 350), (242, 412), (302, 338), (370, 418), (442, 360)],
        dtype=np.int32,
    )
    cv2.polylines(frame, [trail], False, ui.CYAN, 5, cv2.LINE_AA)
    for point in trail[1::2]:
        cv2.circle(frame, tuple(point), 7, ui.GREEN, -1, cv2.LINE_AA)

    grid_x, grid_y = 378, 345
    cell = 42
    for row in range(3):
        for column in range(3):
            x = grid_x + column * cell
            y = grid_y + row * cell
            fill = (42, 51, 62) if (row + column) % 2 == 0 else (30, 37, 48)
            cv2.rectangle(frame, (x, y), (x + cell - 4, y + cell - 4), fill, -1)
            cv2.rectangle(frame, (x, y), (x + cell - 4, y + cell - 4), ui.BORDER, 1)
    cv2.rectangle(frame, (grid_x + cell, grid_y + cell), (grid_x + cell * 2 - 4, grid_y + cell * 2 - 4), ui.GREEN, 3, cv2.LINE_AA)


def draw_system_strip(frame) -> None:
    ui.panel(frame, (70, 520, 490, 74), fill=(16, 20, 28), border=ui.BORDER_SOFT, alpha=0.88)
    labels = [
        ("WEBCAM", ui.CYAN),
        ("MEDIAPIPE", ui.GREEN),
        ("REALTIME", ui.YELLOW),
    ]
    x = 96
    for label, color in labels:
        ui.chip(frame, (x, 542, 128, 32), label, color=color, active=True)
        x += 144


def draw_menu_item(frame, item: MenuItem, index: int, selected: bool, origin: tuple[int, int]) -> None:
    x, y = origin
    width = 528
    height = 82
    color = menu_color(item.action)
    fill = ui.SURFACE_RAISED if selected else ui.SURFACE
    border = color if selected else ui.BORDER_SOFT
    ui.panel(frame, (x, y, width, height), fill=fill, border=border, alpha=0.92, thickness=2 if selected else 1, shadow=False)
    if selected:
        ui.accent_bar(frame, (x, y, 6, height), color)

    badge_x = x + 20
    badge_y = y + 19
    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + 46, badge_y + 44), color, -1)
    ui.put_center(frame, item.shortcut, (badge_x + 23, badge_y + 22), 0.70, (8, 10, 14), 2)

    ui.put_text(frame, item.title, (x + 88, y + 32), 0.78, ui.TEXT, 2)
    ui.put_text(frame, item.subtitle, (x + 90, y + 62), 0.48, ui.TEXT_MUTED, 1)
    if selected:
        ui.put_text(frame, "READY", (x + width - 82, y + 32), 0.46, color, 1)


def menu_color(action: MenuAction) -> tuple[int, int, int]:
    if action == MenuAction.DRAWING:
        return ui.CYAN
    if action == MenuAction.PUZZLE:
        return ui.GREEN
    return ui.RED


if __name__ == "__main__":
    raise SystemExit(main())

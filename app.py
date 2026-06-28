from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from calibration import CalibrationConfig, run_calibration
import game_main
import main as drawing_main


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
        "Draw in the air, use a gesture toolbar, and snap simple symbols.",
        "1",
    ),
    MenuItem(
        MenuAction.PUZZLE,
        "Gesture Puzzle",
        "Capture a webcam image and solve a hand-controlled 3x3 puzzle.",
        "2",
    ),
    MenuItem(
        MenuAction.QUIT,
        "Quit",
        "Close AirGesture Studio.",
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
            game_main.main()


def render_menu(selected_index: int):
    frame = np.zeros((MENU_HEIGHT, MENU_WIDTH, 3), dtype=np.uint8)
    frame[:] = (14, 16, 22)
    draw_background(frame)

    put_text(frame, "AirGesture Studio", (70, 118), 1.65, (245, 247, 250), 4)
    put_text(
        frame,
        "Air drawing and gesture puzzle gameplay powered by OpenCV and MediaPipe.",
        (74, 164),
        0.68,
        (190, 200, 214),
        1,
    )

    start_y = 245
    for index, item in enumerate(MENU_ITEMS):
        draw_menu_item(
            frame,
            item,
            index=index,
            selected=index == selected_index,
            origin=(72, start_y + index * 118),
        )

    put_text(
        frame,
        "Keys: 1 Drawing   2 Puzzle   W/S Select   Enter Open   Q/Esc Quit",
        (74, MENU_HEIGHT - 54),
        0.62,
        (176, 185, 198),
        1,
    )
    return frame


def draw_background(frame) -> None:
    cv2.rectangle(frame, (0, 0), (MENU_WIDTH, MENU_HEIGHT), (14, 16, 22), -1)
    cv2.circle(frame, (MENU_WIDTH - 140, 140), 190, (24, 42, 58), -1, cv2.LINE_AA)
    cv2.circle(frame, (MENU_WIDTH - 240, MENU_HEIGHT - 110), 230, (24, 52, 44), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (45, 42), (MENU_WIDTH - 45, MENU_HEIGHT - 42), (38, 43, 54), 1)


def draw_menu_item(frame, item: MenuItem, index: int, selected: bool, origin: tuple[int, int]) -> None:
    x, y = origin
    width = 780
    height = 88
    fill = (34, 39, 50) if selected else (25, 29, 38)
    border = (0, 210, 255) if selected else (68, 76, 92)
    cv2.rectangle(frame, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), border, 2 if selected else 1)

    badge_x = x + 20
    badge_y = y + 22
    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + 46, badge_y + 44), border, -1)
    put_text(frame, item.shortcut, (badge_x + 15, badge_y + 31), 0.72, (10, 12, 18), 2)

    put_text(frame, item.title, (x + 88, y + 34), 0.84, (245, 247, 250), 2)
    put_text(frame, item.subtitle, (x + 90, y + 65), 0.52, (184, 193, 207), 1)


def put_text(frame, text: str, origin: tuple[int, int], scale: float, color, thickness: int) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())

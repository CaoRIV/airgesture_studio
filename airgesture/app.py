from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys


if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from airgesture.calibration import CameraCheckConfig, run_camera_check
from airgesture.config import require_valid_settings
from airgesture.core.camera import Camera, CameraConfig
from airgesture.drawing import main as drawing_main
from airgesture.puzzle import main as puzzle_main
from airgesture.ui import theme as ui
from airgesture.ui.runtime_errors import run_with_error_dialog
from airgesture.ui.window import ResponsiveWindow


WINDOW_NAME = "AirGesture Studio"
MENU_WIDTH = 1280
MENU_HEIGHT = 720

# Start-screen palette.  These colors intentionally live beside the menu
# renderer instead of in ``ui.theme`` because the drawing and puzzle views use
# the dark application theme.
MENU_PAPER = (247, 247, 243)
MENU_PAPER_ALT = (252, 252, 249)
MENU_INK = (8, 9, 10)
MENU_GRID = (218, 220, 215)
MENU_GRID_STRONG = (197, 202, 197)
MENU_CYAN = (232, 190, 0)
MENU_LIME = (34, 239, 169)
MENU_YELLOW = (0, 220, 255)
MENU_RED = (66, 86, 255)
MENU_FONT = cv2.FONT_HERSHEY_DUPLEX


class MenuAction(Enum):
    DRAWING = "Air Drawing"
    PUZZLE = "Gesture Puzzle"
    CAMERA_CHECK = "Camera Check"
    QUIT = "Quit"


@dataclass(frozen=True)
class MenuItem:
    action: MenuAction
    title: str
    subtitle: str
    shortcut: str


@dataclass
class MenuPointerState:
    width: int = MENU_WIDTH
    height: int = MENU_HEIGHT
    hovered_index: int | None = None
    activated_action: MenuAction | None = None


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
        MenuAction.CAMERA_CHECK,
        "Camera Check",
        "Optional diagnostics for lighting and hand tracking.",
        "K",
    ),
    MenuItem(
        MenuAction.QUIT,
        "Quit",
        "Close the studio.",
        "Q",
    ),
]


def main() -> int:
    return run_with_error_dialog(WINDOW_NAME, _run_menu)


def _run_menu() -> int:
    settings = require_valid_settings()
    camera_indices = refresh_camera_indices(settings.camera)
    window = ResponsiveWindow(WINDOW_NAME)
    pointer = MenuPointerState()
    try:
        window.create()
        cv2.setMouseCallback(WINDOW_NAME, handle_menu_mouse, pointer)
        return _menu_loop(settings.camera, camera_indices, window, pointer)
    finally:
        cv2.destroyAllWindows()


def _menu_loop(
    camera_config: CameraConfig,
    camera_indices: list[int],
    window: ResponsiveWindow,
    pointer: MenuPointerState,
) -> int:
    selected_index = 0

    while True:
        viewport = window.viewport()
        pointer.width = viewport.width
        pointer.height = viewport.height
        if pointer.hovered_index is not None:
            selected_index = pointer.hovered_index
        frame = render_menu(
            selected_index,
            camera_index=camera_config.camera_index,
            camera_indices=camera_indices,
            width=viewport.width,
            height=viewport.height,
        )
        cv2.imshow(WINDOW_NAME, frame)
        key_code = cv2.waitKeyEx(30)

        if window.handle_window_key(key_code):
            continue

        clicked_action = pointer.activated_action
        pointer.activated_action = None
        if clicked_action is not None:
            if clicked_action == MenuAction.QUIT:
                break
            run_action(clicked_action)
            camera_indices = refresh_camera_indices(camera_config)
            window.recreate()
            cv2.setMouseCallback(WINDOW_NAME, handle_menu_mouse, pointer)
            pointer.hovered_index = None
            continue

        if key_code in (27, ord("q"), ord("Q")):
            break
        if key_code in (ord("1"),):
            run_action(MenuAction.DRAWING)
            camera_indices = refresh_camera_indices(camera_config)
            window.recreate()
            cv2.setMouseCallback(WINDOW_NAME, handle_menu_mouse, pointer)
        elif key_code in (ord("2"),):
            run_action(MenuAction.PUZZLE)
            camera_indices = refresh_camera_indices(camera_config)
            window.recreate()
            cv2.setMouseCallback(WINDOW_NAME, handle_menu_mouse, pointer)
        elif key_code in (ord("k"), ord("K")):
            run_action(MenuAction.CAMERA_CHECK)
            camera_indices = refresh_camera_indices(camera_config)
            window.recreate()
            cv2.setMouseCallback(WINDOW_NAME, handle_menu_mouse, pointer)
        elif key_code in (13, 10):
            action = MENU_ITEMS[selected_index].action
            if action == MenuAction.QUIT:
                break
            run_action(action)
            camera_indices = refresh_camera_indices(camera_config)
            window.recreate()
            cv2.setMouseCallback(WINDOW_NAME, handle_menu_mouse, pointer)
        elif key_code in (ord("w"), ord("W")):
            pointer.hovered_index = None
            selected_index = (selected_index - 1) % len(MENU_ITEMS)
        elif key_code in (ord("s"), ord("S")):
            pointer.hovered_index = None
            selected_index = (selected_index + 1) % len(MENU_ITEMS)
        elif key_code in (ord("a"), ord("A")):
            camera_config.camera_index = cycle_camera_index(
                camera_indices,
                camera_config.camera_index,
                -1,
            )
        elif key_code in (ord("d"), ord("D")):
            camera_config.camera_index = cycle_camera_index(
                camera_indices,
                camera_config.camera_index,
                1,
            )
        elif key_code in (ord("r"), ord("R")):
            camera_indices = refresh_camera_indices(camera_config)
        elif key_code == 0:
            # Some OpenCV builds report arrow keys through a second waitKey call.
            selected_index = selected_index

    return 0


def menu_item_rects(width: int, height: int) -> list[tuple[int, int, int, int]]:
    layout = ui.Layout(width, height)
    return [
        layout.rect(674, 212 + index * 98, 504, 80)
        for index in range(len(MENU_ITEMS))
    ]


def menu_item_at(x: int, y: int, width: int, height: int) -> int | None:
    for index, (left, top, item_width, item_height) in enumerate(
        menu_item_rects(width, height)
    ):
        if left <= x <= left + item_width and top <= y <= top + item_height:
            return index
    return None


def handle_menu_mouse(
    event: int,
    x: int,
    y: int,
    flags: int,
    state: MenuPointerState,
) -> None:
    del flags
    index = menu_item_at(x, y, state.width, state.height)
    if event == cv2.EVENT_MOUSEMOVE:
        state.hovered_index = index
    elif event == cv2.EVENT_LBUTTONUP and index is not None:
        state.hovered_index = index
        state.activated_action = MENU_ITEMS[index].action


def refresh_camera_indices(camera_config: CameraConfig) -> list[int]:
    camera_indices = Camera.discover_indices(
        max_devices=camera_config.discovery_max_devices,
    )
    camera_config.available_indices = tuple(camera_indices)
    if camera_indices and camera_config.camera_index not in camera_indices:
        camera_config.camera_index = camera_indices[0]
    return camera_indices


def cycle_camera_index(
    camera_indices: list[int],
    current_index: int,
    direction: int,
) -> int:
    if not camera_indices:
        return current_index
    try:
        current_position = camera_indices.index(current_index)
    except ValueError:
        return camera_indices[0]
    return camera_indices[(current_position + direction) % len(camera_indices)]


def run_action(action: MenuAction) -> None:
    cv2.destroyWindow(WINDOW_NAME)
    if action == MenuAction.DRAWING:
        drawing_main.main()
    elif action == MenuAction.PUZZLE:
        puzzle_main.main()
    elif action == MenuAction.CAMERA_CHECK:
        run_camera_check(
            CameraCheckConfig(
                title="Optional diagnostics for camera, lighting, and tracking.",
                required_hands=1,
            )
        )


def render_menu(
    selected_index: int,
    camera_index: int = 0,
    camera_indices: list[int] | None = None,
    width: int = MENU_WIDTH,
    height: int = MENU_HEIGHT,
):
    frame = np.full((height, width, 3), MENU_PAPER, dtype=np.uint8)
    layout = ui.Layout(width, height)
    draw_background(frame, layout)

    _outlined_label(
        frame,
        "COMPUTER VISION PLAYGROUND",
        layout.rect(98, 64, 300, 32),
        layout,
    )
    _put_text(
        frame,
        "AirGesture Studio",
        layout.point(90, 164),
        layout.font(1.62),
        MENU_INK,
        layout.px(4),
    )
    cv2.line(
        frame,
        layout.point(92, 187),
        layout.point(465, 187),
        MENU_CYAN,
        layout.px(5),
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        layout.point(465, 187),
        layout.point(594, 187),
        MENU_LIME,
        layout.px(5),
        cv2.LINE_AA,
    )
    _put_text(
        frame,
        "Draw and play with hand gestures.",
        layout.point(92, 220),
        layout.font(0.61),
        MENU_INK,
        layout.px(1),
    )

    draw_mode_visual(frame, layout)
    draw_system_strip(frame, layout, camera_index, camera_indices)

    _menu_panel(frame, layout.rect(645, 100, 560, 505), layout, radius=7)
    _put_text(
        frame,
        "Select mode",
        layout.point(680, 157),
        layout.font(1.12),
        MENU_INK,
        layout.px(3),
    )
    _dashed_line(frame, layout.point(682, 170), layout.point(1082, 170), MENU_CYAN, layout.px(2), layout.px(8), layout.px(6))
    _put_text(
        frame,
        "Use mouse, shortcuts, or W/S + Enter.",
        layout.point(682, 194),
        layout.font(0.43),
        MENU_INK,
        layout.px(1),
    )
    _draw_target(frame, layout, (1146, 151), MENU_CYAN, 22)

    start_y = 212
    for index, item in enumerate(MENU_ITEMS):
        draw_menu_item(
            frame,
            item,
            index=index,
            selected=index == selected_index,
            origin=(674, start_y + index * 98),
            layout=layout,
        )

    _draw_shortcut_strip(frame, layout)
    return frame


def draw_background(frame, layout: ui.Layout | None = None) -> None:
    layout = layout or ui.layout_for(frame)
    frame[:] = MENU_PAPER

    # Technical drawing grid across the entire start screen.
    for x in range(32, MENU_WIDTH, 54):
        _dashed_line(
            frame,
            layout.point(x, 0),
            layout.point(x, MENU_HEIGHT),
            MENU_GRID,
            layout.px(1),
            layout.px(4),
            layout.px(5),
        )
    for y in range(74, MENU_HEIGHT, 51):
        _dashed_line(
            frame,
            layout.point(0, y),
            layout.point(MENU_WIDTH, y),
            MENU_GRID,
            layout.px(1),
            layout.px(4),
            layout.px(5),
        )

    _corner_bracket(frame, layout, 33, 65, 24, 24, MENU_GRID_STRONG)
    _corner_bracket(frame, layout, 1245, 65, -24, 24, MENU_GRID_STRONG)
    _corner_bracket(frame, layout, 33, 633, 24, -24, MENU_CYAN)
    _corner_bracket(frame, layout, 1245, 633, -24, -24, MENU_LIME)
    _draw_target(frame, layout, (40, 245), MENU_CYAN, 12)
    _draw_plus(frame, layout, (422, 76), MENU_CYAN, 9)
    _draw_plus(frame, layout, (656, 76), (90, 111, 116), 8)
    _draw_corner_pixels(frame, layout)


def draw_mode_visual(frame, layout: ui.Layout | None = None) -> None:
    layout = layout or ui.layout_for(frame)
    _menu_panel(frame, layout.rect(79, 243, 516, 267), layout, radius=6)
    _put_text(frame, "LIVE GESTURE SURFACE", layout.point(110, 283), layout.font(0.64), MENU_INK, layout.px(2))
    _put_text(frame, "Clear status, large targets, low-latency feedback.", layout.point(111, 316), layout.font(0.43), MENU_INK, layout.px(1))

    _corner_bracket(frame, layout, 112, 341, 16, 16, (136, 187, 196))
    _corner_bracket(frame, layout, 112, 474, 16, -16, (136, 187, 196))
    _draw_plus(frame, layout, (187, 475), (86, 154, 162), 8)

    trail = np.array(
        [(136, 421), (207, 355), (272, 426), (334, 344), (395, 428), (449, 372)],
        dtype=np.int32,
    )
    trail = np.array([layout.point(x, y) for x, y in trail], dtype=np.int32)
    cv2.polylines(frame, [trail], False, MENU_CYAN, layout.px(9), cv2.LINE_AA)
    for point in trail[1::2]:
        cv2.circle(frame, tuple(point), layout.px(13), MENU_INK, -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), layout.px(9), MENU_LIME, -1, cv2.LINE_AA)

    grid_x, grid_y = 408, 339
    cell = 52
    for row in range(3):
        for column in range(3):
            x = grid_x + column * cell
            y = grid_y + row * cell
            fill = (245, 245, 242) if (row + column) % 2 == 0 else (235, 235, 232)
            cv2.rectangle(frame, layout.point(x, y), layout.point(x + cell - 5, y + cell - 5), fill, -1)
            cv2.rectangle(frame, layout.point(x, y), layout.point(x + cell - 5, y + cell - 5), MENU_INK, layout.px(3), cv2.LINE_AA)
    cv2.rectangle(frame, layout.point(grid_x + cell, grid_y + cell), layout.point(grid_x + cell * 2 - 5, grid_y + cell * 2 - 5), MENU_CYAN, layout.px(2), cv2.LINE_AA)
    cv2.rectangle(frame, layout.point(grid_x + cell + 3, grid_y + cell + 3), layout.point(grid_x + cell * 2 - 8, grid_y + cell * 2 - 8), MENU_LIME, layout.px(1), cv2.LINE_AA)


def draw_system_strip(
    frame,
    layout: ui.Layout | None = None,
    camera_index: int = 0,
    camera_indices: list[int] | None = None,
) -> None:
    layout = layout or ui.layout_for(frame)
    _menu_panel(frame, layout.rect(79, 529, 516, 73), layout, radius=5)
    if camera_indices is None:
        labels = [
            ("WEBCAM", MENU_CYAN, "camera"),
            ("MEDIAPIPE", MENU_LIME, "target"),
            ("REALTIME", MENU_YELLOW, "arrows"),
        ]
    elif camera_indices:
        labels = [
            (f"CAMERA {camera_index}", MENU_CYAN, "camera"),
            (f"{len(camera_indices)} FOUND", MENU_LIME, "target"),
            ("A/D CHANGE", MENU_YELLOW, "arrows"),
        ]
    else:
        labels = [
            ("NO CAMERA", MENU_RED, "camera"),
            ("CHECK USB", MENU_YELLOW, "target"),
            ("R RESCAN", MENU_CYAN, "arrows"),
        ]
    chip_specs = ((90, 158), (261, 146), (422, 160))
    for (label, color, icon), (x, width) in zip(labels, chip_specs):
        _status_chip(frame, layout, layout.rect(x, 541, width, 47), label, color, icon)


def draw_menu_item(
    frame,
    item: MenuItem,
    index: int,
    selected: bool,
    origin: tuple[int, int],
    layout: ui.Layout | None = None,
) -> None:
    layout = layout or ui.layout_for(frame)
    x, y = origin
    width = 504
    height = 80
    color = menu_color(item.action)
    fill = color if selected else MENU_PAPER_ALT
    _menu_panel(frame, layout.rect(x, y, width, height), layout, fill=fill, radius=4, shadow=True)

    badge_x = x + 11
    badge_y = y + 11
    _rounded_rect(frame, layout.rect(badge_x, badge_y, 58, 58), color, -1, layout.px(2))
    _rounded_rect(frame, layout.rect(badge_x, badge_y, 58, 58), MENU_INK, layout.px(3), layout.px(2))
    _put_center(frame, item.shortcut, layout.point(badge_x + 29, badge_y + 31), layout.font(0.88), MENU_INK, layout.px(3))

    _put_text(frame, item.title, layout.point(x + 88, y + 38), layout.font(0.72), MENU_INK, layout.px(2))
    _put_text(frame, item.subtitle, layout.point(x + 89, y + 62), layout.font(0.39), MENU_INK, layout.px(1))
    if selected:
        ready_rect = layout.rect(x + width - 90, y + 13, 75, 33)
        _rounded_rect(frame, ready_rect, MENU_INK, -1, layout.px(3))
        _put_center(frame, "READY", layout.point(x + width - 53, y + 31), layout.font(0.39), MENU_LIME, layout.px(2))


def menu_color(action: MenuAction) -> tuple[int, int, int]:
    if action == MenuAction.DRAWING:
        return MENU_CYAN
    if action == MenuAction.PUZZLE:
        return MENU_LIME
    if action == MenuAction.CAMERA_CHECK:
        return MENU_YELLOW
    return MENU_RED


def _put_text(
    frame,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(frame, text, origin, MENU_FONT, scale, color, thickness, cv2.LINE_AA)


def _put_center(
    frame,
    text: str,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    size, _ = cv2.getTextSize(text, MENU_FONT, scale, thickness)
    origin = (center[0] - size[0] // 2, center[1] + size[1] // 2)
    _put_text(frame, text, origin, scale, color, thickness)


def _rounded_rect(
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


def _menu_panel(
    frame,
    rect: tuple[int, int, int, int],
    layout: ui.Layout,
    fill: tuple[int, int, int] = MENU_PAPER_ALT,
    radius: int = 5,
    shadow: bool = True,
) -> None:
    x, y, width, height = rect
    scaled_radius = layout.px(radius)
    if shadow:
        offset = layout.px(7)
        _rounded_rect(frame, (x + offset, y + offset, width, height), MENU_INK, -1, scaled_radius)
    _rounded_rect(frame, rect, fill, -1, scaled_radius)
    _rounded_rect(frame, rect, MENU_INK, layout.px(3), scaled_radius)


def _outlined_label(
    frame,
    text: str,
    rect: tuple[int, int, int, int],
    layout: ui.Layout,
) -> None:
    x, y, width, height = rect
    cv2.rectangle(frame, (x + layout.px(4), y + layout.px(4)), (x + width + layout.px(4), y + height + layout.px(4)), MENU_GRID_STRONG, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), MENU_PAPER_ALT, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), MENU_CYAN, layout.px(2), cv2.LINE_AA)
    _put_center(frame, text, (x + width // 2, y + height // 2 + layout.px(1)), layout.font(0.42), MENU_INK, layout.px(1))


def _dashed_line(
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


def _draw_target(
    frame,
    layout: ui.Layout,
    center: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    cx, cy = layout.point(*center)
    r = layout.px(radius)
    cv2.circle(frame, (cx, cy), r, color, layout.px(2), cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), layout.px(3), color, -1, cv2.LINE_AA)
    cv2.line(frame, (cx - r - layout.px(8), cy), (cx + r + layout.px(8), cy), color, layout.px(2), cv2.LINE_AA)
    cv2.line(frame, (cx, cy - r - layout.px(8)), (cx, cy + r + layout.px(8)), color, layout.px(2), cv2.LINE_AA)


def _draw_plus(
    frame,
    layout: ui.Layout,
    center: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    cx, cy = layout.point(*center)
    r = layout.px(radius)
    cv2.line(frame, (cx - r, cy), (cx + r, cy), color, layout.px(1), cv2.LINE_AA)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), color, layout.px(1), cv2.LINE_AA)


def _corner_bracket(
    frame,
    layout: ui.Layout,
    x: int,
    y: int,
    dx: int,
    dy: int,
    color: tuple[int, int, int],
) -> None:
    origin = layout.point(x, y)
    horizontal = layout.point(x + dx, y)
    vertical = layout.point(x, y + dy)
    cv2.line(frame, origin, horizontal, color, layout.px(1), cv2.LINE_AA)
    cv2.line(frame, origin, vertical, color, layout.px(1), cv2.LINE_AA)


def _draw_corner_pixels(frame, layout: ui.Layout) -> None:
    size = 10
    for x, y, color in ((23, 662, MENU_CYAN), (35, 672, MENU_CYAN), (1224, 672, MENU_LIME), (1236, 662, MENU_LIME)):
        cv2.rectangle(frame, layout.point(x, y), layout.point(x + size, y + size), color, -1)


def _status_chip(
    frame,
    layout: ui.Layout,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    icon: str,
) -> None:
    x, y, width, height = rect
    offset = layout.px(5)
    _rounded_rect(frame, (x + offset, y + offset, width, height), MENU_INK, -1, layout.px(2))
    _rounded_rect(frame, rect, color, -1, layout.px(2))
    _rounded_rect(frame, rect, MENU_INK, layout.px(3), layout.px(2))
    icon_center = (x + layout.px(30), y + height // 2)
    if icon == "camera":
        cv2.rectangle(frame, (icon_center[0] - layout.px(12), icon_center[1] - layout.px(8)), (icon_center[0] + layout.px(12), icon_center[1] + layout.px(9)), MENU_INK, -1)
        cv2.rectangle(frame, (icon_center[0] - layout.px(7), icon_center[1] - layout.px(12)), (icon_center[0] + layout.px(3), icon_center[1] - layout.px(7)), MENU_INK, -1)
        cv2.circle(frame, icon_center, layout.px(5), color, layout.px(2), cv2.LINE_AA)
    elif icon == "target":
        cv2.circle(frame, icon_center, layout.px(10), MENU_INK, layout.px(2), cv2.LINE_AA)
        cv2.line(frame, (icon_center[0] - layout.px(15), icon_center[1]), (icon_center[0] + layout.px(15), icon_center[1]), MENU_INK, layout.px(2), cv2.LINE_AA)
        cv2.line(frame, (icon_center[0], icon_center[1] - layout.px(15)), (icon_center[0], icon_center[1] + layout.px(15)), MENU_INK, layout.px(2), cv2.LINE_AA)
    else:
        cv2.line(frame, (icon_center[0] - layout.px(13), icon_center[1]), (icon_center[0] + layout.px(13), icon_center[1]), MENU_INK, layout.px(4), cv2.LINE_AA)
        for direction in (-1, 1):
            tip_x = icon_center[0] + direction * layout.px(14)
            base_x = icon_center[0] + direction * layout.px(7)
            cv2.line(frame, (tip_x, icon_center[1]), (base_x, icon_center[1] - layout.px(6)), MENU_INK, layout.px(3), cv2.LINE_AA)
            cv2.line(frame, (tip_x, icon_center[1]), (base_x, icon_center[1] + layout.px(6)), MENU_INK, layout.px(3), cv2.LINE_AA)
    _put_text(frame, label, (x + layout.px(54), y + height // 2 + layout.px(7)), layout.font(0.42), MENU_INK, layout.px(2))


def _draw_shortcut_strip(frame, layout: ui.Layout) -> None:
    rect = layout.rect(84, 638, 1101, 45)
    _menu_panel(frame, rect, layout, radius=4, shadow=True)
    x, y, width, height = rect

    # Mouse outline.
    cv2.ellipse(frame, (x + layout.px(38), y + height // 2), (layout.px(8), layout.px(14)), 0, 0, 360, MENU_INK, layout.px(2), cv2.LINE_AA)
    cv2.line(frame, (x + layout.px(38), y + layout.px(8)), (x + layout.px(38), y + layout.px(19)), MENU_CYAN, layout.px(2), cv2.LINE_AA)

    segments = [
        ("Mouse/Enter Open", 62, 223),
        ("1  Drawing", 223, 342),
        ("2  Puzzle", 342, 446),
        ("K  Check", 446, 548),
        ("A/D  Camera", 548, 680),
        ("R  Rescan", 680, 790),
        ("F11  Fullscreen", 790, 938),
        ("Q/Esc  Exit", 938, 1082),
    ]
    for index, (label, start, end) in enumerate(segments):
        if index > 0:
            separator_x = x + layout.px(start)
            cv2.line(frame, (separator_x, y + layout.px(10)), (separator_x, y + height - layout.px(10)), MENU_CYAN, layout.px(2), cv2.LINE_AA)
        _put_center(frame, label, (x + layout.px((start + end) / 2), y + height // 2 + layout.px(1)), layout.font(0.35), MENU_INK, layout.px(1))


if __name__ == "__main__":
    raise SystemExit(main())

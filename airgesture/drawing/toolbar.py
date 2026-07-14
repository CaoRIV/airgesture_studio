from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2

from airgesture.ui import theme as ui


class ToolbarAction(Enum):
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    YELLOW = "Yellow"
    WHITE = "White"
    ERASER = "Eraser"
    THIN = "Thin"
    THICK = "Thick"
    UNDO = "Undo"
    CLEAR = "Clear"
    SAVE = "Save"
    OPEN_FOLDER = "Open Folder"


@dataclass(frozen=True)
class ToolbarButton:
    action: ToolbarAction
    label: str
    rect: tuple[int, int, int, int]
    color: tuple[int, int, int]


class GestureToolbar:
    """Toolbar that can be selected with the smoothed fingertip cursor."""

    def __init__(self, select_cooldown_seconds: float = 0.65) -> None:
        self.select_cooldown_seconds = select_cooldown_seconds
        self._last_selected_at = 0.0
        self._selected_while_hovered: ToolbarAction | None = None

    def buttons(self, display_width: int) -> list[ToolbarButton]:
        specs = [
            (ToolbarAction.RED, "Red", (0, 0, 255)),
            (ToolbarAction.GREEN, "Green", (0, 230, 70)),
            (ToolbarAction.BLUE, "Blue", (255, 80, 0)),
            (ToolbarAction.YELLOW, "Yellow", (0, 235, 255)),
            (ToolbarAction.WHITE, "White", (255, 255, 255)),
            (ToolbarAction.ERASER, "Erase", (210, 220, 230)),
            (ToolbarAction.THIN, "Thin", (160, 180, 210)),
            (ToolbarAction.THICK, "Thick", (160, 180, 210)),
            (ToolbarAction.UNDO, "Undo", (255, 190, 90)),
            (ToolbarAction.CLEAR, "Clear", (120, 150, 255)),
            (ToolbarAction.SAVE, "Save", (120, 220, 180)),
            (ToolbarAction.OPEN_FOLDER, "Folder", (255, 190, 90)),
        ]

        button_width = 96
        button_height = 50
        gap = 8
        total_width = len(specs) * button_width + (len(specs) - 1) * gap
        start_x = max((display_width - total_width) // 2, 16)
        y = 94

        buttons = []
        for index, (action, label, color) in enumerate(specs):
            x = start_x + index * (button_width + gap)
            buttons.append(ToolbarButton(action, label, (x, y, button_width, button_height), color))
        return buttons

    def hit_test(self, point: tuple[int, int] | None, display_width: int) -> ToolbarAction | None:
        if point is None:
            return None

        point_x, point_y = point
        for button in self.buttons(display_width):
            x, y, width, height = button.rect
            if x <= point_x <= x + width and y <= point_y <= y + height:
                return button.action
        return None

    def select(
        self,
        action: ToolbarAction | None,
        now_seconds: float,
    ) -> ToolbarAction | None:
        if action is None:
            self._selected_while_hovered = None
            return None

        if action == self._selected_while_hovered:
            return None

        if now_seconds - self._last_selected_at < self.select_cooldown_seconds:
            return None

        self._last_selected_at = now_seconds
        self._selected_while_hovered = action
        return action


def draw_toolbar(
    display_frame,
    toolbar: GestureToolbar,
    active_action: ToolbarAction,
    hovered_action: ToolbarAction | None,
) -> None:
    buttons = toolbar.buttons(display_frame.shape[1])
    if not buttons:
        return

    first_x = buttons[0].rect[0]
    first_y = buttons[0].rect[1]
    last = buttons[-1].rect
    bar_width = last[0] + last[2] - first_x
    ui.panel(
        display_frame,
        (first_x - 12, first_y - 12, bar_width + 24, 74),
        fill=(12, 16, 24),
        border=ui.BORDER_SOFT,
        alpha=0.76,
        shadow=True,
    )

    for button in buttons:
        x, y, width, height = button.rect
        is_active = button.action == active_action
        is_hovered = button.action == hovered_action
        fill = ui.SURFACE_RAISED if is_active else ((34, 42, 54) if is_hovered else ui.SURFACE)
        border = button.color if is_active or is_hovered else (88, 96, 110)
        border_thickness = 3 if is_active or is_hovered else 1

        ui.panel(
            display_frame,
            (x, y, width, height),
            fill=fill,
            border=border,
            alpha=0.94,
            thickness=border_thickness,
            shadow=False,
        )
        if is_active:
            ui.accent_bar(display_frame, (x, y, width, 4), button.color)

        draw_tool_icon(
            display_frame,
            button.action,
            (x + 15, y + 14, 22, 22),
            button.color,
            is_active or is_hovered,
        )
        ui.put_text(
            display_frame,
            button.label,
            (x + 42, y + 31),
            0.47,
            ui.TEXT,
            1,
        )


def draw_tool_icon(
    frame,
    action: ToolbarAction,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    active: bool,
) -> None:
    x, y, width, height = rect
    center = (x + width // 2, y + height // 2)
    line_color = color if active else ui.TEXT_MUTED

    if action in {
        ToolbarAction.RED,
        ToolbarAction.GREEN,
        ToolbarAction.BLUE,
        ToolbarAction.YELLOW,
        ToolbarAction.WHITE,
    }:
        cv2.circle(frame, center, 9, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 11, ui.WHITE, 1, cv2.LINE_AA)
    elif action == ToolbarAction.ERASER:
        cv2.rectangle(frame, (x + 4, y + 7), (x + width - 3, y + height - 5), line_color, 2)
        cv2.line(frame, (x + 5, y + height - 4), (x + width, y + height - 4), ui.TEXT_DIM, 1)
    elif action == ToolbarAction.THIN:
        cv2.line(frame, (x + 3, center[1]), (x + width - 3, center[1]), line_color, 2, cv2.LINE_AA)
    elif action == ToolbarAction.THICK:
        cv2.line(frame, (x + 3, center[1]), (x + width - 3, center[1]), line_color, 6, cv2.LINE_AA)
    elif action == ToolbarAction.UNDO:
        cv2.ellipse(frame, center, (8, 8), 0, 210, 520, line_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x + 3, y + 7), (x + 9, y + 4), line_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x + 3, y + 7), (x + 7, y + 13), line_color, 2, cv2.LINE_AA)
    elif action == ToolbarAction.CLEAR:
        cv2.line(frame, (x + 4, y + 4), (x + width - 4, y + height - 4), line_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x + width - 4, y + 4), (x + 4, y + height - 4), line_color, 2, cv2.LINE_AA)
    elif action == ToolbarAction.SAVE:
        cv2.line(frame, (center[0], y + 3), (center[0], y + height - 8), line_color, 2, cv2.LINE_AA)
        cv2.line(frame, (center[0], y + height - 8), (x + 7, y + height - 15), line_color, 2, cv2.LINE_AA)
        cv2.line(frame, (center[0], y + height - 8), (x + width - 7, y + height - 15), line_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x + 4, y + height - 3), (x + width - 4, y + height - 3), line_color, 2, cv2.LINE_AA)
    elif action == ToolbarAction.OPEN_FOLDER:
        cv2.rectangle(frame, (x + 3, y + 7), (x + width - 3, y + height - 4), line_color, 2)
        cv2.line(frame, (x + 4, y + 7), (x + 10, y + 2), line_color, 2, cv2.LINE_AA)
        cv2.line(frame, (x + 10, y + 2), (x + width - 7, y + 2), line_color, 2, cv2.LINE_AA)

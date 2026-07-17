from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2

from airgesture.drawing import style
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

    def buttons(
        self,
        display_width: int,
        display_height: int = 720,
    ) -> list[ToolbarButton]:
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

        if display_width >= 900:
            layout = ui.Layout(display_width, display_height)
            widths = (88, 92, 92, 96, 94, 100, 86, 90, 92, 88, 88, 92)
            gap = 10
            total_width = sum(widths) + gap * (len(widths) - 1)
            x = (1280 - total_width) // 2
            buttons = []
            for (action, label, color), width in zip(specs, widths):
                buttons.append(
                    ToolbarButton(
                        action,
                        label,
                        layout.rect(x, 23, width, 47),
                        color,
                    )
                )
                x += width + gap
            return buttons
        if display_width >= 640:
            columns = 4
        else:
            columns = 3

        margin = max(10, min(24, display_width // 60))
        gap = max(5, min(10, display_width // 140))
        available_width = max(1, display_width - margin * 2)
        density_scale = max(0.82, min(1.35, display_height / 720.0))
        button_width = min(
            int(round(112 * density_scale)),
            max(72, (available_width - gap * (columns - 1)) // columns),
        )
        button_height = int(round(50 * density_scale))
        row_gap = max(6, int(round(8 * density_scale)))
        top = max(78, int(round(104 * density_scale)))

        buttons = []
        for index, (action, label, color) in enumerate(specs):
            row, column = divmod(index, columns)
            items_in_row = min(columns, len(specs) - row * columns)
            row_width = items_in_row * button_width + (items_in_row - 1) * gap
            start_x = max((display_width - row_width) // 2, margin)
            x = start_x + column * (button_width + gap)
            y = top + row * (button_height + row_gap)
            buttons.append(ToolbarButton(action, label, (x, y, button_width, button_height), color))
        return buttons

    def hit_test(
        self,
        point: tuple[int, int] | None,
        display_width: int,
        display_height: int = 720,
    ) -> ToolbarAction | None:
        if point is None:
            return None

        point_x, point_y = point
        for button in self.buttons(display_width, display_height):
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
    buttons = toolbar.buttons(display_frame.shape[1], display_frame.shape[0])
    if not buttons:
        return

    first_x = min(button.rect[0] for button in buttons)
    first_y = min(button.rect[1] for button in buttons)
    last_x = max(button.rect[0] + button.rect[2] for button in buttons)
    last_y = max(button.rect[1] + button.rect[3] for button in buttons)
    bar_width = last_x - first_x
    bar_height = last_y - first_y
    layout = ui.layout_for(display_frame)
    if display_frame.shape[1] >= 900:
        panel_rect = layout.rect(6, 11, 1268, 70)
    else:
        panel_rect = (first_x - 10, first_y - 10, bar_width + 20, bar_height + 20)
    style.panel(display_frame, panel_rect, layout, radius=5)

    if display_frame.shape[1] >= 900:
        by_action = {button.action: button for button in buttons}
        for action in (ToolbarAction.ERASER, ToolbarAction.THICK):
            button = by_action[action]
            separator_x = button.rect[0] + button.rect[2] + layout.px(4)
            style.dashed_line(
                display_frame,
                (separator_x, panel_rect[1] + layout.px(10)),
                (separator_x, panel_rect[1] + panel_rect[3] - layout.px(10)),
                style.INK,
                layout.px(2),
                layout.px(5),
                layout.px(4),
            )

    for button in buttons:
        x, y, width, height = button.rect
        density = max(0.72, min(1.45, height / 47.0))
        is_active = button.action == active_action
        is_hovered = button.action == hovered_action
        fill = button.color if is_active else style.PAPER_ALT
        style.panel(
            display_frame,
            (x, y, width, height),
            layout,
            fill=fill,
            radius=3,
            shadow=True,
            thickness=3 if is_active or is_hovered else 2,
        )

        draw_tool_icon(
            display_frame,
            button.action,
            (
                x + int(round(15 * density)),
                y + int(round(14 * density)),
                int(round(22 * density)),
                int(round(22 * density)),
            ),
            button.color,
            is_active or is_hovered,
        )
        label_scale = (0.38 if width >= layout.px(84) else 0.34) * density
        style.put_text(
            display_frame,
            button.label,
            (x + int(round(37 * density)), y + int(round(30 * density))),
            label_scale,
            style.INK,
            max(1, int(round(density))),
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
    line_color = style.INK

    if action in {
        ToolbarAction.RED,
        ToolbarAction.GREEN,
        ToolbarAction.BLUE,
        ToolbarAction.YELLOW,
        ToolbarAction.WHITE,
    }:
        cv2.circle(frame, center, 9, color, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 10, style.INK, 2, cv2.LINE_AA)
    elif action == ToolbarAction.ERASER:
        cv2.rectangle(frame, (x + 4, y + 7), (x + width - 3, y + height - 5), line_color, 2)
        cv2.line(frame, (x + 5, y + height - 4), (x + width, y + height - 4), style.INK, 1)
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

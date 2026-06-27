from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2


class ToolbarAction(Enum):
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    YELLOW = "Yellow"
    WHITE = "White"
    ERASER = "Eraser"
    THIN = "Thin"
    THICK = "Thick"
    CLEAR = "Clear"
    SAVE = "Save"


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
            (ToolbarAction.CLEAR, "Clear", (120, 150, 255)),
            (ToolbarAction.SAVE, "Save", (120, 220, 180)),
        ]

        button_width = 104
        button_height = 46
        gap = 10
        total_width = len(specs) * button_width + (len(specs) - 1) * gap
        start_x = max((display_width - total_width) // 2, 16)
        y = 82

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

    for button in buttons:
        x, y, width, height = button.rect
        is_active = button.action == active_action
        is_hovered = button.action == hovered_action
        fill = (42, 46, 54)
        border = button.color if is_active or is_hovered else (88, 96, 110)
        border_thickness = 3 if is_active or is_hovered else 1

        cv2.rectangle(display_frame, (x, y), (x + width, y + height), fill, -1)
        cv2.rectangle(
            display_frame,
            (x, y),
            (x + width, y + height),
            border,
            border_thickness,
            cv2.LINE_AA,
        )

        swatch_center = (x + 18, y + height // 2)
        cv2.circle(display_frame, swatch_center, 9, button.color, -1, cv2.LINE_AA)
        cv2.putText(
            display_frame,
            button.label,
            (x + 34, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (238, 241, 245),
            1,
            cv2.LINE_AA,
        )

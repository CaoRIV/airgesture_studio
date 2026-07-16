from __future__ import annotations

from datetime import datetime
from math import hypot
import time

import cv2

from airgesture.config import require_valid_settings
from airgesture.core.camera import Camera
from airgesture.core.hand_tracker import HandTracker
from airgesture.core.smoothing import AdaptivePointSmoother
from airgesture.drawing.canvas import (
    CanvasConfig,
    DrawingCanvas,
    open_output_directory,
)
from airgesture.drawing.display import (
    DisplayConfig,
    draw_app_overlay,
    fit_frame_to_display,
    frame_point_to_display,
    frame_point_to_workspace,
)
from airgesture.drawing.gesture_controller import GestureController, GestureMode
from airgesture.drawing.letter_recognizer import (
    LetterRecognizer,
    RecognitionAnalysis,
)
from airgesture.drawing.stroke_state import StrokeEndDebouncer
from airgesture.drawing.toolbar import GestureToolbar, ToolbarAction, draw_toolbar
from airgesture.errors import DrawingSaveError, OutputDirectoryError
from airgesture.paths import DRAWINGS_DIR
from airgesture.puzzle.gesture import PinchGesture
from airgesture.ui.runtime_errors import (
    log_user_error,
    run_with_error_dialog,
)
from airgesture.ui.window import ResponsiveWindow


WINDOW_NAME = "Hand Gesture Air Drawing - Gesture Toolbar"
OUTPUT_DIR = DRAWINGS_DIR
THUMB_TIP = 4
NOTIFICATION_SECONDS = 4.0
TOOL_COLORS = {
    ToolbarAction.RED: (0, 0, 255),
    ToolbarAction.GREEN: (0, 230, 70),
    ToolbarAction.BLUE: (255, 80, 0),
    ToolbarAction.YELLOW: (0, 235, 255),
    ToolbarAction.WHITE: (255, 255, 255),
}


def should_quit(key_code: int) -> bool:
    return key_code in (27, ord("q"), ord("Q"))


def should_clear(key_code: int) -> bool:
    return key_code in (ord("c"), ord("C"))


def should_undo(key_code: int) -> bool:
    return key_code in (26, ord("u"), ord("U"), ord("z"), ord("Z"))


def should_open_output(key_code: int) -> bool:
    return key_code in (ord("o"), ord("O"))


def save_filename() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"drawing_{timestamp}.png"


def save_drawing_with_feedback(
    drawing_canvas: DrawingCanvas,
) -> tuple[str, bool]:
    try:
        saved_path = drawing_canvas.save(OUTPUT_DIR, save_filename())
        return f"SAVED: {saved_path.name}", False
    except DrawingSaveError as exc:
        log_user_error("Drawing save failed", exc)
        return f"SAVE FAILED: {exc}", True


def open_drawings_with_feedback() -> tuple[str, bool]:
    try:
        open_output_directory(OUTPUT_DIR)
        return "DRAWINGS FOLDER OPENED", False
    except OutputDirectoryError as exc:
        log_user_error("Opening drawings folder failed", exc)
        return f"FOLDER ERROR: {exc}", True


def point_distance(start: tuple[int, int], end: tuple[int, int]) -> float:
    return hypot(end[0] - start[0], end[1] - start[1])


def finalize_stroke(
    drawing_canvas: DrawingCanvas,
    recognizer: LetterRecognizer,
    stroke_points: list[tuple[int, int]],
) -> RecognitionAnalysis | None:
    if not stroke_points:
        return None

    analysis = recognizer.analyze(stroke_points)
    if analysis is not None and analysis.accepted is not None:
        drawing_canvas.clear_stroke()
        drawing_canvas.draw_clean_letter(
            analysis.accepted.letter,
            analysis.accepted.bounds,
        )
        return analysis

    cleaned_points = (
        analysis.cleaned_points
        if analysis is not None
        else recognizer.clean_points(stroke_points)
    )
    drawing_canvas.commit_clean_stroke(cleaned_points)
    return analysis


def main() -> int:
    return run_with_error_dialog(WINDOW_NAME, _run)


def _run() -> int:
    settings = require_valid_settings()
    air_settings = settings.air_drawing
    drawing_settings = air_settings.drawing
    camera = Camera(settings.camera)
    drawing_canvas = DrawingCanvas(
        CanvasConfig(
            brush_thickness=drawing_settings.default_brush_size,
            eraser_thickness=drawing_settings.eraser_size,
            max_history_steps=drawing_settings.max_undo_steps,
        )
    )
    gesture_controller = GestureController()
    letter_recognizer = LetterRecognizer(config=air_settings.recognition)
    toolbar = GestureToolbar()
    pinch_detector = PinchGesture(config=air_settings.pinch)
    point_smoother = AdaptivePointSmoother(air_settings.adaptive_smoothing)
    stroke_debouncer = StrokeEndDebouncer(
        delay_seconds=drawing_settings.stroke_end_debounce_seconds,
        grace_frames=drawing_settings.draw_grace_frames,
    )
    current_color_action = ToolbarAction.RED
    active_toolbar_action = ToolbarAction.RED
    erasing = False
    previous_draw_point: tuple[int, int] | None = None
    erase_history_started = False
    stroke_points: list[tuple[int, int]] = []
    last_detected_symbol: str | None = None
    last_recognition_suggestions: tuple[tuple[str, float], ...] = ()
    last_detected_until = 0.0
    last_notification_message: str | None = None
    last_notification_is_error = False
    last_notification_until = 0.0
    drawing_canvas.set_brush_color(TOOL_COLORS[current_color_action])

    camera.open_or_raise()
    window = ResponsiveWindow(WINDOW_NAME, start_maximized=True)

    try:
        window.create()
        camera.apply_window_title(WINDOW_NAME)
        with HandTracker(air_settings.tracker) as hand_tracker:
            previous_time = time.perf_counter()
            smoothed_fps = 0.0

            while True:
                frame_time = time.perf_counter()
                frame = camera.read_or_raise()

                drawing_canvas.ensure_size(frame.shape)
                results = hand_tracker.detect(frame)
                gesture_state = gesture_controller.analyze(results, frame.shape)
                index_tip = point_smoother.update(
                    gesture_state.index_tip,
                    timestamp_seconds=frame_time,
                )
                thumb_tip = hand_tracker.get_landmark_pixel(results, frame.shape, THUMB_TIP)
                pinch = pinch_detector.update(thumb_tip, index_tip)
                raw_drawing_active = pinch.active

                if raw_drawing_active and index_tip is not None:
                    stroke_debouncer.mark_active()
                    if previous_draw_point is not None:
                        bridge_distance = point_distance(previous_draw_point, index_tip)
                        if bridge_distance <= drawing_settings.max_bridge_distance:
                            if erasing:
                                if not erase_history_started:
                                    erase_history_started = drawing_canvas.begin_history_action()
                                drawing_canvas.erase_line(previous_draw_point, index_tip)
                            else:
                                drawing_canvas.draw_line(previous_draw_point, index_tip)
                                stroke_points.append(index_tip)
                        elif not erasing:
                            analysis = finalize_stroke(
                                drawing_canvas,
                                letter_recognizer,
                                stroke_points,
                            )
                            if analysis is not None:
                                last_detected_symbol = (
                                    analysis.accepted.letter
                                    if analysis.accepted is not None
                                    else None
                                )
                                last_recognition_suggestions = tuple(
                                    (candidate.symbol, candidate.confidence)
                                    for candidate in analysis.suggestions
                                )
                                last_detected_until = (
                                    time.perf_counter()
                                    + drawing_settings.detection_display_seconds
                                )
                            else:
                                last_detected_symbol = None
                                last_recognition_suggestions = ()
                                last_detected_until = 0.0
                            drawing_canvas.clear_stroke()
                            stroke_points = [index_tip]
                        elif erasing:
                            erase_history_started = False
                    elif not erasing:
                        drawing_canvas.clear_stroke()
                        stroke_points = [index_tip]
                    previous_draw_point = index_tip
                elif previous_draw_point is not None and stroke_debouncer.should_finalize(
                    frame_time,
                    has_open_stroke=True,
                ):
                    if not erasing:
                        analysis = finalize_stroke(
                            drawing_canvas,
                            letter_recognizer,
                            stroke_points,
                        )
                        if analysis is not None:
                            last_detected_symbol = (
                                analysis.accepted.letter
                                if analysis.accepted is not None
                                else None
                            )
                            last_recognition_suggestions = tuple(
                                (candidate.symbol, candidate.confidence)
                                for candidate in analysis.suggestions
                            )
                            last_detected_until = (
                                time.perf_counter()
                                + drawing_settings.detection_display_seconds
                            )
                        else:
                            last_detected_symbol = None
                            last_recognition_suggestions = ()
                            last_detected_until = 0.0
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                elif previous_draw_point is None:
                    stroke_debouncer.reset()

                frame = drawing_canvas.compose(frame)
                hand_tracker.draw_landmarks(frame, results)
                current_time = time.perf_counter()
                instant_fps = 1.0 / max(current_time - previous_time, 0.0001)
                previous_time = current_time
                smoothed_fps = (
                    instant_fps if smoothed_fps == 0.0 else smoothed_fps * 0.9 + instant_fps * 0.1
                )

                viewport = window.viewport()
                display_frame, frame_bounds = fit_frame_to_display(
                    frame,
                    DisplayConfig(width=viewport.width, height=viewport.height),
                )
                camera_cursor_point = frame_point_to_display(
                    index_tip,
                    frame.shape,
                    frame_bounds,
                )
                toolbar_cursor_point = frame_point_to_workspace(
                    index_tip,
                    frame.shape,
                    display_frame.shape,
                )
                hovered_action = toolbar.hit_test(
                    toolbar_cursor_point
                    if gesture_state.mode == GestureMode.MOVE and not pinch.active
                    else None,
                    display_frame.shape[1],
                    display_frame.shape[0],
                )
                selected_action = toolbar.select(hovered_action, current_time)

                if selected_action in TOOL_COLORS:
                    current_color_action = selected_action
                    active_toolbar_action = selected_action
                    erasing = False
                    drawing_canvas.clear_stroke()
                    drawing_canvas.set_brush_color(TOOL_COLORS[selected_action])
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                elif selected_action == ToolbarAction.ERASER:
                    active_toolbar_action = ToolbarAction.ERASER
                    erasing = True
                    drawing_canvas.clear_stroke()
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                elif selected_action == ToolbarAction.THIN:
                    drawing_canvas.clear_stroke()
                    drawing_canvas.set_brush_thickness(drawing_settings.thin_brush_size)
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                elif selected_action == ToolbarAction.THICK:
                    drawing_canvas.clear_stroke()
                    drawing_canvas.set_brush_thickness(drawing_settings.thick_brush_size)
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                elif selected_action == ToolbarAction.UNDO:
                    drawing_canvas.clear_stroke()
                    drawing_canvas.undo()
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                    last_detected_symbol = None
                    last_recognition_suggestions = ()
                    last_detected_until = 0.0
                elif selected_action == ToolbarAction.CLEAR:
                    drawing_canvas.clear()
                    point_smoother.reset()
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                    last_detected_symbol = None
                    last_recognition_suggestions = ()
                    last_detected_until = 0.0
                elif selected_action == ToolbarAction.SAVE:
                    drawing_canvas.clear_stroke()
                    (
                        last_notification_message,
                        last_notification_is_error,
                    ) = save_drawing_with_feedback(drawing_canvas)
                    last_notification_until = current_time + NOTIFICATION_SECONDS
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                elif selected_action == ToolbarAction.OPEN_FOLDER:
                    (
                        last_notification_message,
                        last_notification_is_error,
                    ) = open_drawings_with_feedback()
                    last_notification_until = current_time + NOTIFICATION_SECONDS

                hand_detected = bool(results.hand_landmarks)
                detected_symbol = (
                    last_detected_symbol
                    if last_detected_symbol is not None and current_time <= last_detected_until
                    else None
                )
                recognition_suggestions = (
                    last_recognition_suggestions
                    if current_time <= last_detected_until
                    else ()
                )
                notification_message = (
                    last_notification_message
                    if current_time <= last_notification_until
                    else None
                )
                draw_app_overlay(
                    display_frame,
                    frame_bounds,
                    hand_detected=hand_detected,
                    mode="Draw" if pinch.active else gesture_state.mode.value,
                    fps=smoothed_fps,
                    detected_symbol=detected_symbol,
                    recognition_suggestions=recognition_suggestions,
                    notification_message=notification_message,
                    notification_is_error=last_notification_is_error,
                )
                draw_toolbar(
                    display_frame,
                    toolbar,
                    active_toolbar_action,
                    hovered_action,
                )
                if index_tip is not None:
                    visible_cursor_point = (
                        toolbar_cursor_point
                        if gesture_state.mode == GestureMode.MOVE and not pinch.active
                        else camera_cursor_point
                    )
                    if visible_cursor_point is not None:
                        drawing_canvas.draw_cursor(
                            display_frame,
                            visible_cursor_point,
                            erasing=erasing,
                        )

                window.present(display_frame)
                key_code = cv2.waitKeyEx(1)
                if window.handle_window_key(key_code):
                    continue
                if should_quit(key_code):
                    break
                if should_clear(key_code):
                    drawing_canvas.clear()
                    point_smoother.reset()
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                    last_detected_symbol = None
                    last_recognition_suggestions = ()
                    last_detected_until = 0.0
                elif should_undo(key_code):
                    drawing_canvas.clear_stroke()
                    drawing_canvas.undo()
                    stroke_points = []
                    previous_draw_point = None
                    erase_history_started = False
                    stroke_debouncer.reset()
                    last_detected_symbol = None
                    last_recognition_suggestions = ()
                    last_detected_until = 0.0
                elif should_open_output(key_code):
                    (
                        last_notification_message,
                        last_notification_is_error,
                    ) = open_drawings_with_feedback()
                    last_notification_until = (
                        time.perf_counter() + NOTIFICATION_SECONDS
                    )
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

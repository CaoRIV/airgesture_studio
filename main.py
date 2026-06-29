from __future__ import annotations

from datetime import datetime
from math import hypot
from pathlib import Path
import time

import cv2

from camera import Camera, CameraConfig
from canvas import DrawingCanvas
from display import DisplayConfig, draw_app_overlay, fit_frame_to_display, frame_point_to_display
from game_gesture import PinchGesture
from gesture_controller import GestureController, GestureMode
from hand_tracker import HandTracker, HandTrackerConfig
from letter_recognizer import LetterRecognizer
from smoothing import PointSmoother, SmoothingConfig
from toolbar import GestureToolbar, ToolbarAction, draw_toolbar


WINDOW_NAME = "Hand Gesture Air Drawing - Gesture Toolbar"
OUTPUT_DIR = Path("outputs") / "saved_drawings"
DRAW_GRACE_FRAMES = 4
MAX_BRIDGE_DISTANCE = 180
THUMB_TIP = 4
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


def save_filename() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"drawing_{timestamp}.png"


def point_distance(start: tuple[int, int], end: tuple[int, int]) -> float:
    return hypot(end[0] - start[0], end[1] - start[1])


def finalize_stroke(
    drawing_canvas: DrawingCanvas,
    recognizer: LetterRecognizer,
    stroke_points: list[tuple[int, int]],
) -> None:
    if not stroke_points:
        return

    recognized = recognizer.recognize(stroke_points)
    if recognized is not None:
        drawing_canvas.clear_stroke()
        drawing_canvas.draw_clean_letter(recognized.letter, recognized.bounds)
        return

    cleaned_points = recognizer.clean_points(stroke_points)
    drawing_canvas.commit_clean_stroke(cleaned_points)


def main() -> int:
    display_config = DisplayConfig(width=1280, height=720)
    camera_config = CameraConfig(
        camera_index=0,
        mirror=True,
        width=display_config.width,
        height=display_config.height,
        fps=30,
    )
    hand_tracker_config = HandTrackerConfig(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    camera = Camera(camera_config)
    drawing_canvas = DrawingCanvas()
    gesture_controller = GestureController()
    letter_recognizer = LetterRecognizer()
    toolbar = GestureToolbar()
    pinch_detector = PinchGesture(pinch_threshold=46.0, release_threshold=68.0)
    point_smoother = PointSmoother(SmoothingConfig(alpha=0.35))
    current_color_action = ToolbarAction.RED
    active_toolbar_action = ToolbarAction.RED
    erasing = False
    previous_draw_point: tuple[int, int] | None = None
    missing_draw_frames = 0
    stroke_points: list[tuple[int, int]] = []
    drawing_canvas.set_brush_color(TOOL_COLORS[current_color_action])

    if not camera.open():
        print(
            "Error: Could not open webcam at index 0. "
            "Check that a webcam is connected and not being used by another app."
        )
        return 1

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        with HandTracker(hand_tracker_config) as hand_tracker:
            previous_time = time.perf_counter()
            smoothed_fps = 0.0

            while True:
                success, frame = camera.read()
                if not success:
                    print("Error: Could not read frame from webcam.")
                    return 1

                drawing_canvas.ensure_size(frame.shape)
                results = hand_tracker.detect(frame)
                gesture_state = gesture_controller.analyze(results, frame.shape)
                index_tip = point_smoother.update(gesture_state.index_tip)
                thumb_tip = hand_tracker.get_landmark_pixel(results, frame.shape, THUMB_TIP)
                pinch = pinch_detector.update(thumb_tip, index_tip)
                raw_drawing_active = pinch.active
                keep_stroke_open = (
                    gesture_state.mode == GestureMode.IDLE
                    and previous_draw_point is not None
                    and missing_draw_frames < DRAW_GRACE_FRAMES
                )

                if raw_drawing_active and index_tip is not None:
                    missing_draw_frames = 0
                    if previous_draw_point is not None:
                        bridge_distance = point_distance(previous_draw_point, index_tip)
                        if bridge_distance <= MAX_BRIDGE_DISTANCE:
                            if erasing:
                                drawing_canvas.erase_line(previous_draw_point, index_tip)
                            else:
                                drawing_canvas.draw_line(previous_draw_point, index_tip)
                                stroke_points.append(index_tip)
                        elif not erasing:
                            finalize_stroke(drawing_canvas, letter_recognizer, stroke_points)
                            drawing_canvas.clear_stroke()
                            stroke_points = [index_tip]
                    elif not erasing:
                        drawing_canvas.clear_stroke()
                        stroke_points = [index_tip]
                    previous_draw_point = index_tip
                elif keep_stroke_open:
                    missing_draw_frames += 1
                else:
                    if not erasing:
                        finalize_stroke(drawing_canvas, letter_recognizer, stroke_points)
                    stroke_points = []
                    previous_draw_point = None
                    missing_draw_frames = 0

                frame = drawing_canvas.compose(frame)
                hand_tracker.draw_landmarks(frame, results)
                if index_tip is not None:
                    drawing_canvas.draw_cursor(frame, index_tip, erasing=erasing)

                current_time = time.perf_counter()
                instant_fps = 1.0 / max(current_time - previous_time, 0.0001)
                previous_time = current_time
                smoothed_fps = (
                    instant_fps if smoothed_fps == 0.0 else smoothed_fps * 0.9 + instant_fps * 0.1
                )

                display_frame, frame_bounds = fit_frame_to_display(frame, display_config)
                cursor_display_point = frame_point_to_display(
                    index_tip,
                    frame.shape,
                    frame_bounds,
                )
                hovered_action = toolbar.hit_test(
                    cursor_display_point if gesture_state.mode == GestureMode.MOVE and not pinch.active else None,
                    display_frame.shape[1],
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
                    missing_draw_frames = 0
                elif selected_action == ToolbarAction.ERASER:
                    active_toolbar_action = ToolbarAction.ERASER
                    erasing = True
                    drawing_canvas.clear_stroke()
                    stroke_points = []
                    previous_draw_point = None
                    missing_draw_frames = 0
                elif selected_action == ToolbarAction.THIN:
                    drawing_canvas.clear_stroke()
                    drawing_canvas.set_brush_thickness(7)
                    stroke_points = []
                    previous_draw_point = None
                    missing_draw_frames = 0
                elif selected_action == ToolbarAction.THICK:
                    drawing_canvas.clear_stroke()
                    drawing_canvas.set_brush_thickness(16)
                    stroke_points = []
                    previous_draw_point = None
                    missing_draw_frames = 0
                elif selected_action == ToolbarAction.CLEAR:
                    drawing_canvas.clear()
                    point_smoother.reset()
                    stroke_points = []
                    previous_draw_point = None
                    missing_draw_frames = 0
                elif selected_action == ToolbarAction.SAVE:
                    drawing_canvas.save(OUTPUT_DIR, save_filename())
                    stroke_points = []
                    previous_draw_point = None
                    missing_draw_frames = 0

                hand_detected = bool(results.hand_landmarks)
                draw_app_overlay(
                    display_frame,
                    frame_bounds,
                    hand_detected=hand_detected,
                    mode="Draw" if pinch.active else gesture_state.mode.value,
                    fps=smoothed_fps,
                )
                draw_toolbar(
                    display_frame,
                    toolbar,
                    active_toolbar_action,
                    hovered_action,
                )

                cv2.imshow(WINDOW_NAME, display_frame)
                key_code = cv2.waitKey(1) & 0xFF
                if should_quit(key_code):
                    break
                if should_clear(key_code):
                    drawing_canvas.clear()
                    point_smoother.reset()
                    stroke_points = []
                    previous_draw_point = None
                    missing_draw_frames = 0
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

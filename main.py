from __future__ import annotations

import time

import cv2

from camera import Camera, CameraConfig
from canvas import DrawingCanvas
from display import DisplayConfig, draw_phase_one_overlay, fit_frame_to_display
from hand_tracker import HandTracker, HandTrackerConfig


WINDOW_NAME = "Hand Gesture Air Drawing - Phase 2"
INDEX_FINGER_TIP = 8


def should_quit(key_code: int) -> bool:
    return key_code in (27, ord("q"), ord("Q"))


def should_clear(key_code: int) -> bool:
    return key_code in (ord("c"), ord("C"))


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
    previous_draw_point: tuple[int, int] | None = None

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
                index_tip = hand_tracker.get_landmark_pixel(
                    results,
                    frame.shape,
                    INDEX_FINGER_TIP,
                )
                drawing_active = index_tip is not None

                if index_tip is None:
                    previous_draw_point = None
                else:
                    if previous_draw_point is not None:
                        drawing_canvas.draw_line(previous_draw_point, index_tip)
                    previous_draw_point = index_tip

                frame = drawing_canvas.compose(frame)
                hand_tracker.draw_landmarks(frame, results)
                if index_tip is not None:
                    drawing_canvas.draw_cursor(frame, index_tip)

                current_time = time.perf_counter()
                instant_fps = 1.0 / max(current_time - previous_time, 0.0001)
                previous_time = current_time
                smoothed_fps = (
                    instant_fps if smoothed_fps == 0.0 else smoothed_fps * 0.9 + instant_fps * 0.1
                )

                display_frame, frame_bounds = fit_frame_to_display(frame, display_config)
                hand_detected = bool(results.hand_landmarks)
                draw_phase_one_overlay(
                    display_frame,
                    frame_bounds,
                    hand_detected=hand_detected,
                    drawing_active=drawing_active,
                    fps=smoothed_fps,
                )

                cv2.imshow(WINDOW_NAME, display_frame)
                key_code = cv2.waitKey(1) & 0xFF
                if should_quit(key_code):
                    break
                if should_clear(key_code):
                    drawing_canvas.clear()
                    previous_draw_point = None
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Hand Gesture Air Drawing System

This project has two gesture-controlled experiences:

- Air drawing: draw with your hand, use a gesture toolbar, clean strokes, and snap simple strokes into clean letters or digits.
- Gesture puzzle: capture a webcam image and solve a 3x3 tile puzzle with pinch gestures.

## Requirements

- Python 3.10+
- A working webcam

Install dependencies:

```powershell
D:\Python3\python.exe -m pip install -r requirements.txt
```

## Adjust Tracking Settings

Runtime thresholds are stored in `airgesture/config/settings.json`. Restart the application after editing it.

- `camera`: camera index, resolution, FPS, mirroring, and frame buffer size
- `air_drawing.cursor_smoothing`: cursor smoothing alpha and missing-frame tolerance
- `air_drawing.pinch`: pinch/release distances and missing-frame tolerance
- `air_drawing.drawing`: stroke bridging, recognition display time, brush, and eraser sizes
- `puzzle.capture`: spread ratio, stable-frame count, centering, and motion thresholds
- `puzzle.game`: countdown, default difficulty, and board size
- `tracker.landmark_filter`: adaptive landmark smoothing values for each mode

Useful tuning directions:

- Increase smoothing `alpha` for faster response; decrease it for steadier movement.
- Increase `pinch_threshold` if pinching is hard to trigger. Keep `release_threshold` higher.
- Decrease `spread_ratio_required` if puzzle auto-capture is difficult to activate.
- Decrease `stable_frames_required` or increase motion thresholds if holding still is difficult.
- Increase `thin_brush_size`, `thick_brush_size`, or `eraser_size` to change tool sizes.

Invalid JSON, unknown fields, and unsafe values are rejected with a clear startup error.

## Run Main Menu

```powershell
D:\Python3\python.exe -m airgesture
```

Menu controls:

- `1`: Air Drawing
- `2`: Gesture Puzzle
- `W` / `S`: select menu item
- `Enter`: open selected item
- `Q` or `Esc`: quit

Calibration:

- Choosing a mode from the main menu opens a webcam calibration screen first
- `Space`: continue when status is ready
- `Enter`: skip calibration and continue anyway
- `Q` or `Esc`: cancel back to the menu

## Run Air Drawing

```powershell
D:\Python3\python.exe -m airgesture.drawing.main
```

Air drawing controls:

- Pinch thumb + index finger: draw or erase
- Index + middle fingers: move cursor without drawing and select toolbar items
- Draw a one-stroke letter or digit (`A-Z`, `0-9`), then release pinch: snap it into a clean symbol when recognized
- The top overlay briefly shows `Phat hien: X` after a symbol is detected
- `c`: clear drawing canvas
- `q`: quit
- `Esc`: quit

Toolbar:

- `Red`, `Green`, `Blue`, `Yellow`, `White`: change brush color
- `Erase`: erase parts of the canvas
- `Thin`, `Thick`: change brush size
- `Clear`: clear the canvas
- `Save`: save the drawing to `outputs/saved_drawings`

## Run Gesture Puzzle

```powershell
D:\Python3\python.exe -m airgesture.puzzle.main
```

Puzzle controls:

- Two-hand capture gesture: show both hands, open them wide enough to frame the shot, then hold still briefly for auto capture
- `3` / `4`: choose 3x3 or 4x4 difficulty before capture
- `Space`, `Enter`, or `C`: fallback capture
- The HUD shows `Hands: 0/2`, `1/2`, or `2/2`; auto capture starts when both hands are visible and stable
- A short countdown runs after capture before the puzzle starts
- Move hand: control cursor
- Pinch on a tile: grab/select tile
- Release over another tile: swap tiles
- `r`: restart during play or after victory
- `q` or `Esc`: quit

## Current Scope

Implemented:

- Webcam capture with mirrored preview
- MediaPipe hand landmark detection
- MediaPipe video-mode tracking with monotonic frame timestamps
- Adaptive One Euro filtering across all 21 hand landmarks
- Separate tracking profiles for drawing, puzzle, and calibration
- Two-frame cursor and pinch dropout tolerance for brief detection loss
- Local MediaPipe Tasks model at `models/hand_landmarker.task`
- Home menu launcher for selecting drawing or puzzle mode
- Webcam calibration screen with hand-count and brightness checks
- Gesture-controlled 3x3 webcam puzzle game
- 3x3 and 4x4 puzzle difficulty selection
- Countdown before puzzle start
- Pinch gesture tile selection and swapping
- Puzzle timer, move counter, cursor, and victory screen
- Debug landmark drawing
- Smoothed index fingertip drawing on a separate virtual canvas
- Opaque high-saturation drawing colors for stronger strokes
- Short tracking-drop tolerance to reduce broken strokes
- Stroke-based cleanup after each completed drawing gesture
- Pinch-to-draw gesture with hysteresis for fewer accidental strokes
- Gesture modes for draw, move, and idle
- Gesture toolbar for colors, eraser, brush size, clear, and save
- Template recognition for one-stroke `A-Z` letters and `0-9` digits
- Clear canvas with `c`
- 16:9 preview frame that preserves camera aspect ratio
- Basic status overlay with hand detection state and FPS
- Safe camera/window cleanup

Not implemented yet:

- Recognition for full handwriting words
- General OCR or handwriting recognition model

## Project Structure

```text
airgesture/
|-- __main__.py
|-- app.py
|-- calibration.py
|-- paths.py
|-- config/
|   |-- settings.py
|   `-- settings.json
|-- core/
|   |-- camera.py
|   |-- hand_tracker.py
|   `-- smoothing.py
|-- drawing/
|   |-- main.py
|   |-- canvas.py
|   |-- display.py
|   |-- gesture_controller.py
|   |-- letter_recognizer.py
|   `-- toolbar.py
|-- puzzle/
|   |-- main.py
|   |-- board.py
|   |-- capture_gesture.py
|   |-- gesture.py
|   `-- hud.py
`-- ui/
    `-- theme.py

models/
outputs/
tests/
```

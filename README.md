# Hand Gesture Air Drawing System

The app opens the default webcam, detects one hand with MediaPipe Hands, follows a smoothed index fingertip point, draws strokes only in draw mode, supports a gesture toolbar, cleans each completed stroke, and can snap simple strokes into clean letters or digits.

## Requirements

- Python 3.10+
- A working webcam

Install dependencies:

```powershell
D:\Python3\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
D:\Python3\python.exe main.py
```

Controls:

- Index finger only: draw
- Index + middle fingers: move cursor without drawing and select toolbar items
- Draw a simple letter or digit, then leave draw mode: snap it into a clean symbol when recognized
- `c`: clear drawing canvas
- `q`: quit
- `Esc`: quit

Toolbar:

- `Red`, `Green`, `Blue`, `Yellow`, `White`: change brush color
- `Erase`: erase parts of the canvas
- `Thin`, `Thick`: change brush size
- `Clear`: clear the canvas
- `Save`: save the drawing to `outputs/saved_drawings`

## Current Scope

Implemented:

- Webcam capture with mirrored preview
- MediaPipe hand landmark detection
- Local MediaPipe Tasks model at `models/hand_landmarker.task`
- Debug landmark drawing
- Smoothed index fingertip drawing on a separate virtual canvas
- Opaque high-saturation drawing colors for stronger strokes
- Short tracking-drop tolerance to reduce broken strokes
- Stroke-based cleanup after each completed drawing gesture
- Gesture modes for draw, move, and idle
- Gesture toolbar for colors, eraser, brush size, clear, and save
- Template recognition for `C`, `O`, `L`, `V`, `Z`, `S`, `1`, `2`, and `3`
- Clear canvas with `c`
- 16:9 preview frame that preserves camera aspect ratio
- Basic status overlay with hand detection state and FPS
- Safe camera/window cleanup

Not implemented yet:

- Recognition for full handwriting words
- Handwriting or digit recognition model

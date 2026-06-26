# Hand Gesture Air Drawing System

Phase 2 adds a virtual drawing canvas on top of webcam-based hand tracking. The app opens the default webcam, detects one hand with MediaPipe Hands, follows the index fingertip, draws strokes on a separate canvas, and exits safely with `q` or `Esc`.

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

- Move index fingertip: draw
- `c`: clear drawing canvas
- `q`: quit
- `Esc`: quit

## Current Scope

Implemented:

- Webcam capture with mirrored preview
- MediaPipe hand landmark detection
- Local MediaPipe Tasks model at `models/hand_landmarker.task`
- Debug landmark drawing
- Index fingertip drawing on a separate virtual canvas
- Clear canvas with `c`
- 16:9 preview frame that preserves camera aspect ratio
- Basic status overlay with hand detection state and FPS
- Safe camera/window cleanup

Not implemented yet:

- Gesture-based drawing mode
- Color, eraser, brush size, or save image tools

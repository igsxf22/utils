# grabber

macOS screen region capture as numpy arrays, with a visual overlay border and optional MJPEG streaming.

Requires macOS (uses native CoreGraphics and AppKit APIs) and Python 3.14+.

## Setup

```bash
uv init grabber && cd grabber
uv add numpy opencv-python pyobjc-framework-cocoa pyobjc-framework-quartz
```

Then drop in the Python files:
- `grabber.py` — screen capture class
- `mjpeg_stream.py` — MJPEG streaming server
- `stream_screen_example.py` — integration example

## Usage

### Basic capture

```python
from grabber import ScreenGrabber

# Capture a 400x300 region starting at (100, 100)
grabber = ScreenGrabber(x=100, y=100, width=400, height=300)
grabber.show_overlay()  # green border so you can see the region

frame = grabber.frame  # (300, 400, 3) RGB uint8 numpy array

grabber.close()
```

### Stream to browser

```bash
python stream_screen_example.py --x 100 --y 100 --width 640 --height 480 --port 8080
```

Then open `http://localhost:8080` in a browser.

### MJPEG server standalone (webcam or simulated)

```bash
python mjpeg_stream.py --sim          # random noise frames
python mjpeg_stream.py --camera-id 0  # real webcam
```

## Permissions

macOS requires Screen Recording permission for screen capture. Grant it in **System Settings > Privacy & Security > Screen Recording** for your terminal app or Python.

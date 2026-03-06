#!/usr/bin/env python3
"""
Stream a screen region as MJPEG via the mjpeg_stream server.

Combines ScreenGrabber (screen capture) with the MJPEG server to let you
view a portion of your screen in a browser or consume it as a video stream.

Three things run at once:
  1. A producer thread that captures screen frames and pushes them to a queue.
  2. An HTTP server thread that streams those frames as MJPEG.
  3. The main loop that prints frame info to the terminal once per second.

Usage:
    python stream_screen_example.py --x 100 --y 100 --width 640 --height 480 --port 8080

Then open http://localhost:8080 in a browser, or consume the raw stream at
http://localhost:8080/stream with cv2.VideoCapture(url).
"""

import argparse
import queue
import threading
import time

import cv2
import numpy as np

from grabber import ScreenGrabber
from mjpeg_stream import PooledHTTPServer, StreamingHandler, frame_queue

# The producer thread writes the latest frame here so the main loop can
# read it without doing its own screen capture (which would conflict).
_latest_frame: np.ndarray | None = None
_frame_lock = threading.Lock()


def screen_producer(grabber: ScreenGrabber, q: queue.Queue, fps: int):
    """Continuously capture the screen and push frames to the MJPEG queue.

    Runs in a daemon thread. Captures at the target FPS, converts from
    RGB (what ScreenGrabber returns) to BGR (what OpenCV/MJPEG expects),
    and drops old frames if the queue is backing up.

    Args:
        grabber: The ScreenGrabber instance to capture from.
        q:       The shared frame queue that the MJPEG server reads from.
        fps:     Target frames per second.
    """
    global _latest_frame
    interval = 1.0 / fps

    while True:
        t0 = time.time()

        # Grab the screen region as an RGB numpy array
        frame = grabber.frame

        # Store a copy for the main loop's status display
        with _frame_lock:
            _latest_frame = frame

        # Convert RGB to BGR for the MJPEG handler (OpenCV uses BGR)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Push to queue, dropping stale frames to keep latency low
        try:
            while q.qsize() > 2:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            q.put_nowait(bgr)
        except queue.Full:
            pass  # Skip this frame if queue is still full

        # Sleep just enough to hit the target FPS
        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))


def main():
    parser = argparse.ArgumentParser(description="Stream a screen region as MJPEG")
    parser.add_argument("--x", type=int, default=100, help="Left edge of capture region")
    parser.add_argument("--y", type=int, default=100, help="Top edge of capture region")
    parser.add_argument("--width", type=int, default=640, help="Width of capture region")
    parser.add_argument("--height", type=int, default=480, help="Height of capture region")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second")
    parser.add_argument("--no-overlay", action="store_true", help="Hide the screen overlay")
    args = parser.parse_args()

    # Set up the screen grabber for the specified region
    grabber = ScreenGrabber(args.x, args.y, args.width, args.height)
    if not args.no_overlay:
        grabber.show_overlay()  # Green border so you can see the capture area

    # Start the producer thread — it captures frames and feeds the queue
    threading.Thread(
        target=screen_producer,
        args=(grabber, frame_queue, args.fps),
        daemon=True,
    ).start()

    # Start the MJPEG HTTP server in the background
    server = PooledHTTPServer(("0.0.0.0", args.port), StreamingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"Streaming screen region ({args.x}, {args.y}, {args.width}x{args.height})")
    print(f"  http://localhost:{args.port}")
    print(f"  http://localhost:{args.port}/stream")
    print()

    # Main loop: print frame stats once per second until Ctrl+C
    try:
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame is not None:
                # Show the frame shape and the color of the center pixel
                # as a quick sanity check that capture is working
                cy, cx = frame.shape[0] // 2, frame.shape[1] // 2
                r, g, b = frame[cy, cx]
                print(f"{frame.shape}  center=({r},{g},{b})")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        grabber.close()
        server.shutdown()


if __name__ == "__main__":
    main()

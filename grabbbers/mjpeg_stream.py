#!/usr/bin/env python3
"""
Motion JPEG streaming server with modular numpy array input.

Serves an MJPEG stream at /stream that can be consumed by any client that
understands multipart JPEG — most commonly a browser <img> tag, or
cv2.VideoCapture(url) in a Python script.

The server doesn't capture frames itself. Instead, a "producer" thread
feeds numpy arrays into a shared queue, and the HTTP handler encodes
and streams them as JPEG. This makes it easy to swap in different frame
sources (webcam, simulated noise, screen capture, etc.).

Usage:
    python mjpeg_stream.py [OPTIONS]

    --host HOST         Host for HTTP server (default: 0.0.0.0)
    --port PORT         Port for HTTP server (default: 8080)
    --camera-id ID      Camera ID for cv2.VideoCapture (default: 0)
    --fps FPS           Frames per second (default: 30)
    --sim               Use simulated random-noise frames instead of a real camera
"""

import argparse
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame queue — the bridge between producers and the HTTP streaming handler.
# Producers push numpy arrays in; the StreamingHandler pops them out.
# maxsize=5 prevents memory buildup if the consumer is slower than producer.
# ---------------------------------------------------------------------------
frame_queue = queue.Queue(maxsize=5)

# The active frame producer (set in main). Kept as a global so we can
# stop it cleanly on shutdown.
camera_producer = None


class SimFrameProducer:
    """Generates random noise frames for testing (no real camera needed).

    Useful for verifying the stream works before connecting a real source.

    Args:
        frame_queue: The shared queue to push frames into.
        width:       Frame width in pixels.
        height:      Frame height in pixels.
        fps:         How many frames per second to generate.
    """

    def __init__(self, frame_queue, width=640, height=480, fps=30):
        self.frame_queue = frame_queue
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.thread = None

    def start(self):
        """Start generating frames in a background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._produce_loop, daemon=True)
        self.thread.start()

    def _produce_loop(self):
        frame_interval = 1.0 / self.fps
        while self.running:
            start_time = time.time()

            # Create a random static/noise frame (like an old TV with no signal)
            frame = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Drop the frame — better to skip than to lag behind

            # Sleep just enough to maintain the target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        """Stop generating frames."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


class CameraFrameProducer:
    """Reads frames from a real webcam using OpenCV and pushes them to a queue.

    The camera is opened immediately on init so errors are caught early.
    Call start() to begin the background capture loop.

    Args:
        frame_queue: The shared queue to push frames into.
        camera_id:   Which camera to use (0 = default/built-in).
        fps:         Target frames per second.
    """

    def __init__(self, frame_queue, camera_id=0, fps=30):
        self.frame_queue = frame_queue
        self.camera_id = camera_id
        self.fps = fps
        self.running = False
        self.thread = None

        print(f"Initializing camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)

        # Some cameras need a moment to warm up
        time.sleep(1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        # Request a standard resolution and frame rate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Read and throw away a few frames — cameras often return
        # black or corrupt frames right after opening
        for _ in range(5):
            self.cap.read()

        # Make sure we can actually get a usable frame
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            raise RuntimeError("Camera opened but cannot read frames")

        print(f"Camera initialized: {test_frame.shape}")

    def start(self):
        """Start capturing frames in a background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("Camera capture thread started")

    def _capture_loop(self):
        """Continuously read frames from the camera and push to the queue."""
        frame_interval = 1.0 / self.fps

        while self.running:
            start_time = time.time()

            ret, frame = self.cap.read()
            if ret and frame is not None:
                try:
                    # If frames are piling up, drop old ones so the stream
                    # always shows the most recent frame (low latency)
                    while self.frame_queue.qsize() > 2:
                        try:
                            self.frame_queue.get_nowait()
                        except Exception:
                            break
                    self.frame_queue.put_nowait(frame)
                except Exception:
                    pass  # Queue full, just skip this frame

            # Sleep to maintain the target frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        """Stop capturing and release the camera."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print("Camera capture stopped")


# ---------------------------------------------------------------------------
# HTTP server and handler
# ---------------------------------------------------------------------------

# How many clients can watch the stream at the same time
MAX_CONNECTIONS = 4


class PooledHTTPServer(HTTPServer):
    """HTTP server that handles each request in a thread pool.

    The standard HTTPServer processes one request at a time. This version
    uses a thread pool so multiple clients can watch the stream at once,
    up to MAX_CONNECTIONS.
    """

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self._pool = ThreadPoolExecutor(max_workers=MAX_CONNECTIONS)

    def process_request(self, request, client_address):
        """Hand off each incoming request to the thread pool."""
        self._pool.submit(self.process_request_thread, request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


class StreamingHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the MJPEG stream.

    Routes:
        /        — Simple HTML page with an <img> tag showing the stream.
        /stream  — Raw MJPEG stream (multipart JPEG over HTTP).
    """

    def log_message(self, format, *args):
        """Suppress the default per-request log messages."""
        pass

    def do_GET(self):
        if self.path == '/':
            # Serve a minimal HTML page that displays the stream
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html().encode('utf-8'))

        elif self.path == '/stream':
            # Start an MJPEG stream — this connection stays open and
            # continuously sends JPEG frames separated by boundaries.
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()

            try:
                while True:
                    # Wait for the next frame from the producer
                    try:
                        frame = frame_queue.get(timeout=1)
                    except queue.Empty:
                        continue  # No frame ready yet, try again

                    # Compress the numpy array to JPEG (quality 85 is a
                    # good balance between file size and visual quality)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    _, jpeg = cv2.imencode('.jpg', frame, encode_param)
                    jpeg_bytes = jpeg.tobytes()

                    # Send this frame as one part of the multipart response
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(jpeg_bytes))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b'\r\n')
                    except (BrokenPipeError, ConnectionResetError):
                        break  # Client disconnected

            except Exception as e:
                print(f"Streaming error: {e}")

        else:
            self.send_error(404)

    def get_html(self):
        """Minimal HTML page that shows the stream fullscreen."""
        return """<!DOCTYPE html>
<html><body style="margin:0"><img src="/stream" style="width:100%;height:100vh;object-fit:contain"></body></html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Motion JPEG streaming server"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument(
        "--camera-id", type=int, default=0, help="Camera ID (default: 0)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--sim", action="store_true", default=False,
        help="Use simulated random-noise frames instead of a real camera"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Motion JPEG Streaming Server")
    print("=" * 60)

    # Choose a frame source: simulated noise or real webcam
    global camera_producer
    if args.sim:
        camera_producer = SimFrameProducer(
            frame_queue,
            width=640,
            height=480,
            fps=args.fps
        )
        camera_producer.start()
        print(f"Simulated camera started successfully")
    else:
        try:
            camera_producer = CameraFrameProducer(
                frame_queue,
                camera_id=args.camera_id,
                fps=args.fps
            )
            camera_producer.start()
            print(f"Camera {args.camera_id} started successfully")
        except Exception as e:
            print(f"Failed to start camera: {e}")
            print("\nTroubleshooting:")
            print("1. Check camera permissions (System Settings > Privacy > Camera)")
            print("2. Make sure no other app is using the camera")
            print("3. Try: python test_camera.py")
            return 1

    # Figure out the local IP so you can access the stream from other devices
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "YOUR_IP"

    print(f"🌐 Local:  http://localhost:{args.port}")
    print(f"📱 iPhone: http://{local_ip}:{args.port}")
    print("=" * 60)
    print()

    # Start the HTTP server and block until Ctrl+C
    try:
        server = PooledHTTPServer((args.host, args.port), StreamingHandler)
        print("Server started. Press Ctrl+C to stop.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        if camera_producer:
            camera_producer.stop()
        server.shutdown()
        print("Server stopped.")


if __name__ == "__main__":
    exit(main() or 0)

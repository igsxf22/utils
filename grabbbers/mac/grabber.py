"""
Screen region capture with overlay for macOS.

Uses macOS-native CoreGraphics to capture a rectangular area of the screen
and return it as a numpy array. Can also draw a colored border overlay on
screen so you can see exactly what region is being captured.

Requires macOS Screen Recording permission (System Settings > Privacy).
"""

import multiprocessing

import numpy as np
from Quartz import (
    CGDataProviderCopyData,
    CGImageGetBytesPerRow,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CGRectMake,
    CGWindowListCreateImage,
    kCGWindowImageDefault,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
)


class ScreenGrabber:
    """Captures a screen region as numpy arrays with an optional visual overlay.

    Coordinates use macOS screen points (not pixels). On Retina displays the
    captured image is automatically downscaled to match your requested
    width/height, so you always get exactly the dimensions you asked for.

    Can be used as a context manager:
        with ScreenGrabber(100, 100, 400, 300) as grabber:
            frame = grabber.frame

    Args:
        x:      Left edge of capture region in screen points.
        y:      Top edge of capture region in screen points.
        width:  Width of capture region in screen points.
        height: Height of capture region in screen points.
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # Overlay runs in a separate process so it doesn't block your code
        self._overlay_proc: multiprocessing.Process | None = None

    @property
    def frame(self) -> np.ndarray:
        """Grab and return the current frame. Shorthand for grab()."""
        return self.grab()

    def grab(self) -> np.ndarray:
        """Capture the screen region right now.

        Returns:
            numpy array with shape (height, width, 3), dtype uint8, in RGB order.
        """
        # Define the rectangle to capture (in screen-point coordinates)
        cg_rect = CGRectMake(self.x, self.y, self.width, self.height)

        # Ask macOS to screenshot that region — captures all visible windows
        cg_image = CGWindowListCreateImage(
            cg_rect,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
            kCGWindowImageDefault,
        )
        if cg_image is None:
            raise RuntimeError("Screen capture failed — check Screen Recording permission in System Settings")

        # Get the raw pixel dimensions (may be 2x on Retina displays)
        px_w = CGImageGetWidth(cg_image)
        px_h = CGImageGetHeight(cg_image)
        bpr = CGImageGetBytesPerRow(cg_image)  # bytes per row (may include padding)

        # Copy the raw bitmap data into a flat bytes buffer
        data = CGDataProviderCopyData(CGImageGetDataProvider(cg_image))

        # Reshape into a 2D pixel array, trimming any row padding
        arr = np.frombuffer(data, dtype=np.uint8).reshape(px_h, bpr)
        arr = arr[:, : px_w * 4].reshape(px_h, px_w, 4)

        # macOS gives us BGRA channel order — rearrange to standard RGB
        arr = arr[:, :, [2, 1, 0]]

        # On Retina displays the image is 2x the requested size.
        # Downscale by averaging blocks of pixels (e.g. every 2x2 block → 1 pixel).
        if px_w != self.width or px_h != self.height:
            scale_x = px_w // self.width
            scale_y = px_h // self.height
            arr = arr.reshape(self.height, scale_y, self.width, scale_x, 3)
            arr = arr.mean(axis=(1, 3)).astype(np.uint8)

        return arr

    def show_overlay(self, color: tuple[float, float, float] = (0, 1, 0), line_width: float = 2.0):
        """Draw a colored border on screen showing the capture region.

        The overlay is click-through (doesn't interfere with mouse clicks),
        stays on top of all windows, and runs in a background process.

        Args:
            color:      RGB color as floats from 0.0 to 1.0. Default is green.
            line_width: Thickness of the border in points.
        """
        # Don't start a second overlay if one is already running
        if self._overlay_proc and self._overlay_proc.is_alive():
            return

        # Launch the overlay in a separate process because macOS requires
        # its own event loop (NSApplication.run) for drawing windows.
        # daemon=True means it auto-terminates when the main script exits.
        self._overlay_proc = multiprocessing.Process(
            target=_run_overlay,
            args=(self.x, self.y, self.width, self.height, color, line_width),
            daemon=True,
        )
        self._overlay_proc.start()

    def hide_overlay(self):
        """Remove the overlay from the screen."""
        if self._overlay_proc and self._overlay_proc.is_alive():
            self._overlay_proc.terminate()
            self._overlay_proc.join(timeout=1)
        self._overlay_proc = None

    def close(self):
        """Clean up — removes the overlay if it's showing."""
        self.hide_overlay()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _run_overlay(x, y, w, h, color, line_width):
    """Draw a transparent overlay window on screen. Runs in its own process.

    This function never returns (it starts an AppKit event loop). It's meant
    to be launched via multiprocessing.Process and terminated externally.

    The overlay window is slightly larger than the capture region so the
    border is drawn just outside it and doesn't appear in captured frames.
    """
    # AppKit imports are done here (not at module level) because this code
    # only runs in the overlay subprocess — no need to load AppKit in the
    # main process.
    from AppKit import (
        NSApplication,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSColor,
        NSScreen,
        NSView,
        NSWindow,
        NSWindowStyleMaskBorderless,
    )

    # Extra padding around the capture region so the border sits outside it
    pad = int(line_width) + 2

    # Custom view that draws just the colored border outline
    class OverlayView(NSView):
        def drawRect_(self, dirty):
            # Fill with transparent background
            NSColor.clearColor().set()
            NSBezierPath.fillRect_(self.bounds())

            # Draw the colored border rectangle
            r, g, b = color
            NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 1.0).set()
            path = NSBezierPath.bezierPathWithRect_(((pad, pad), (w, h)))
            path.setLineWidth_(line_width)
            path.stroke()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)  # "Accessory" mode — no icon in the Dock

    # macOS uses bottom-left origin, so convert our top-left y coordinate
    screen_h = NSScreen.mainScreen().frame().size.height
    win_frame = ((x - pad, screen_h - y - h - pad), (w + 2 * pad, h + 2 * pad))

    # Create a borderless, transparent, always-on-top window
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        win_frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
    )
    window.setBackgroundColor_(NSColor.clearColor())
    window.setOpaque_(False)
    window.setHasShadow_(False)
    window.setLevel_(1000)              # Stay above all normal windows
    window.setIgnoresMouseEvents_(True)  # Clicks pass through to apps below
    window.setCollectionBehavior_(1 << 4)  # Visible on all Spaces/desktops

    view = OverlayView.alloc().initWithFrame_(((0, 0), (w + 2 * pad, h + 2 * pad)))
    window.setContentView_(view)
    window.orderFrontRegardless()

    # Start the macOS event loop (blocks forever until the process is killed)
    app.run()


if __name__ == "__main__":
    import cv2

    # Quick demo: capture a 400x300 region at (100, 100) and show live preview
    g = ScreenGrabber(100, 100, 400, 300)
    g.show_overlay()

    # Display frames in a window — press 'q' to quit
    while True:
        frame = g.frame
        # Convert RGB (what ScreenGrabber returns) to BGR (what OpenCV expects)
        cv2.imshow("ScreenGrabber", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    g.close()
    cv2.destroyAllWindows()

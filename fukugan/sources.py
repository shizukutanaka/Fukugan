"""Frame sources: real USB cameras (OpenCV) and a virtual test-pattern camera.

Every source honours the same minimal contract:

    open()  -> None (raises SourceError with a reason on failure)
    read()  -> (monotonic_timestamp, frame) or None when no frame is ready
    close() -> None

The virtual camera is a product feature, not just a test stub: it lets a
user without cameras verify the whole pipeline end to end.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import CameraConfig

# The frame counter is drawn as a barcode strip: BARCODE_BITS white/black
# cells at the top-left of every virtual frame, MSB first, preceded by one
# always-white sync cell. Tests (and users) can read the counter back from
# any downstream frame -- including one decoded from the HLS stream -- to
# prove that frames actually advance through the pipeline.
BARCODE_BITS = 24
BARCODE_CELL = 8  # pixels per cell
BARCODE_HEIGHT = 16


class SourceError(RuntimeError):
    """A source failed; the message always carries the reason."""


def draw_frame_counter(frame: np.ndarray, counter: int) -> np.ndarray:
    """Draw the binary barcode strip encoding `counter` onto `frame`."""
    h, w = frame.shape[:2]
    strip_w = (BARCODE_BITS + 1) * BARCODE_CELL
    if h < BARCODE_HEIGHT or w < strip_w:
        return frame
    strip = frame[0:BARCODE_HEIGHT, 0:strip_w]
    strip[:, :] = 0
    # sync cell: always white
    strip[:, 0:BARCODE_CELL] = 255
    for i in range(BARCODE_BITS):
        bit = (counter >> (BARCODE_BITS - 1 - i)) & 1
        if bit:
            x0 = (i + 1) * BARCODE_CELL
            strip[:, x0 : x0 + BARCODE_CELL] = 255
    return frame


def read_frame_counter(region: np.ndarray) -> int | None:
    """Read the counter back from a frame (or the top-left region of one).

    Returns None when the strip does not look like a barcode (sync cell
    missing), so callers can distinguish "not a virtual frame" from 0.
    """
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    if gray.shape[0] < BARCODE_HEIGHT or gray.shape[1] < (BARCODE_BITS + 1) * BARCODE_CELL:
        return None
    strip = gray[0:BARCODE_HEIGHT, :]
    cell_means = []
    for i in range(BARCODE_BITS + 1):
        cell = strip[:, i * BARCODE_CELL : (i + 1) * BARCODE_CELL]
        cell_means.append(float(cell.mean()))
    if cell_means[0] < 128:  # sync cell must be white
        return None
    value = 0
    for i in range(BARCODE_BITS):
        value = (value << 1) | (1 if cell_means[i + 1] >= 128 else 0)
    return value


def render_test_pattern(
    width: int,
    height: int,
    counter: int,
    *,
    hue: int = 0,
    parallax: int = 24,
    label: str = "",
) -> np.ndarray:
    """Render one deterministic virtual-camera frame.

    Rectangles sit at x positions shifted by `parallax` relative to a
    sibling camera with a different parallax, so a composed side-by-side
    frame visibly contains two different viewpoints. A marker moves with
    the counter so consecutive frames always differ.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Background: vertical gradient tinted by hue, distinct per camera.
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :] = (60, 60, 60)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hue % 180
    hsv[:, :, 1] = 90
    grad = np.linspace(70, 130, height, dtype=np.uint8)
    hsv[:, :, 2] = grad[:, None]
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Static "scene" rectangles at several depths (parallax scales depth).
    depths = [(0.25, 0.35, 90, 130, (0, 200, 255)),
              (0.55, 0.30, 120, 180, (0, 255, 120)),
              (0.40, 0.60, 160, 90, (255, 160, 0))]
    for fx, fy, rw, rh, color in depths:
        cx = int(fx * width) + int(parallax * (rw / 120.0))
        cy = int(fy * height)
        cv2.rectangle(frame, (cx, cy), (min(cx + rw, width - 1), min(cy + rh, height - 1)), color, -1)
        cv2.rectangle(frame, (cx, cy), (min(cx + rw, width - 1), min(cy + rh, height - 1)), (255, 255, 255), 2)

    # Moving marker: proves temporal change even in a still thumbnail.
    mx = int((counter * 7) % max(1, (width - 60)))
    cv2.circle(frame, (mx + 30, height - 50), 22, (255, 255, 255), -1)
    cv2.circle(frame, (mx + 30, height - 50), 22, (0, 0, 0), 2)

    cv2.putText(
        frame,
        f"#{counter}",
        (strip_offset(width), 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if label:
        cv2.putText(
            frame,
            label,
            (20, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    draw_frame_counter(frame, counter)
    return frame


def strip_offset(width: int) -> int:
    strip_w = (BARCODE_BITS + 1) * BARCODE_CELL
    return min(strip_w + 16, max(0, width - 200))


class DeviceCameraSource:
    """A real camera accessed through OpenCV VideoCapture."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.name = cfg.label or f"Camera {cfg.index}"
        self._cap: cv2.VideoCapture | None = None
        self._failures = 0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.cfg.index)
        if not cap.isOpened():
            cap.release()
            raise SourceError(
                f"{self.name}: cannot open device index {self.cfg.index} "
                "(not connected, or already in use by another application)"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # Keep the driver buffer shallow so we track the live edge instead
        # of accumulating latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def read(self) -> tuple[float, np.ndarray] | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        ts = time.monotonic()
        if not ok or frame is None:
            self._failures += 1
            return None
        return ts, frame

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    def note_success(self) -> None:
        self._failures = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VirtualCameraSource:
    """Generates test-pattern frames at the configured rate.

    Two instances with different `parallax` values behave like two
    cameras looking at the same scene from different positions.
    """

    def __init__(self, cfg: CameraConfig, *, parallax: int = 24):
        self.cfg = cfg
        self.name = cfg.label or f"Virtual {cfg.index}"
        self.parallax = parallax
        self._counter = 0
        self._next_ts: float | None = None
        self._opened = False

    def open(self) -> None:
        self._opened = True
        self._next_ts = time.monotonic()

    def read(self) -> tuple[float, np.ndarray] | None:
        if not self._opened:
            return None
        now = time.monotonic()
        period = 1.0 / max(1, self.cfg.fps)
        if self._next_ts is not None and now < self._next_ts:
            return None
        if self._next_ts is None or now - self._next_ts > period * 4:
            self._next_ts = now
        self._next_ts += period
        frame = render_test_pattern(
            self.cfg.width,
            self.cfg.height,
            self._counter,
            hue=self.cfg.hue,
            parallax=self.parallax,
            label=self.name,
        )
        ts = time.monotonic()
        self._counter += 1
        return ts, frame

    @property
    def counter(self) -> int:
        return self._counter

    def close(self) -> None:
        self._opened = False


def make_source(cfg: CameraConfig, *, virtual_parallax: int = 24):
    if cfg.type == "virtual":
        return VirtualCameraSource(cfg, parallax=virtual_parallax)
    return DeviceCameraSource(cfg)

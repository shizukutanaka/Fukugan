"""Pipeline orchestration: capture -> sync -> compose -> encode.

One reader thread per source feeds short per-source histories; the main
loop synchronizes, composes and writes raw frames to FFmpeg's stdin.
Every state transition and failure carries a human-readable reason that
is exposed through `status()` -- nothing fails silently.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque

import numpy as np

from . import calibrate as calibrate_mod
from .compose import StereoComposer
from .config import Config
from .encoder import EncoderChoice, build_command, select_encoder
from .sources import SourceError, make_source
from .sync import FrameSynchronizer

HISTORY_LEN = 8


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = "idle"  # idle|starting|running|degraded|error
        self.reasons: list[str] = []
        self.encoder_choice: EncoderChoice | None = None
        self.stats = {
            "frames_composed": 0,
            "frames_written": 0,
            "sync_events": 0,
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sources: list = []
        self._histories: list[deque] = []
        self._proc: subprocess.Popen | None = None
        self._latest_preview: np.ndarray | None = None
        self._calibration_warnings: list[str] = []
        self._started_at: float | None = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> tuple[bool, str]:
        with self._lock:
            if self.state in ("starting", "running", "degraded"):
                return False, "pipeline is already running"
            errors = self.cfg.validate()
            if errors:
                self.state = "error"
                self.reasons = errors
                return False, "; ".join(errors)
            self.state = "starting"
            self.reasons = []
        try:
            self._open_sources()
            self._load_calibration()
            self.encoder_choice = select_encoder(
                self.cfg.encoder.preference, self.cfg.encoder.preset
            )
            self._start_ffmpeg()
        except (SourceError, OSError, RuntimeError) as exc:
            self._fail(str(exc))
            return False, str(exc)
        self._stop.clear()
        self._started_at = time.monotonic()
        with self._lock:
            self.state = "running"
            self.reasons = [self.encoder_choice.reason] + self._calibration_warnings
        return True, "started"

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = []
        for src in self._sources:
            try:
                src.close()
            except Exception:
                pass
        self._sources = []
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None
        with self._lock:
            if self.state != "error":
                self.state = "idle"
            self.reasons = []

    def _fail(self, reason: str) -> None:
        with self._lock:
            self.state = "error"
            self.reasons = [reason]
        for src in self._sources:
            try:
                src.close()
            except Exception:
                pass
        self._sources = []

    # -- setup ------------------------------------------------------------

    def _open_sources(self) -> None:
        n = len(self.cfg.cameras)
        ref = self.cfg.reference_index()
        self._sources = []
        self._source_ok: list[bool] = []
        self._histories = [deque(maxlen=HISTORY_LEN) for _ in range(n)]
        failures: list[str] = []
        for i, cam_cfg in enumerate(self.cfg.cameras):
            # Virtual cameras fan out symmetrically around the center so
            # the outermost pair shows the widest parallax.
            parallax = 24 + 14 * abs(i - ref)
            src = make_source(cam_cfg, virtual_parallax=parallax)
            try:
                src.open()
                self._source_ok.append(True)
            except SourceError as exc:
                failures.append(str(exc))
                self._source_ok.append(False)
            self._sources.append(src)
        pair = self.cfg.select_pair()
        if not any(self._source_ok):
            raise SourceError(
                "no camera could be opened: " + "; ".join(failures)
            )
        for idx in pair:
            if not self._source_ok[idx]:
                raise SourceError(
                    f"stereo pair camera {idx} could not be opened: "
                    + "; ".join(failures)
                )
        if failures:
            with self._lock:
                self.reasons = failures

    def _load_calibration(self) -> None:
        self._rectify_maps = None
        self._calibration_warnings = []
        directory = self.cfg.calibration_dir
        if not directory:
            return
        if not os.path.isdir(directory):
            self._calibration_warnings.append(
                f"calibration_dir {directory!r} does not exist; "
                "running without rectification"
            )
            return
        try:
            _, maps, warnings = calibrate_mod.load_calibration(directory)
            self._rectify_maps = maps
            self._calibration_warnings = warnings
        except (OSError, KeyError, ValueError) as exc:
            self._calibration_warnings.append(
                f"calibration could not be loaded ({exc}); "
                "running without rectification"
            )

    def _start_ffmpeg(self) -> None:
        os.makedirs(self.cfg.stream.hls_dir, exist_ok=True)
        assert self.encoder_choice is not None
        cmd = build_command(self.cfg, self.encoder_choice)
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # -- capture threads ----------------------------------------------------

    def _reader(self, index: int) -> None:
        src = self._sources[index]
        while not self._stop.is_set():
            item = src.read()
            if item is None:
                time.sleep(0.002)
                continue
            ts, frame = item
            self._histories[index].append((ts, frame))

    # -- main loop ----------------------------------------------------------

    def run_forever(self) -> None:
        """Blocking compose/encode loop; call from the main thread or a
        dedicated thread after start()."""
        cfg = self.cfg
        pair = cfg.select_pair()
        ref = cfg.reference_index()
        syncer = FrameSynchronizer(
            len(cfg.cameras),
            tolerance_ms=cfg.sync.tolerance_ms,
            max_hold_frames=cfg.sync.max_hold_frames,
        )
        composer = StereoComposer(
            cfg.output.width, cfg.output.height,
            left_offset=(cfg.alignment.left_x, cfg.alignment.left_y),
            right_offset=(cfg.alignment.right_x, cfg.alignment.right_y),
            rectify_maps=getattr(self, "_rectify_maps", None),
        )
        for i in range(len(cfg.cameras)):
            if not self._source_ok[i]:
                continue  # failed sources stay silent; sync reports them
            thread = threading.Thread(target=self._reader, args=(i,),
                                      daemon=True, name=f"reader-{i}")
            thread.start()
            self._threads.append(thread)

        period = 1.0 / cfg.output.fps
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.005))
                continue
            next_tick = max(next_tick + period, now)
            histories = [list(h) for h in self._histories]
            if not all(histories[i] for i in pair):
                continue  # still warming up
            reference_ts = self._reference_ts(histories, ref, pair)
            result = syncer.synchronize(histories, reference_ts)
            # A missing non-pair input must not stall the stream: the
            # composer decides whether both eyes can be filled (with the
            # center camera as fallback) and reports why when it cannot.
            frame, notes = composer.compose_with_fallback(
                result.frames, pair, ref
            )
            if result.reasons or notes:
                self.stats["sync_events"] += 1
                with self._lock:
                    self.reasons = (
                        [self.encoder_choice.reason]
                        + self._calibration_warnings
                        + result.reasons + notes
                    )
                    if self.state == "running" and (result.degraded or notes):
                        self.state = "degraded"
                    elif self.state == "degraded" and not result.degraded and not notes:
                        self.state = "running"
            if frame is None:
                continue
            self.stats["frames_composed"] += 1
            self._latest_preview = frame
            if self._proc is not None and self._proc.stdin is not None:
                try:
                    self._proc.stdin.write(frame.tobytes())
                    self.stats["frames_written"] += 1
                except (BrokenPipeError, OSError) as exc:
                    self._fail(f"encoder pipe broke: {exc}")
                    return
            if self._proc is not None and self._proc.poll() is not None:
                self._fail(
                    f"ffmpeg exited unexpectedly with code "
                    f"{self._proc.returncode}"
                )
                return

    @staticmethod
    def _reference_ts(histories, ref: int, pair: tuple[int, int]) -> float:
        """Sync reference: the center camera's newest frame when it is not
        part of the pair, else the newest frame across the pair."""
        if ref not in pair and histories[ref]:
            return histories[ref][-1][0]
        candidates = [histories[i][-1][0] for i in pair if histories[i]]
        return max(candidates)

    # -- introspection ------------------------------------------------------

    def preview_jpeg(self) -> bytes | None:
        frame = self._latest_preview
        if frame is None:
            return None
        import cv2

        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "reasons": list(self.reasons),
                "encoder": (
                    {
                        "codec": self.encoder_choice.codec,
                        "hardware": self.encoder_choice.hardware,
                        "reason": self.encoder_choice.reason,
                    }
                    if self.encoder_choice
                    else None
                ),
                "stats": dict(self.stats),
                "cameras": [
                    {"index": i, "type": c.type,
                     "label": c.label or f"Camera {i}"}
                    for i, c in enumerate(self.cfg.cameras)
                ],
                "pair": list(self.cfg.select_pair()),
                "reference": self.cfg.reference_index(),
                "uptime_seconds": (
                    round(time.monotonic() - self._started_at, 1)
                    if self._started_at and self.state in ("running", "degraded")
                    else 0
                ),
            }

"""Configuration definition and validation.

Validation never raises on the first problem: it returns the full list of
human-readable reasons so the UI / CLI can show *why* a config is rejected
instead of failing silently or with a bare traceback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CameraConfig:
    type: str = "device"  # "device" | "virtual"
    index: int = 0  # device index (type == "device")
    label: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 30
    # Virtual cameras get a hue shift so left/right are distinguishable.
    hue: int = 0


@dataclass
class OutputConfig:
    width: int = 2560
    height: int = 720
    fps: int = 30
    bitrate_kbps: int = 12000


@dataclass
class SyncConfig:
    tolerance_ms: float = 50.0
    max_hold_frames: int = 5


@dataclass
class EncoderConfig:
    preference: str = "auto"  # "auto" | "nvenc" | "x264"
    preset: str = "veryfast"  # libx264 preset (ignored by nvenc except mapping)


@dataclass
class StreamConfig:
    hls_dir: str = "hls"
    hls_segment_seconds: int = 2
    hls_window: int = 6  # segments kept in the playlist
    rtsp_url: str = ""  # e.g. rtsp://localhost:8554/fukugan
    srt_url: str = ""  # e.g. srt://localhost:8890?streamid=publish:fukugan


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass
class AlignmentConfig:
    # Manual per-eye pixel offsets applied after rectification.
    left_x: int = 0
    left_y: int = 0
    right_x: int = 0
    right_y: int = 0


@dataclass
class Config:
    cameras: list[CameraConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    # Explicit stereo pair as camera indices; empty = auto (outermost pair).
    pair: list[int] = field(default_factory=list)
    calibration_dir: str = ""  # directory written by fukugan.calibrate

    # -- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        for cam in data.get("cameras", []):
            cfg.cameras.append(_filtered(CameraConfig, cam))
        for section, klass in (
            ("output", OutputConfig),
            ("sync", SyncConfig),
            ("encoder", EncoderConfig),
            ("stream", StreamConfig),
            ("server", ServerConfig),
            ("alignment", AlignmentConfig),
        ):
            if section in data:
                setattr(cfg, section, _filtered(klass, data[section]))
        cfg.pair = list(data.get("pair", []))
        cfg.calibration_dir = str(data.get("calibration_dir", ""))
        return cfg

    @classmethod
    def from_json_file(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def default_virtual(cls) -> "Config":
        """Two virtual cameras: the no-hardware demo / verification setup."""
        cfg = cls()
        cfg.cameras = [
            CameraConfig(type="virtual", label="Virtual L", hue=0),
            CameraConfig(type="virtual", label="Virtual R", hue=40),
        ]
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- stereo pair selection --------------------------------------------

    def select_pair(self) -> tuple[int, int]:
        """Return (left_index, right_index) of the stereo pair.

        Explicit config wins. Otherwise the outermost two cameras form the
        pair (widest baseline); see docs/DESIGN.md section 6.
        """
        if len(self.pair) == 2:
            return int(self.pair[0]), int(self.pair[1])
        n = len(self.cameras)
        return 0, n - 1

    def reference_index(self) -> int:
        """The center camera: sync reference and fallback eye."""
        return len(self.cameras) // 2

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []
        n = len(self.cameras)
        if n < 2:
            errors.append(
                f"at least 2 cameras are required for stereo capture, got {n}"
            )
        for i, cam in enumerate(self.cameras):
            if cam.type not in ("device", "virtual"):
                errors.append(
                    f"camera {i}: unknown type {cam.type!r} "
                    "(expected 'device' or 'virtual')"
                )
            if cam.type == "device" and cam.index < 0:
                errors.append(f"camera {i}: device index must be >= 0")
            if cam.width <= 0 or cam.height <= 0:
                errors.append(
                    f"camera {i}: width/height must be positive, "
                    f"got {cam.width}x{cam.height}"
                )
            if not (1 <= cam.fps <= 240):
                errors.append(f"camera {i}: fps out of range: {cam.fps}")
        if self.pair:
            if len(self.pair) != 2:
                errors.append("pair must contain exactly 2 camera indices")
            else:
                for idx in self.pair:
                    if not (0 <= idx < n):
                        errors.append(
                            f"pair index {idx} out of range (cameras: {n})"
                        )
                if self.pair[0] == self.pair[1]:
                    errors.append("pair indices must differ")
                elif self.pair[0] > self.pair[1]:
                    errors.append(
                        "pair must be ordered (left index < right index)"
                    )
        out = self.output
        if out.width <= 0 or out.height <= 0:
            errors.append(
                f"output size must be positive, got {out.width}x{out.height}"
            )
        if out.width % 2 != 0:
            errors.append(
                "output width must be even (two eyes side by side)"
            )
        if not (1 <= out.fps <= 240):
            errors.append(f"output fps out of range: {out.fps}")
        if out.bitrate_kbps < 100:
            errors.append(
                f"output bitrate too low: {out.bitrate_kbps} kbps (min 100)"
            )
        if self.sync.tolerance_ms <= 0:
            errors.append("sync tolerance_ms must be positive")
        if self.sync.max_hold_frames < 0:
            errors.append("sync max_hold_frames must be >= 0")
        if self.encoder.preference not in ("auto", "nvenc", "x264"):
            errors.append(
                f"encoder preference {self.encoder.preference!r} is unknown "
                "(expected 'auto', 'nvenc' or 'x264')"
            )
        if not (1 <= self.server.port <= 65535):
            errors.append(f"server port out of range: {self.server.port}")
        if self.stream.hls_segment_seconds < 1:
            errors.append("hls_segment_seconds must be >= 1")
        if self.stream.hls_window < 2:
            errors.append("hls_window must be >= 2")
        for name, url in (
            ("rtsp_url", self.stream.rtsp_url),
            ("srt_url", self.stream.srt_url),
        ):
            if url and "://" not in url:
                errors.append(f"stream {name} is not a URL: {url!r}")
        return errors


def _filtered(klass, data: dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys."""
    allowed = {f.name for f in klass.__dataclass_fields__.values()}
    return klass(**{k: v for k, v in data.items() if k in allowed})

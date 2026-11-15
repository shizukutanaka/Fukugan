"""FFmpeg encoding: command construction and hardware-encoder probing.

Hardware encoding policy: *probe by doing*, not by listing. An encoder
can appear in `ffmpeg -encoders` yet fail at runtime (no GPU, driver
mismatch). At startup we encode one real frame with NVENC; only success
earns its use. Failure is recorded with its reason and we fall back to
libx264 -- the stream must never silently not start.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from .config import Config

X264_PRESETS = {
    "ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
    "slow", "slower", "veryslow",
}
# NVENC preset names differ; map the x264-style intent across.
NVENC_PRESET_MAP = {
    "ultrafast": "p1", "superfast": "p2", "veryfast": "p3", "faster": "p3",
    "fast": "p4", "medium": "p4", "slow": "p5", "slower": "p6",
    "veryslow": "p7",
}


@dataclass
class EncoderChoice:
    codec: str  # ffmpeg encoder name, e.g. "h264_nvenc" or "libx264"
    hardware: bool
    reason: str  # why this encoder was chosen (always stated, never silent)


def probe_nvenc(ffmpeg: str = "ffmpeg", timeout: float = 20.0) -> tuple[bool, str]:
    """Try to encode one black frame with NVENC. Returns (ok, detail)."""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:size=320x240:rate=1",
        "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except FileNotFoundError:
        return False, "ffmpeg executable not found"
    except subprocess.TimeoutExpired:
        return False, "NVENC probe timed out"
    if proc.returncode == 0:
        return True, "NVENC probe encode succeeded"
    lines = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
    # The most informative line is usually the first error, not the
    # trailing "nothing was written" summary.
    detail = next(
        (ln for ln in lines
         if any(k in ln.lower() for k in ("error", "failed", "cannot", "invalid"))),
        lines[-1] if lines else f"exit code {proc.returncode}",
    )
    return False, f"NVENC probe failed: {detail}"


def select_encoder(preference: str, preset: str = "veryfast",
                   ffmpeg: str = "ffmpeg") -> EncoderChoice:
    if preference == "x264":
        return EncoderChoice("libx264", False,
                             "libx264 selected by configuration")
    if preference == "nvenc":
        ok, detail = probe_nvenc(ffmpeg)
        if ok:
            return EncoderChoice("h264_nvenc", True,
                                 "NVENC requested and probe succeeded")
        return EncoderChoice(
            "libx264", False,
            f"NVENC was requested but is unusable ({detail}); "
            "fell back to libx264",
        )
    # auto
    ok, detail = probe_nvenc(ffmpeg)
    if ok:
        return EncoderChoice("h264_nvenc", True,
                             "NVENC probe succeeded; using hardware encoding")
    return EncoderChoice(
        "libx264", False,
        f"NVENC unavailable ({detail}); using libx264 software encoding",
    )


def encoder_args(choice: EncoderChoice, cfg: Config) -> list[str]:
    bitrate = f"{cfg.output.bitrate_kbps}k"
    if choice.codec == "h264_nvenc":
        preset = NVENC_PRESET_MAP.get(cfg.encoder.preset, "p3")
        return [
            "-c:v", "h264_nvenc", "-preset", preset,
            "-tune", "ll", "-rc", "vbr",
            "-b:v", bitrate, "-maxrate", bitrate,
            "-pix_fmt", "yuv420p",
        ]
    preset = cfg.encoder.preset if cfg.encoder.preset in X264_PRESETS else "veryfast"
    return [
        "-c:v", "libx264", "-preset", preset,
        "-tune", "zerolatency",
        "-b:v", bitrate, "-maxrate", bitrate,
        "-bufsize", f"{cfg.output.bitrate_kbps * 2}k",
        "-pix_fmt", "yuv420p",
    ]


def _hls_options(cfg: Config) -> list[tuple[str, str]]:
    return [
        ("hls_time", str(cfg.stream.hls_segment_seconds)),
        ("hls_list_size", str(cfg.stream.hls_window)),
        ("hls_flags", "delete_segments+append_list"),
        ("hls_segment_filename",
         os.path.join(cfg.stream.hls_dir, "seg%05d.ts")),
    ]


def _tee_target(fmt: str, options: list[tuple[str, str]], dest: str) -> str:
    opts = ":".join(f"{k}={v}" for k, v in options)
    return f"[f={fmt}" + (f":{opts}" if opts else "") + f"]{dest}"


def build_command(cfg: Config, choice: EncoderChoice,
                  ffmpeg: str = "ffmpeg") -> list[str]:
    """Build the full FFmpeg command: rawvideo in on stdin, HLS (+push) out.

    With a single output the muxer is used directly; with several, the
    tee muxer fans the one encoded stream out to every target so the
    video is encoded exactly once.
    """
    out = cfg.output
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out.width}x{out.height}", "-r", str(out.fps),
        "-i", "-",
        # Explicit mapping: the tee muxer (used when pushing) performs
        # no automatic stream selection on modern FFmpeg.
        "-map", "0:v:0",
    ]
    cmd += encoder_args(choice, cfg)
    cmd += ["-r", str(out.fps),
            "-g", str(out.fps * cfg.stream.hls_segment_seconds)]

    hls_path = os.path.join(cfg.stream.hls_dir, "index.m3u8")
    # Push targets get onfail=ignore: an unreachable MediaMTX must never
    # take the local HLS output down with it.
    pushes: list[tuple[str, list[tuple[str, str]], str]] = []
    if cfg.stream.rtsp_url:
        pushes.append(("rtsp", [("rtsp_transport", "tcp"),
                                ("onfail", "ignore")],
                       cfg.stream.rtsp_url))
    if cfg.stream.srt_url:
        pushes.append(("mpegts", [("onfail", "ignore")],
                       cfg.stream.srt_url))

    if not pushes:
        for key, value in _hls_options(cfg):
            cmd += [f"-{key}", value]
        cmd += ["-f", "hls", hls_path]
    else:
        targets = [_tee_target("hls", _hls_options(cfg), hls_path)]
        targets += [_tee_target(fmt, opts, dest) for fmt, opts, dest in pushes]
        cmd += ["-f", "tee", "|".join(targets)]
    return cmd

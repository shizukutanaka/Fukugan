"""End-to-end verification with virtual cameras.

Full path, no hardware: virtual capture -> sync -> side-by-side compose
-> FFmpeg encode -> HLS files -> HTTP fetch -> decode -> read the frame
counter barcode back out of the decoded video and prove it advances.
"""

import json
import subprocess
import threading
import time
import urllib.request

import numpy as np
import pytest

from fukugan.config import CameraConfig, Config
from fukugan.server import AppServer
from fukugan.sources import read_frame_counter

PORT = 18099
BASE = f"http://127.0.0.1:{PORT}"
OUT_W, OUT_H = 640, 240


def fetch(path: str, timeout: float = 5.0) -> bytes:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return resp.read()


def decode_middle_frame(segment: bytes) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        input=segment, capture_output=True, timeout=30,
    )
    frame_bytes = OUT_W * OUT_H * 3
    total = len(proc.stdout) // frame_bytes
    assert total >= 1, "no frames decoded from HLS segment"
    mid = total // 2
    raw = proc.stdout[mid * frame_bytes:(mid + 1) * frame_bytes]
    return np.frombuffer(raw, dtype=np.uint8).reshape(OUT_H, OUT_W, 3)


def left_eye_counter(segment: bytes) -> int:
    frame = decode_middle_frame(segment)
    # The left eye occupies the left half; its barcode strip is top-left.
    counter = read_frame_counter(frame[:, : OUT_W // 2])
    assert counter is not None, "frame counter barcode not readable"
    return counter


def wait_for_playlist(deadline_s: float = 30.0) -> str:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            text = fetch("/hls/index.m3u8").decode()
            if ".ts" in text:
                return text
        except Exception:
            pass
        time.sleep(0.5)
    raise AssertionError("HLS playlist never appeared")


def latest_segment_name(playlist: str) -> str:
    names = [ln.strip() for ln in playlist.splitlines()
             if ln.strip().endswith(".ts")]
    assert names, "playlist has no segments"
    return names[-1]


@pytest.fixture(scope="module")
def running_server(tmp_path_factory):
    cfg = Config()
    cfg.cameras = [
        CameraConfig(type="virtual", label="Virtual L", width=320,
                     height=240, fps=30, hue=0),
        CameraConfig(type="virtual", label="Virtual R", width=320,
                     height=240, fps=30, hue=40),
    ]
    cfg.output.width, cfg.output.height = OUT_W, OUT_H
    cfg.output.fps = 30
    cfg.output.bitrate_kbps = 8000
    cfg.stream.hls_dir = str(tmp_path_factory.mktemp("hls"))
    cfg.stream.hls_segment_seconds = 1
    cfg.stream.hls_window = 6
    cfg.server.host, cfg.server.port = "127.0.0.1", PORT
    assert cfg.validate() == []

    server = AppServer(cfg)
    ok, message = server.pipeline.start()
    assert ok, message
    server.start_loop()
    http_thread = threading.Thread(target=server._httpd.serve_forever,
                                   daemon=True)
    http_thread.start()
    yield server
    server.pipeline.stop()
    server._httpd.shutdown()


def test_status_reports_running_with_encoder_reason(running_server):
    # The pipeline is asynchronous: wait until frames actually flow.
    deadline = time.monotonic() + 20
    status = json.loads(fetch("/api/status"))
    while status["stats"]["frames_written"] == 0:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"no frames written within 20 s; status={status}"
            )
        time.sleep(0.25)
        status = json.loads(fetch("/api/status"))
    assert status["state"] in ("running", "degraded")
    assert status["encoder"] is not None
    assert status["encoder"]["codec"] in ("libx264", "h264_nvenc")
    # The encoder choice always carries its reason.
    assert status["encoder"]["reason"]
    assert status["stats"]["frames_written"] > 0


def test_ui_is_served(running_server):
    body = fetch("/").decode("utf-8")
    assert "Fukugan" in body
    assert 'id="startBtn"' in body


def test_hls_stream_frames_advance(running_server):
    playlist = wait_for_playlist()
    seg1 = latest_segment_name(playlist)
    counter1 = left_eye_counter(fetch(f"/hls/{seg1}"))

    # Wait until a newer segment is published, then sample again.
    deadline = time.monotonic() + 30
    counter2 = counter1
    while time.monotonic() < deadline:
        playlist = fetch("/hls/index.m3u8").decode()
        seg2 = latest_segment_name(playlist)
        if seg2 != seg1:
            counter2 = left_eye_counter(fetch(f"/hls/{seg2}"))
            if counter2 > counter1:
                break
        time.sleep(0.5)
    assert counter2 > counter1, (
        f"frame counter did not advance through the HLS stream: "
        f"{counter1} -> {counter2}"
    )
    # At 30 fps the counter should have moved substantially, not by 1.
    assert counter2 - counter1 >= 5

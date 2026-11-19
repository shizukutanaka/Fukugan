"""Built-in HTTP server: control UI, HLS delivery, preview, JSON API.

Endpoints:
    /                the single-file control UI
    /hls/...         HLS playlist and segments written by FFmpeg
    /preview.mjpg    MJPEG preview of the composed frame
    /api/status      pipeline status JSON (state + reasons, always honest)
                     plus `headset_urls`: HLS URLs reachable from the LAN
    /api/cameras     configured cameras + detected device indices
    /api/start       POST: start the pipeline
    /api/stop        POST: stop the pipeline
    /api/alignment   POST: update manual per-eye offsets (JSON body)

There is deliberately no authentication: the server binds to 127.0.0.1
by default, and binding it to a LAN address is an explicit user choice
that the README and UI both warn about.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

from .config import Config
from .pipeline import Pipeline

WEBUI_PATH = os.path.join(os.path.dirname(__file__), "webui", "index.html")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
WILDCARD_HOSTS = ("0.0.0.0", "")


def lan_address() -> str | None:
    """The IPv4 address this machine uses on its LAN, or None.

    A UDP connect only selects a route; no packet leaves the machine.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("10.254.254.254", 1))
            address = sock.getsockname()[0]
        except OSError:
            return None
    return None if address.startswith("127.") else address


def headset_urls(host: str, port: int) -> list[str]:
    """HLS URLs a headset on the same network can open.

    Empty when the server is bound to loopback: nothing off this machine
    can reach it, so showing a 127.0.0.1 URL for a headset would mislead.
    """
    if host in LOOPBACK_HOSTS:
        return []
    if host in WILDCARD_HOSTS:
        address = lan_address()
        hosts = [address] if address else []
    else:
        hosts = [host]
    return [f"http://{h}:{port}/hls/index.m3u8" for h in hosts]


def list_device_cameras(max_index: int = 10) -> list[int]:
    """Best-effort probe of present device indices (for the UI picker)."""
    found: list[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    return found


class _Handler(BaseHTTPRequestHandler):
    server_version = "Fukugan/0.1"
    # Quieten default logging to stderr but keep errors visible.
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
        if "error" in fmt.lower():
            super().log_message(fmt, *args)

    # -- helpers ----------------------------------------------------------

    @property
    def app(self) -> "AppServer":
        return self.server.app  # type: ignore[attr-defined]

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # -- GET ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            try:
                with open(WEBUI_PATH, "rb") as fh:
                    body = fh.read()
            except OSError:
                self._json(500, {"error": "web UI file is missing"})
                return
            self._send(200, body, "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(200, {**self.app.pipeline.status(),
                             "headset_urls": self.app.headset_urls()})
        elif path == "/api/cameras":
            cfg = self.app.pipeline.cfg
            self._json(200, {
                "configured": [
                    {"index": i, "type": c.type, "label": c.label,
                     "device_index": c.index}
                    for i, c in enumerate(cfg.cameras)
                ],
                "detected_devices": list_device_cameras(),
            })
        elif path == "/preview.mjpg":
            self._preview()
        elif path.startswith("/hls/"):
            self._serve_hls(path[len("/hls/"):])
        else:
            self._json(404, {"error": f"not found: {path}"})

    def _serve_hls(self, name: str) -> None:
        # Path traversal guard: only plain file names inside the HLS dir.
        if os.path.basename(name) != name or name.startswith("."):
            self._json(403, {"error": "forbidden"})
            return
        full = os.path.join(self.app.pipeline.cfg.stream.hls_dir, name)
        if not os.path.isfile(full):
            self._json(404, {"error": "stream not available yet"})
            return
        content_type = (
            "application/vnd.apple.mpegurl" if name.endswith(".m3u8")
            else "video/mp2t"
        )
        with open(full, "rb") as fh:
            body = fh.read()
        self._send(200, body, content_type)

    def _preview(self) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                jpeg = self.app.pipeline.preview_jpeg()
                if jpeg is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg + b"\r\n")
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass  # viewer went away; that is normal

    # -- POST -----------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        path = urlparse(self.path).path
        pipeline = self.app.pipeline
        if path == "/api/start":
            ok, message = pipeline.start()
            if ok:
                self.app.start_loop()
            self._json(200 if ok else 409,
                       {"ok": ok, "message": message,
                        "status": pipeline.status()})
        elif path == "/api/stop":
            pipeline.stop()
            self.app.stop_loop()
            self._json(200, {"ok": True, "status": pipeline.status()})
        elif path == "/api/alignment":
            body = self._read_json_body()
            align = pipeline.cfg.alignment
            for key in ("left_x", "left_y", "right_x", "right_y"):
                if key in body:
                    try:
                        setattr(align, key, int(body[key]))
                    except (TypeError, ValueError):
                        self._json(400, {"error": f"{key} must be an integer"})
                        return
            self._json(200, {
                "ok": True,
                "message": "alignment updated; restart the stream to apply",
                "alignment": {
                    "left_x": align.left_x, "left_y": align.left_y,
                    "right_x": align.right_x, "right_y": align.right_y,
                },
            })
        else:
            self._json(404, {"error": f"not found: {path}"})


class AppServer:
    """Owns the HTTP server and the pipeline's compose loop thread."""

    def __init__(self, cfg: Config):
        self.pipeline = Pipeline(cfg)
        self._httpd = ThreadingHTTPServer((cfg.server.host, cfg.server.port),
                                          _Handler)
        self._httpd.app = self  # type: ignore[attr-defined]
        self._loop_thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        return self._httpd.server_address[:2]

    def headset_urls(self) -> list[str]:
        host, port = self.address
        return headset_urls(host, port)

    def start_loop(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(
            target=self.pipeline.run_forever, daemon=True, name="pipeline"
        )
        self._loop_thread.start()

    def stop_loop(self) -> None:
        if self._loop_thread:
            self._loop_thread.join(timeout=3.0)
            self._loop_thread = None

    def serve_forever(self) -> None:
        host, port = self.address
        print(f"Fukugan UI: http://{host}:{port}/")
        urls = self.headset_urls()
        for url in urls:
            print(f"Open on the headset: {url}")
        if not urls:
            print(
                "Headset viewing: this address is not reachable from other "
                "devices; restart with --host 0.0.0.0 to stream to a headset "
                "on the same network."
            )
        if host not in LOOPBACK_HOSTS:
            print(
                "WARNING: bound to a non-local address with no "
                "authentication; anyone on this network can view the "
                "stream and control the app."
            )
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.stop()
            self._httpd.shutdown()

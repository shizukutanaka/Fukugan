"""Command line entry point.

Usage:
    python -m fukugan [--config config.json] [--virtual] [--no-autostart]

Without --config, a two-device-camera default is used; --virtual selects
the built-in virtual cameras so the whole pipeline can be tried (and
verified) on a machine with no cameras attached.
"""

from __future__ import annotations

import argparse
import json
import sys

from .calibrate import calibrate_pair, save_calibration
from .config import Config
from .server import AppServer


def build_config(args: argparse.Namespace) -> Config:
    if args.config:
        cfg = Config.from_json_file(args.config)
    elif args.virtual:
        cfg = Config.default_virtual()
    else:
        cfg = Config()
        from .config import CameraConfig

        cfg.cameras = [
            CameraConfig(type="device", index=0, label="Left"),
            CameraConfig(type="device", index=1, label="Right"),
        ]
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fukugan", description=__doc__)
    parser.add_argument("--config", help="path to a JSON config file")
    parser.add_argument("--virtual", action="store_true",
                        help="use virtual cameras (no hardware needed)")
    parser.add_argument("--host", help="override bind host")
    parser.add_argument("--port", type=int, help="override bind port")
    parser.add_argument("--no-autostart", action="store_true",
                        help="do not start streaming on launch")
    sub = parser.add_subparsers(dest="command")
    cal = sub.add_parser("calibrate",
                         help="run stereo calibration from image pairs")
    cal.add_argument("--left", nargs="+", required=True,
                     help="left-eye image files")
    cal.add_argument("--right", nargs="+", required=True,
                     help="right-eye image files")
    cal.add_argument("--out", required=True, help="output directory")
    cal.add_argument("--cols", type=int, default=9)
    cal.add_argument("--rows", type=int, default=6)
    cal.add_argument("--square-mm", type=float, default=25.0)
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        return _run_calibrate(args)

    cfg = build_config(args)
    errors = cfg.validate()
    if errors:
        for err in errors:
            print(f"config error: {err}", file=sys.stderr)
        return 2
    server = AppServer(cfg)
    if not args.no_autostart:
        ok, message = server.pipeline.start()
        print(f"pipeline: {message}")
        if ok:
            server.start_loop()
    server.serve_forever()
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    import cv2

    left = [cv2.imread(p) for p in args.left]
    right = [cv2.imread(p) for p in args.right]
    bad = [p for p, img in zip(args.left + args.right, left + right)
           if img is None]
    if bad:
        print(f"could not read images: {bad}", file=sys.stderr)
        return 2
    try:
        result, notes = calibrate_pair(
            left, right,
            pattern_size=(args.cols, args.rows),
            square_mm=args.square_mm,
        )
    except ValueError as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1
    for note in notes:
        print(note)
    save_calibration(result, args.out)
    print(f"saved to {args.out} (baseline {result.baseline_mm:.1f} mm)")
    print(json.dumps({"stereo_error_px": result.stereo_error,
                      "quality_ok": result.quality_ok}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

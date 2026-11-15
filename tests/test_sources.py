import numpy as np

from fukugan.config import CameraConfig
from fukugan.sources import (
    VirtualCameraSource,
    draw_frame_counter,
    read_frame_counter,
    render_test_pattern,
)


def test_counter_barcode_round_trip():
    for value in (0, 1, 42, 65535, 2**24 - 1, 1234567):
        frame = np.zeros((100, 400, 3), dtype=np.uint8)
        draw_frame_counter(frame, value)
        assert read_frame_counter(frame) == value


def test_counter_absent_on_blank_frame():
    frame = np.zeros((100, 400, 3), dtype=np.uint8)
    assert read_frame_counter(frame) is None


def test_test_pattern_is_deterministic_and_changes():
    a = render_test_pattern(320, 240, 10, hue=0, parallax=24)
    b = render_test_pattern(320, 240, 10, hue=0, parallax=24)
    c = render_test_pattern(320, 240, 11, hue=0, parallax=24)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert read_frame_counter(a) == 10
    assert read_frame_counter(c) == 11


def test_parallax_shifts_scene_between_cameras():
    left = render_test_pattern(320, 240, 5, hue=0, parallax=10)
    right = render_test_pattern(320, 240, 5, hue=0, parallax=60)
    assert not np.array_equal(left, right)


def test_virtual_source_produces_frames_at_rate():
    cfg = CameraConfig(type="virtual", width=160, height=120, fps=240)
    src = VirtualCameraSource(cfg)
    src.open()
    got = 0
    import time

    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if src.read() is not None:
            got += 1
    src.close()
    assert got >= 10
    assert src.counter == got

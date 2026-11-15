import numpy as np
import pytest

from fukugan.compose import StereoComposer, translate


def solid(value: int, w: int = 100, h: int = 50) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_side_by_side_layout():
    comp = StereoComposer(200, 50)
    out = comp.compose(solid(80), solid(160))
    assert out.shape == (50, 200, 3)
    assert out[0, 0, 0] == 80
    assert out[0, 199, 0] == 160


def test_resize_into_eye():
    comp = StereoComposer(200, 100)
    out = comp.compose(solid(10, 640, 480), solid(20, 320, 240))
    assert out.shape == (100, 200, 3)


def test_odd_output_width_rejected():
    with pytest.raises(ValueError):
        StereoComposer(201, 50)


def test_translate_shifts_and_fills_black():
    img = solid(255, 10, 10)
    moved = translate(img, 3, 0)
    assert moved[0, 0, 0] == 0
    assert moved[0, 3, 0] == 255
    assert translate(img, 0, 0) is img


def test_manual_offsets_applied_per_eye():
    left = np.zeros((50, 100, 3), dtype=np.uint8)
    left[25, 50] = (255, 255, 255)
    right = np.zeros((50, 100, 3), dtype=np.uint8)
    right[25, 50] = (255, 255, 255)
    comp = StereoComposer(200, 50, left_offset=(5, -2), right_offset=(-5, 2))
    out = comp.compose(left, right)
    # Left white pixel moved +5 x, -2 y within the left half.
    assert out[23, 55, 0] == 255
    # Right white pixel moved -5 x, +2 y within the right half.
    assert out[27, 100 + 45, 0] == 255


def test_fallback_substitutes_center_camera():
    comp = StereoComposer(200, 50)
    frames = [solid(30), solid(90), None]  # pair (0, 2), reference 1
    out, reasons = comp.compose_with_fallback(frames, (0, 2), 1)
    assert out is not None
    assert out[0, 199, 0] == 90  # right eye filled from center camera
    assert any("right camera" in r for r in reasons)


def test_fallback_fails_when_no_substitute():
    comp = StereoComposer(200, 50)
    frames = [solid(30), None, None]
    out, reasons = comp.compose_with_fallback(frames, (0, 2), 1)
    assert out is None
    assert reasons

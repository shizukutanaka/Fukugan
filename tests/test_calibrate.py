"""Calibration tests use synthetic checkerboard views: a planar board
texture warped by known homographies. No real camera is involved, but
the full OpenCV detection + calibration path runs for real."""

import cv2
import numpy as np
import pytest

from fukugan.calibrate import (
    board_object_points,
    calibrate_pair,
    find_board,
    load_calibration,
    rectification_maps,
    save_calibration,
)

PATTERN = (9, 6)  # inner corners
SQUARE_PX = 40


def board_texture() -> np.ndarray:
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    tex = np.zeros((rows * SQUARE_PX, cols * SQUARE_PX), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                tex[r * SQUARE_PX:(r + 1) * SQUARE_PX,
                    c * SQUARE_PX:(c + 1) * SQUARE_PX] = 255
    return tex


def render_view(tex: np.ndarray, quad: np.ndarray,
                size=(640, 480)) -> np.ndarray:
    h, w = tex.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    canvas = np.full((size[1], size[0]), 200, dtype=np.uint8)
    warped = cv2.warpPerspective(tex, m, size, dst=canvas,
                                 borderMode=cv2.BORDER_TRANSPARENT)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def make_pairs(n: int = 6, shift: int = 28):
    tex = board_texture()
    rng = np.random.default_rng(7)
    left_views, right_views = [], []
    for i in range(n):
        cx, cy = 320 + rng.integers(-30, 30), 240 + rng.integers(-20, 20)
        bw, bh = 340 + rng.integers(-30, 30), 235 + rng.integers(-20, 20)
        tilt = rng.integers(-25, 25)
        quad = np.float32([
            [cx - bw / 2, cy - bh / 2 + tilt],
            [cx + bw / 2, cy - bh / 2 - tilt],
            [cx + bw / 2 - tilt, cy + bh / 2],
            [cx - bw / 2 + tilt, cy + bh / 2],
        ])
        left_views.append(render_view(tex, quad))
        right_views.append(render_view(tex, quad + np.float32([shift, 3])))
    return left_views, right_views


def test_board_object_points_shape():
    objp = board_object_points(9, 6, 25.0)
    assert objp.shape == (54, 3)
    assert objp[1, 0] == 25.0
    assert objp[0, 2] == 0.0


def test_find_board_on_rendered_view():
    tex = board_texture()
    quad = np.float32([[120, 90], [520, 90], [520, 330], [120, 330]])
    view = render_view(tex, quad)
    found, corners = find_board(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY),
                                PATTERN)
    assert found
    assert corners.reshape(-1, 2).shape == (54, 2)


def test_find_board_rejects_blank_image():
    blank = np.full((480, 640), 128, dtype=np.uint8)
    found, _ = find_board(blank, PATTERN)
    assert not found


def test_calibrate_pair_end_to_end(tmp_path):
    left, right = make_pairs()
    result, notes = calibrate_pair(left, right, pattern_size=PATTERN,
                                   square_mm=25.0)
    assert np.isfinite(result.stereo_error)
    assert result.stereo_error < 5.0  # generous bound for synthetic views
    assert result.baseline_mm > 0
    assert result.left_camera_matrix.shape == (3, 3)
    assert any("pairs" in n for n in notes)

    (lmapx, lmapy), (rmapx, rmapy) = rectification_maps(result)
    assert lmapx.shape == (480, 640)
    assert rmapy.shape == (480, 640)

    save_calibration(result, str(tmp_path))
    loaded, maps, warnings = load_calibration(str(tmp_path))
    assert loaded.image_size == result.image_size
    assert np.allclose(loaded.translation, result.translation)
    assert maps[0][0].shape == (480, 640)


def test_calibrate_pair_requires_enough_pairs():
    left, right = make_pairs(n=2)
    with pytest.raises(ValueError, match="at least 5"):
        calibrate_pair(left, right, pattern_size=PATTERN)


def test_calibrate_pair_rejects_mismatched_counts():
    left, right = make_pairs(n=3)
    with pytest.raises(ValueError, match="differ"):
        calibrate_pair(left, right[:2], pattern_size=PATTERN)

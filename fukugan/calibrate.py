"""Checkerboard stereo calibration.

Workflow: collect synchronized image pairs that show the board in both
cameras -> calibrate each camera -> stereoCalibrate -> stereoRectify.
Results persist as a directory:

    calibration.json   human-readable parameters + reprojection errors
    maps.npz           rectification maps (numeric arrays)

A calibration whose stereo reprojection error exceeds the quality
threshold is still loadable, but the loader reports it as low quality so
the application can warn instead of silently trusting it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

QUALITY_ERROR_THRESHOLD_PX = 1.0


@dataclass
class CalibrationResult:
    image_size: tuple[int, int]
    left_camera_matrix: np.ndarray
    left_distortion: np.ndarray
    right_camera_matrix: np.ndarray
    right_distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    left_error: float
    right_error: float
    stereo_error: float

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.translation))

    @property
    def quality_ok(self) -> bool:
        return self.stereo_error <= QUALITY_ERROR_THRESHOLD_PX


def board_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_mm
    return objp


def find_board(gray: np.ndarray, pattern_size: tuple[int, int]):
    """Locate checkerboard corners with sub-pixel refinement."""
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        return False, None
    # OpenCV versions differ: corners arrive as (N, 1, 2) or (N, 2).
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined.reshape(-1, 1, 2)


def calibrate_pair(
    left_images: list[np.ndarray],
    right_images: list[np.ndarray],
    pattern_size: tuple[int, int] = (9, 6),
    square_mm: float = 25.0,
) -> tuple[CalibrationResult, list[str]]:
    """Run stereo calibration from synchronized image pairs.

    Returns (result, notes). Raises ValueError with a reason when there
    are not enough usable pairs -- calibration must never fail silently.
    """
    if len(left_images) != len(right_images):
        raise ValueError(
            f"left/right image counts differ: {len(left_images)} vs "
            f"{len(right_images)}"
        )
    objp = board_object_points(pattern_size[0], pattern_size[1], square_mm)
    obj_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    notes: list[str] = []
    image_size: tuple[int, int] | None = None
    for i, (limg, rimg) in enumerate(zip(left_images, right_images)):
        lgray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
        rgray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
        image_size = (lgray.shape[1], lgray.shape[0])
        lok, lcorn = find_board(lgray, pattern_size)
        rok, rcorn = find_board(rgray, pattern_size)
        if lok and rok:
            obj_points.append(objp)
            left_points.append(lcorn)
            right_points.append(rcorn)
        else:
            missing = []
            if not lok:
                missing.append("left")
            if not rok:
                missing.append("right")
            notes.append(
                f"pair {i}: checkerboard not found in {'/'.join(missing)} "
                "image; skipped"
            )
    if len(obj_points) < 5:
        raise ValueError(
            f"only {len(obj_points)} usable pairs; at least 5 are needed "
            "for a trustworthy calibration"
        )
    assert image_size is not None
    lerr, lmtx, ldist, _, _ = cv2.calibrateCamera(
        obj_points, left_points, image_size, None, None
    )
    rerr, rmtx, rdist, _, _ = cv2.calibrateCamera(
        obj_points, right_points, image_size, None, None
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    serr, _, _, _, _, rot, trans, _, _ = cv2.stereoCalibrate(
        obj_points, left_points, right_points, lmtx, ldist, rmtx, rdist,
        image_size, criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    result = CalibrationResult(
        image_size=image_size,
        left_camera_matrix=lmtx,
        left_distortion=ldist,
        right_camera_matrix=rmtx,
        right_distortion=rdist,
        rotation=rot,
        translation=trans,
        left_error=float(lerr),
        right_error=float(rerr),
        stereo_error=float(serr),
    )
    notes.append(
        f"used {len(obj_points)} pairs; stereo reprojection error "
        f"{serr:.3f} px; baseline {result.baseline_mm:.1f} mm"
    )
    if not result.quality_ok:
        notes.append(
            f"warning: stereo error {serr:.3f} px exceeds the "
            f"{QUALITY_ERROR_THRESHOLD_PX:.1f} px quality threshold"
        )
    return result, notes


def rectification_maps(result: CalibrationResult):
    """Compute (and return) the stereo rectification maps for both eyes."""
    size = result.image_size
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        result.left_camera_matrix, result.left_distortion,
        result.right_camera_matrix, result.right_distortion,
        size, result.rotation, result.translation, alpha=0,
    )
    lmapx, lmapy = cv2.initUndistortRectifyMap(
        result.left_camera_matrix, result.left_distortion, r1, p1, size,
        cv2.CV_32FC1,
    )
    rmapx, rmapy = cv2.initUndistortRectifyMap(
        result.right_camera_matrix, result.right_distortion, r2, p2, size,
        cv2.CV_32FC1,
    )
    return (lmapx, lmapy), (rmapx, rmapy)


def save_calibration(result: CalibrationResult, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    data = {
        "image_size": list(result.image_size),
        "left_camera_matrix": result.left_camera_matrix.tolist(),
        "left_distortion": result.left_distortion.tolist(),
        "right_camera_matrix": result.right_camera_matrix.tolist(),
        "right_distortion": result.right_distortion.tolist(),
        "rotation": result.rotation.tolist(),
        "translation": result.translation.tolist(),
        "left_error": result.left_error,
        "right_error": result.right_error,
        "stereo_error": result.stereo_error,
        "baseline_mm": result.baseline_mm,
    }
    with open(os.path.join(directory, "calibration.json"), "w",
              encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    (lmapx, lmapy), (rmapx, rmapy) = rectification_maps(result)
    np.savez_compressed(
        os.path.join(directory, "maps.npz"),
        lmapx=lmapx, lmapy=lmapy, rmapx=rmapx, rmapy=rmapy,
    )


def load_calibration(directory: str) -> tuple[CalibrationResult, tuple, list[str]]:
    """Load a saved calibration.

    Returns (result, rectify_maps, warnings). Warnings is non-empty when
    the calibration is low quality -- callers must surface it, per the
    "never fail (or trust) silently" rule.
    """
    with open(os.path.join(directory, "calibration.json"), "r",
              encoding="utf-8") as fh:
        data = json.load(fh)
    result = CalibrationResult(
        image_size=tuple(data["image_size"]),
        left_camera_matrix=np.array(data["left_camera_matrix"]),
        left_distortion=np.array(data["left_distortion"]),
        right_camera_matrix=np.array(data["right_camera_matrix"]),
        right_distortion=np.array(data["right_distortion"]),
        rotation=np.array(data["rotation"]),
        translation=np.array(data["translation"]),
        left_error=float(data["left_error"]),
        right_error=float(data["right_error"]),
        stereo_error=float(data["stereo_error"]),
    )
    maps_data = np.load(os.path.join(directory, "maps.npz"))
    maps = (
        (maps_data["lmapx"], maps_data["lmapy"]),
        (maps_data["rmapx"], maps_data["rmapy"]),
    )
    warnings: list[str] = []
    if not result.quality_ok:
        warnings.append(
            f"calibration quality is low: stereo reprojection error "
            f"{result.stereo_error:.3f} px exceeds "
            f"{QUALITY_ERROR_THRESHOLD_PX:.1f} px; rectification may be "
            "inaccurate -- consider recalibrating"
        )
    return result, maps, warnings

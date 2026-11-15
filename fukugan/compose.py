"""Side-by-side stereo composition.

Order of operations per eye:
  1. rectification maps (when calibration is loaded),
  2. manual offset translation (always available, calibration or not),
  3. resize into the eye's half of the output frame.
"""

from __future__ import annotations

import cv2
import numpy as np


def translate(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a frame by (dx, dy) pixels, filling exposed edges with black."""
    if dx == 0 and dy == 0:
        return frame
    h, w = frame.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


class StereoComposer:
    def __init__(
        self,
        output_width: int,
        output_height: int,
        *,
        left_offset: tuple[int, int] = (0, 0),
        right_offset: tuple[int, int] = (0, 0),
        rectify_maps: tuple[tuple[np.ndarray, np.ndarray],
                            tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        if output_width % 2 != 0:
            raise ValueError("output width must be even for side-by-side")
        self.out_w = output_width
        self.out_h = output_height
        self.eye_w = output_width // 2
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.rectify_maps = rectify_maps  # ((lmapx, lmapy), (rmapx, rmapy))

    def _prepare(self, frame: np.ndarray, eye: int,
                 offset: tuple[int, int]) -> np.ndarray:
        img = frame
        if self.rectify_maps is not None:
            mapx, mapy = self.rectify_maps[eye]
            if img.shape[1] == mapx.shape[1] and img.shape[0] == mapx.shape[0]:
                img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        img = translate(img, offset[0], offset[1])
        if img.shape[1] != self.eye_w or img.shape[0] != self.out_h:
            img = cv2.resize(img, (self.eye_w, self.out_h),
                             interpolation=cv2.INTER_AREA)
        return img

    def compose(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        out = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        out[:, 0 : self.eye_w] = self._prepare(left, 0, self.left_offset)
        out[:, self.eye_w :] = self._prepare(right, 1, self.right_offset)
        return out

    def compose_with_fallback(
        self,
        frames: list[np.ndarray | None],
        pair: tuple[int, int],
        reference_index: int,
    ) -> tuple[np.ndarray | None, list[str]]:
        """Compose from a full frame list, substituting the reference
        (center) camera when one side of the pair is missing.

        Returns (frame, reasons). Frame is None only when neither the pair
        nor the fallback can produce both eyes.
        """
        reasons: list[str] = []
        li, ri = pair
        left = frames[li] if li < len(frames) else None
        right = frames[ri] if ri < len(frames) else None
        ref = frames[reference_index] if reference_index < len(frames) else None
        if left is None and ref is not None and reference_index != li:
            left = ref
            reasons.append(
                f"left camera {li} unavailable; center camera "
                f"{reference_index} substituted (depth effect reduced)"
            )
        if right is None and ref is not None and reference_index != ri:
            right = ref
            reasons.append(
                f"right camera {ri} unavailable; center camera "
                f"{reference_index} substituted (depth effect reduced)"
            )
        if left is None or right is None:
            return None, reasons + ["both eyes could not be filled"]
        return self.compose(left, right), reasons

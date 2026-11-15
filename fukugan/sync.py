"""Timestamp-based frame synchronization.

Kept as pure functions over plain data so the policy can be unit tested
without cameras or threads:

- each input contributes a short history of (timestamp, frame) samples;
- `choose_set` picks, for a reference timestamp, the closest sample from
  every input and reports the per-input time offsets;
- `FrameSynchronizer` adds the stateful policy on top: hold the previous
  frame when an input is too far off, give up on an input after too many
  consecutive holds, and always explain what happened (never fail silently).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyncResult:
    frames: list[np.ndarray | None]  # None where no frame could be provided
    offsets_ms: list[float]  # per-input offset from the reference time
    held: list[bool]  # True where the previous frame was reused
    dead: list[bool]  # True where the input exceeded the hold limit
    reasons: list[str]  # human-readable notes, empty when perfectly in sync

    @property
    def ok(self) -> bool:
        return all(f is not None for f in self.frames)

    @property
    def degraded(self) -> bool:
        return any(self.held) or any(self.dead)


def closest_index(timestamps: list[float], target: float) -> int:
    """Index of the timestamp closest to `target` (timestamps ascending)."""
    best = 0
    best_diff = abs(timestamps[0] - target)
    for i in range(1, len(timestamps)):
        diff = abs(timestamps[i] - target)
        if diff < best_diff:
            best, best_diff = i, diff
    return best


def choose_set(
    histories: list[list[tuple[float, np.ndarray]]],
    reference_ts: float,
) -> tuple[list[np.ndarray | None], list[float]]:
    """Pick the closest frame to `reference_ts` from each history.

    Returns (frames, offsets_ms). A frame is None only when that history
    is empty.
    """
    frames: list[np.ndarray | None] = []
    offsets: list[float] = []
    for hist in histories:
        if not hist:
            frames.append(None)
            offsets.append(float("inf"))
            continue
        stamps = [ts for ts, _ in hist]
        idx = closest_index(stamps, reference_ts)
        frames.append(hist[idx][1])
        offsets.append((stamps[idx] - reference_ts) * 1000.0)
    return frames, offsets


class FrameSynchronizer:
    """Stateful sync policy: hold-then-declare-dead, always with reasons."""

    def __init__(self, n_inputs: int, tolerance_ms: float = 50.0,
                 max_hold_frames: int = 5):
        self.n = n_inputs
        self.tolerance_ms = tolerance_ms
        self.max_hold_frames = max_hold_frames
        self._last_frames: list[np.ndarray | None] = [None] * n_inputs
        self._hold_counts: list[int] = [0] * n_inputs
        self._dead: list[bool] = [False] * n_inputs
        self.stats = {"sets": 0, "held_frames": 0, "dead_inputs": 0}

    def reset_input(self, index: int) -> None:
        """Revive an input (e.g. after its camera recovered)."""
        self._dead[index] = False
        self._hold_counts[index] = 0

    def synchronize(
        self,
        histories: list[list[tuple[float, np.ndarray]]],
        reference_ts: float,
    ) -> SyncResult:
        fresh, offsets = choose_set(histories, reference_ts)
        frames: list[np.ndarray | None] = []
        held: list[bool] = []
        reasons: list[str] = []
        for i in range(self.n):
            off = offsets[i]
            frame = fresh[i]
            if self._dead[i]:
                frames.append(self._last_frames[i])
                held.append(self._last_frames[i] is not None)
                continue
            if frame is not None and abs(off) <= self.tolerance_ms:
                frames.append(frame)
                held.append(False)
                self._last_frames[i] = frame
                self._hold_counts[i] = 0
            else:
                # Out of tolerance (or nothing arrived): hold previous.
                self._hold_counts[i] += 1
                if self._hold_counts[i] > self.max_hold_frames:
                    self._dead[i] = True
                    self.stats["dead_inputs"] += 1
                    frames.append(self._last_frames[i])
                    held.append(self._last_frames[i] is not None)
                    if frame is None:
                        reasons.append(
                            f"input {i}: no frames arriving; "
                            f"marked stopped after {self.max_hold_frames} holds"
                        )
                    else:
                        reasons.append(
                            f"input {i}: sync offset {off:.0f} ms stayed "
                            f"outside the {self.tolerance_ms:.0f} ms tolerance; "
                            "marked stopped"
                        )
                else:
                    frames.append(self._last_frames[i])
                    held.append(self._last_frames[i] is not None)
                    if self._last_frames[i] is not None:
                        self.stats["held_frames"] += 1
                        reasons.append(
                            f"input {i}: offset {off:.0f} ms exceeds tolerance; "
                            "reusing previous frame"
                        )
        self.stats["sets"] += 1
        return SyncResult(
            frames=frames,
            offsets_ms=offsets,
            held=held,
            dead=list(self._dead),
            reasons=reasons,
        )

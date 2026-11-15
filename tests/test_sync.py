import numpy as np

from fukugan.sync import FrameSynchronizer, choose_set, closest_index


def frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def hist(*pairs):
    return [(ts, frame(v)) for ts, v in pairs]


def test_closest_index():
    stamps = [0.0, 0.033, 0.066, 0.099]
    assert closest_index(stamps, 0.05) == 2
    assert closest_index(stamps, 0.0) == 0
    assert closest_index(stamps, 10.0) == 3


def test_choose_set_picks_nearest_frames():
    h = [hist((0.0, 1), (0.033, 2), (0.066, 3)),
         hist((0.001, 7), (0.034, 8))]
    frames, offsets = choose_set(h, 0.033)
    assert frames[0][0, 0, 0] == 2
    assert frames[1][0, 0, 0] == 8
    assert abs(offsets[0]) < 1.0
    assert abs(offsets[1] - 1.0) < 1.5


def test_choose_set_empty_history():
    frames, offsets = choose_set([[], hist((0.0, 5))], 0.0)
    assert frames[0] is None
    assert frames[1] is not None


def test_in_sync_frames_pass_through():
    syncer = FrameSynchronizer(2, tolerance_ms=50, max_hold_frames=2)
    result = syncer.synchronize(
        [hist((1.000, 10)), hist((1.010, 20))], reference_ts=1.0
    )
    assert result.ok and not result.degraded
    assert result.frames[0][0, 0, 0] == 10
    assert result.frames[1][0, 0, 0] == 20
    assert result.reasons == []


def test_out_of_sync_input_holds_previous_frame_with_reason():
    syncer = FrameSynchronizer(2, tolerance_ms=50, max_hold_frames=3)
    syncer.synchronize([hist((1.0, 10)), hist((1.0, 20))], 1.0)
    # Input 1 jumps 500 ms away from the reference.
    result = syncer.synchronize(
        [hist((2.0, 11)), hist((2.5, 21))], reference_ts=2.0
    )
    assert result.held[1] is True
    assert result.frames[1][0, 0, 0] == 20  # previous frame reused
    assert any("input 1" in r for r in result.reasons)
    assert result.degraded


def test_input_declared_dead_after_hold_limit():
    syncer = FrameSynchronizer(2, tolerance_ms=50, max_hold_frames=2)
    syncer.synchronize([hist((1.0, 10)), hist((1.0, 20))], 1.0)
    result = None
    for step in range(3):
        t = 2.0 + step
        result = syncer.synchronize(
            [hist((t, 11)), hist((t + 5.0, 21))], reference_ts=t
        )
    assert result is not None
    assert result.dead[1] is True
    assert any("marked stopped" in r for r in result.reasons)
    # Reviving works.
    syncer.reset_input(1)
    result = syncer.synchronize(
        [hist((10.0, 12)), hist((10.0, 22))], reference_ts=10.0
    )
    assert result.dead[1] is False
    assert result.frames[1][0, 0, 0] == 22

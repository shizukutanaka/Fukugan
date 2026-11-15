from fukugan.config import Config
from fukugan.encoder import (
    EncoderChoice,
    build_command,
    encoder_args,
    probe_nvenc,
    select_encoder,
)

X264 = EncoderChoice("libx264", False, "test")
NVENC = EncoderChoice("h264_nvenc", True, "test")


def test_probe_with_missing_ffmpeg_reports_reason():
    ok, detail = probe_nvenc(ffmpeg="/nonexistent/ffmpeg")
    assert ok is False
    assert "not found" in detail


def test_select_x264_skips_probe():
    choice = select_encoder("x264", ffmpeg="/nonexistent/ffmpeg")
    assert choice.codec == "libx264"
    assert choice.hardware is False


def test_select_nvenc_falls_back_with_reason_when_probe_fails():
    choice = select_encoder("nvenc", ffmpeg="/nonexistent/ffmpeg")
    assert choice.codec == "libx264"
    assert "fell back" in choice.reason


def test_hls_only_command_shape():
    cfg = Config.default_virtual()
    cmd = build_command(cfg, X264)
    assert cmd[0] == "ffmpeg"
    assert "rawvideo" in cmd and "bgr24" in cmd
    assert "2560x720" in cmd
    assert "libx264" in cmd
    assert cmd[-2] == "hls"
    assert cmd[-1].endswith("index.m3u8")
    assert "tee" not in cmd


def test_push_command_uses_tee_with_all_targets():
    cfg = Config.default_virtual()
    cfg.stream.rtsp_url = "rtsp://localhost:8554/fukugan"
    cfg.stream.srt_url = "srt://localhost:8890?streamid=publish:fukugan"
    cmd = build_command(cfg, X264)
    assert "tee" in cmd
    tee = cmd[-1]
    assert "index.m3u8" in tee
    assert "rtsp://localhost:8554/fukugan" in tee
    assert "srt://localhost:8890" in tee
    # Push targets must not be able to kill the local HLS output.
    assert tee.count("onfail=ignore") == 2


def test_nvenc_args_use_hardware_preset_mapping():
    cfg = Config.default_virtual()
    args = encoder_args(NVENC, cfg)
    assert "h264_nvenc" in args
    assert "yuv420p" in args
    xargs = encoder_args(X264, cfg)
    assert "zerolatency" in xargs


def test_keyframe_interval_matches_segment_length():
    cfg = Config.default_virtual()
    cfg.stream.hls_segment_seconds = 4
    cmd = build_command(cfg, X264)
    gop = cmd[cmd.index("-g") + 1]
    assert gop == str(cfg.output.fps * 4)

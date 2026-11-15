import pytest

from fukugan.config import CameraConfig, Config


def two_virtual() -> Config:
    cfg = Config()
    cfg.cameras = [
        CameraConfig(type="virtual", label="L"),
        CameraConfig(type="virtual", label="R"),
    ]
    return cfg


def test_default_virtual_is_valid():
    assert Config.default_virtual().validate() == []


def test_too_few_cameras_is_an_error_with_reason():
    cfg = Config()
    cfg.cameras = [CameraConfig(type="virtual")]
    errors = cfg.validate()
    assert any("at least 2" in e for e in errors)


def test_unknown_camera_type_rejected():
    cfg = two_virtual()
    cfg.cameras[0].type = "hologram"
    assert any("unknown type" in e for e in cfg.validate())


def test_odd_output_width_rejected():
    cfg = two_virtual()
    cfg.output.width = 2561
    assert any("even" in e for e in cfg.validate())


def test_bad_pair_rejected():
    cfg = two_virtual()
    cfg.pair = [1, 1]
    assert any("differ" in e for e in cfg.validate())
    cfg.pair = [0, 5]
    assert any("out of range" in e for e in cfg.validate())
    cfg.pair = [1, 0]
    assert any("ordered" in e for e in cfg.validate())


def test_pair_selection_defaults_to_outermost():
    cfg = Config()
    cfg.cameras = [CameraConfig(type="virtual") for _ in range(3)]
    assert cfg.select_pair() == (0, 2)
    assert cfg.reference_index() == 1
    cfg.pair = [0, 1]
    assert cfg.select_pair() == (0, 1)


def test_round_trip_through_dict():
    cfg = Config.default_virtual()
    cfg.stream.rtsp_url = "rtsp://localhost:8554/fukugan"
    again = Config.from_dict(cfg.to_dict())
    assert again.stream.rtsp_url == "rtsp://localhost:8554/fukugan"
    assert len(again.cameras) == 2
    assert again.validate() == []


def test_bad_url_and_port_rejected():
    cfg = two_virtual()
    cfg.stream.srt_url = "not a url"
    cfg.server.port = 70000
    errors = cfg.validate()
    assert any("not a URL" in e for e in errors)
    assert any("port" in e for e in errors)

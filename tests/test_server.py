import pytest

from fukugan import server
from fukugan.config import Config
from fukugan.server import AppServer, headset_urls


@pytest.fixture
def lan(monkeypatch):
    monkeypatch.setattr(server, "lan_address", lambda: "192.168.1.23")


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_bind_offers_no_headset_url(lan, host):
    assert headset_urls(host, 8080) == []


@pytest.mark.parametrize("host", ["0.0.0.0", ""])
def test_wildcard_bind_resolves_to_lan_address(lan, host):
    assert headset_urls(host, 8080) == [
        "http://192.168.1.23:8080/hls/index.m3u8"
    ]


def test_wildcard_bind_without_lan_offers_no_url(monkeypatch):
    monkeypatch.setattr(server, "lan_address", lambda: None)
    assert headset_urls("0.0.0.0", 8080) == []


def test_explicit_address_is_used_as_is(lan):
    assert headset_urls("10.0.0.5", 9000) == [
        "http://10.0.0.5:9000/hls/index.m3u8"
    ]


def test_lan_address_is_never_loopback():
    address = server.lan_address()
    assert address is None or not address.startswith("127.")


def test_app_server_reports_the_actually_bound_port(lan):
    cfg = Config.default_virtual()
    cfg.server.host, cfg.server.port = "0.0.0.0", 0
    app = AppServer(cfg)
    try:
        port = app.address[1]
        assert port != 0
        assert app.headset_urls() == [
            f"http://192.168.1.23:{port}/hls/index.m3u8"
        ]
    finally:
        app._httpd.server_close()

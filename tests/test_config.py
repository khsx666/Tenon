"""Unit tests for tenon.config — no network or real API key required."""
import pytest

from tenon.config import ConfigError, load_config

_ENV_KEYS = ("TENON_API_KEY", "TENON_BASE_URL", "TENON_MODEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_from_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "TENON_API_KEY=sk-test-123\n"
        "export TENON_BASE_URL=https://example.test/v1\n"
        'TENON_MODEL="some-model"\n'
    )
    cfg = load_config(str(env))
    assert cfg.api_key == "sk-test-123"
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.model == "some-model"
    assert cfg.max_turns == 50
    assert cfg.permission_mode == "ask"


def test_env_var_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TENON_API_KEY", "sk-from-env")
    monkeypatch.setenv("TENON_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("TENON_MODEL", "env-model")
    env = tmp_path / ".env"
    env.write_text("TENON_API_KEY=sk-from-file\n")
    assert load_config(str(env)).api_key == "sk-from-env"


def test_missing_key_gives_guidance_without_leaking(tmp_path):
    with pytest.raises(ConfigError) as exc_info:
        load_config(str(tmp_path / ".env"))  # file absent entirely
    msg = str(exc_info.value)
    assert "TENON_API_KEY" in msg
    assert ".env.example" in msg


def test_missing_base_url_and_model(tmp_path):
    env = tmp_path / ".env"
    env.write_text("TENON_API_KEY=sk-test-123\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(str(env))
    assert "TENON_BASE_URL" in str(exc_info.value)
    assert "TENON_MODEL" in str(exc_info.value)


def test_malformed_line_reports_location(tmp_path):
    env = tmp_path / ".env"
    env.write_text("TENON_API_KEY no-equals-here\n")
    with pytest.raises(ConfigError) as exc_info:
        load_config(str(env))
    assert ":1:" in str(exc_info.value)

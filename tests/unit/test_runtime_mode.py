"""Unit tests for runtime source mode state and URL handling."""

from __future__ import annotations

import pytest

from pipeworks.api import runtime_mode
from pipeworks.api.services import deprecations
from pipeworks.api.services.deprecations import LEGACY_POLICY_ENV_ALIAS_WARNING_TEMPLATE

_RUNTIME_ENV_VARS = (
    "PW_POLICY_SOURCE_MODE",
    "PW_POLICY_MUD_API_BASE_URL",
    "PW_POLICY_DEV_MUD_API_BASE_URL",
    "PW_POLICY_PROD_MUD_API_BASE_URL",
    "PW_POLICY_LOCAL_MUD_API_BASE_URL",
    "PW_POLICY_REMOTE_DEV_MUD_API_BASE_URL",
    "PW_POLICY_REMOTE_PROD_MUD_API_BASE_URL",
)


@pytest.fixture(autouse=True)
def reset_runtime_mode_state(monkeypatch):
    """Reset runtime mode globals and environment before each test."""
    for env_name in _RUNTIME_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)
    runtime_mode._reset_runtime_mode_for_tests()
    yield
    runtime_mode._reset_runtime_mode_for_tests()


def test_get_runtime_mode_defaults_to_server_dev_profile():
    """Default state should be dev profile with deterministic options."""
    state = runtime_mode.get_runtime_mode()

    assert state.mode_key == "server_dev"
    assert state.source_kind == "server_api"
    assert state.active_server_url == "http://127.0.0.1:8000"
    assert [option.mode_key for option in state.options] == [
        "server_dev",
        "server_prod",
    ]


def test_normalize_server_url_trims_slash_and_whitespace():
    """Server URL normalization should keep only absolute http(s) URLs."""
    assert runtime_mode._normalize_server_url("  https://mud.example.com/  ") == (
        "https://mud.example.com"
    )


def test_set_runtime_mode_accepts_normalized_server_url_override():
    """Switching profiles should persist normalized URL overrides."""
    state = runtime_mode.set_runtime_mode(
        mode_key="server_prod",
        server_url=" https://mud.example.com/ ",
    )

    assert state.mode_key == "server_prod"
    assert state.active_server_url == "https://mud.example.com"


def test_set_runtime_mode_rejects_unknown_mode_key():
    """Unknown mode keys should fail with a stable validation message."""
    with pytest.raises(ValueError, match="Unknown runtime mode"):
        runtime_mode.set_runtime_mode(mode_key="unknown_mode", server_url=None)


def test_set_runtime_mode_rejects_invalid_server_url():
    """Only absolute http(s) URLs are accepted for runtime server overrides."""
    with pytest.raises(ValueError, match="absolute http\\(s\\)"):
        runtime_mode.set_runtime_mode(mode_key="server_dev", server_url="mud.local:8000")


def test_set_runtime_mode_requires_url_when_profile_has_no_default(monkeypatch):
    """A server profile without default/override URL should be rejected."""
    monkeypatch.setattr(runtime_mode, "_DEFAULT_DEV_BASE_URL", "")
    runtime_mode._reset_runtime_mode_for_tests()

    with pytest.raises(ValueError, match="requires a mud-server URL"):
        runtime_mode.set_runtime_mode(mode_key="server_dev", server_url=None)


def test_require_server_api_url_returns_active_url():
    """require_server_api_url should return the currently selected URL."""
    runtime_mode.set_runtime_mode(mode_key="server_prod", server_url="https://prod.example.com")

    assert runtime_mode.require_server_api_url() == "https://prod.example.com"


def test_require_server_api_url_raises_when_active_profile_has_no_url(monkeypatch):
    """require_server_api_url should fail when no active URL is configured."""
    monkeypatch.setattr(runtime_mode, "_DEFAULT_DEV_BASE_URL", "")
    runtime_mode._reset_runtime_mode_for_tests()

    with pytest.raises(runtime_mode.RuntimeModeUnavailableError, match="No mud-server URL"):
        runtime_mode.require_server_api_url()


def test_reset_runtime_mode_uses_env_source_mode_and_clears_overrides(monkeypatch):
    """Reset should restore env-selected mode and clear old URL overrides."""
    runtime_mode.set_runtime_mode(mode_key="server_prod", server_url="https://override.example")
    monkeypatch.setenv("PW_POLICY_SOURCE_MODE", "server_prod")

    runtime_mode._reset_runtime_mode_for_tests()
    state = runtime_mode.get_runtime_mode()

    assert state.mode_key == "server_prod"
    assert state.active_server_url == "https://api.pipe-works.org"


def test_runtime_mode_warns_when_legacy_env_alias_is_effective(monkeypatch, caplog):
    """Legacy policy URL aliases should emit a one-time deprecation warning."""
    deprecations._WARNED_KEYS.clear()
    monkeypatch.setenv("PW_POLICY_REMOTE_DEV_MUD_API_BASE_URL", "https://legacy-dev.example.com")

    with caplog.at_level("WARNING"):
        runtime_mode._reset_runtime_mode_for_tests()
        state = runtime_mode.get_runtime_mode()

    assert state.active_server_url == "https://legacy-dev.example.com"
    warning_text = (
        LEGACY_POLICY_ENV_ALIAS_WARNING_TEMPLATE % "PW_POLICY_REMOTE_DEV_MUD_API_BASE_URL"
    )
    assert warning_text in caplog.text

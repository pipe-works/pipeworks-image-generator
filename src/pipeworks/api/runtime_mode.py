"""Runtime source-mode state for image-generator server-backed snippets.

The image generator supports explicit mud-server operating profiles so users
can choose which canonical API host backs snippet dropdown loading:
- mud-server API workflows (dev/production server profiles)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import RLock
from urllib.parse import urlparse

_MODE_SERVER_DEV = "server_dev"
_MODE_SERVER_PROD = "server_prod"

_SOURCE_SERVER_API = "server_api"

_SOURCE_MODE_ENV = "PW_POLICY_SOURCE_MODE"
_DEV_BASE_URL_ENV = "PW_POLICY_DEV_MUD_API_BASE_URL"
_PROD_BASE_URL_ENV = "PW_POLICY_PROD_MUD_API_BASE_URL"

_DEFAULT_DEV_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_PROD_BASE_URL = "https://api.pipe-works.org"


class RuntimeModeUnavailableError(RuntimeError):
    """Raised when mud-server API behavior is unavailable for active profile."""


@dataclass(frozen=True, slots=True)
class RuntimeModeOption:
    """One selectable source-mode profile."""

    mode_key: str
    label: str
    source_kind: str
    default_server_url: str | None
    url_editable: bool


@dataclass(frozen=True, slots=True)
class RuntimeModeState:
    """Resolved runtime mode state returned to API/UI callers."""

    mode_key: str
    source_kind: str
    active_server_url: str | None
    options: tuple[RuntimeModeOption, ...]


_STATE_LOCK = RLock()
_ACTIVE_MODE_KEY = _MODE_SERVER_DEV
_MODE_URL_OVERRIDES: dict[str, str] = {}


def get_runtime_mode() -> RuntimeModeState:
    """Return current runtime source-mode configuration."""
    with _STATE_LOCK:
        options = _build_options()
        active_option = _option_by_key(options, _ACTIVE_MODE_KEY)
        if active_option is None:
            active_option = options[0]
        active_server_url = _resolve_active_url_for_option(active_option)
        return RuntimeModeState(
            mode_key=active_option.mode_key,
            source_kind=active_option.source_kind,
            active_server_url=active_server_url,
            options=options,
        )


def set_runtime_mode(*, mode_key: str, server_url: str | None) -> RuntimeModeState:
    """Set active runtime source mode and optional server URL override."""
    normalized_key = _normalize_mode_key((mode_key or "").strip())
    with _STATE_LOCK:
        options = _build_options()
        option = _option_by_key(options, normalized_key)
        if option is None:
            raise ValueError(f"Unknown runtime mode: {mode_key!r}")

        if option.source_kind == _SOURCE_SERVER_API:
            resolved_url = _normalize_server_url(
                server_url if server_url is not None else _resolve_active_url_for_option(option)
            )
            if not resolved_url:
                raise ValueError(f"Runtime mode '{option.mode_key}' requires a mud-server URL.")
            _MODE_URL_OVERRIDES[option.mode_key] = resolved_url

        global _ACTIVE_MODE_KEY  # noqa: PLW0603
        _ACTIVE_MODE_KEY = option.mode_key
        return get_runtime_mode()


def require_server_api_url() -> str:
    """Return active mud-server URL or raise a stable mode-unavailable error."""
    state = get_runtime_mode()
    if state.source_kind != _SOURCE_SERVER_API:
        raise RuntimeModeUnavailableError(
            "Runtime mode is not configured for mud-server API usage."
        )
    if not state.active_server_url:
        raise RuntimeModeUnavailableError("No mud-server URL configured for the active mode.")
    return state.active_server_url


def _build_options() -> tuple[RuntimeModeOption, ...]:
    """Build deterministic runtime mode options from environment defaults."""
    dev_raw_value = os.getenv(_DEV_BASE_URL_ENV)
    prod_raw_value = os.getenv(_PROD_BASE_URL_ENV)

    dev_default = _normalize_server_url(dev_raw_value or _DEFAULT_DEV_BASE_URL)
    prod_default = _normalize_server_url(prod_raw_value or _DEFAULT_PROD_BASE_URL)

    return (
        RuntimeModeOption(
            mode_key=_MODE_SERVER_DEV,
            label="Server (Dev)",
            source_kind=_SOURCE_SERVER_API,
            default_server_url=dev_default or _DEFAULT_DEV_BASE_URL,
            url_editable=True,
        ),
        RuntimeModeOption(
            mode_key=_MODE_SERVER_PROD,
            label="Server (Production)",
            source_kind=_SOURCE_SERVER_API,
            default_server_url=prod_default or _DEFAULT_PROD_BASE_URL,
            url_editable=True,
        ),
    )


def _normalize_mode_key(mode_key: str) -> str:
    """Normalize mode keys for runtime mode selection."""
    return mode_key


def _option_by_key(
    options: tuple[RuntimeModeOption, ...],
    mode_key: str,
) -> RuntimeModeOption | None:
    """Return option matching ``mode_key`` or ``None``."""
    for option in options:
        if option.mode_key == mode_key:
            return option
    return None


def _resolve_active_url_for_option(option: RuntimeModeOption) -> str | None:
    """Resolve active URL for a server option including in-memory overrides."""
    if option.source_kind != _SOURCE_SERVER_API:
        return None
    override_url = _MODE_URL_OVERRIDES.get(option.mode_key)
    if override_url:
        return override_url
    return option.default_server_url


def _normalize_server_url(value: str | None) -> str:
    """Normalize server URL and reject unsupported schemes."""
    normalized = (value or "").strip().rstrip("/")
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Mud-server URL must be absolute http(s), got {value!r}.")
    return normalized


def _reset_runtime_mode_for_tests() -> None:
    """Reset global runtime-mode state for unit/integration tests."""
    with _STATE_LOCK:
        initial_key = _normalize_mode_key(
            (os.getenv(_SOURCE_MODE_ENV, _MODE_SERVER_DEV) or "").strip()
        )
        options = _build_options()
        option = _option_by_key(options, initial_key)
        global _ACTIVE_MODE_KEY  # noqa: PLW0603
        _ACTIVE_MODE_KEY = option.mode_key if option is not None else _MODE_SERVER_DEV
        _MODE_URL_OVERRIDES.clear()


_reset_runtime_mode_for_tests()

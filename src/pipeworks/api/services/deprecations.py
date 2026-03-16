"""Deprecation metadata and one-time warning helpers."""

from __future__ import annotations

import logging
from threading import RLock

LEGACY_PROMPT_SCHEMA_DEPRECATION_HEADER = "X-Pipeworks-Deprecation"
LEGACY_PROMPT_SCHEMA_DEPRECATION_VALUE = (
    "prompt-schema-v1 is deprecated and will be removed in the next release; "
    "send prompt_schema_version=2"
)

LEGACY_PROMPTS_JSON_FALLBACK_WARNING = (
    "Legacy prompt fallback from data/prompts.json is deprecated and will be "
    "removed in the next release. Migrate to prepend.json/main.json/append.json."
)

LEGACY_POLICY_ENV_ALIAS_WARNING_TEMPLATE = (
    "Legacy runtime env alias '%s' is deprecated and will be removed in the "
    "next release. Prefer PW_POLICY_DEV_MUD_API_BASE_URL and "
    "PW_POLICY_PROD_MUD_API_BASE_URL."
)

_WARNED_KEYS: set[str] = set()
_WARN_LOCK = RLock()


def warn_once(*, logger: logging.Logger, key: str, message: str) -> None:
    """Emit one warning per process for a stable key."""
    with _WARN_LOCK:
        if key in _WARNED_KEYS:
            return
        _WARNED_KEYS.add(key)
    logger.warning(message)

"""Shared mud-server HTTP transport helpers for image-generator runtime APIs."""

from __future__ import annotations

import json
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class MudApiRuntime(Protocol):
    """Runtime contract needed for session-authenticated API requests."""

    @property
    def base_url(self) -> str: ...

    @property
    def session_id(self) -> str: ...

    @property
    def timeout_seconds(self) -> float: ...


def normalize_base_url(value: str | None) -> str:
    """Normalize mud API base URL into stable, slash-trimmed form."""
    return (value or "").strip().rstrip("/")


def mud_api_http_error_detail(exc: object) -> str:
    """Extract best-effort detail from mud-server API HTTP error payloads."""
    status_code = getattr(exc, "code", "unknown")
    default_detail = f"HTTP {status_code}"

    reader = getattr(exc, "read", None)
    if not callable(reader):
        return default_detail

    try:
        raw_payload = reader()
        if isinstance(raw_payload, bytes):
            text = raw_payload.decode("utf-8")
        else:
            text = str(raw_payload)
        payload = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return default_detail

    if isinstance(payload, dict):
        detail = payload.get("detail")
        code = payload.get("code")
        if detail and code:
            return f"{code}: {detail}"
        if detail:
            return str(detail)
    return default_detail


def request_json(
    *,
    method: str,
    url: str,
    timeout_seconds: float,
    json_payload: dict[str, object] | None = None,
    allow_not_found: bool = False,
    error_prefix: str,
    non_object_error_message: str,
    opener=urlopen,
) -> dict[str, object] | None:
    """Issue one HTTP request and decode JSON response with stable errors."""
    body = None
    headers = {"Accept": "application/json"}
    if json_payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(json_payload).encode("utf-8")

    request = Request(url=url, method=method, data=body, headers=headers)
    try:
        with opener(request, timeout=timeout_seconds) as response:  # noqa: S310
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if allow_not_found and exc.code == 404:
            return None
        detail = mud_api_http_error_detail(exc)
        raise ValueError(f"{error_prefix} ({method} {url}): {detail}") from exc
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{error_prefix} ({method} {url}): {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(non_object_error_message)
    return cast(dict[str, object], parsed)


def fetch_mud_api_json(
    *,
    runtime: MudApiRuntime,
    method: str,
    path: str,
    query_params: dict[str, str],
    json_payload: dict[str, object] | None = None,
    opener=urlopen,
) -> dict[str, object]:
    """Issue one mud-server API request with session injection."""
    normalized_query = {key: value for key, value in query_params.items() if value}
    normalized_query["session_id"] = runtime.session_id
    query = urlencode(normalized_query, doseq=False)
    url = f"{runtime.base_url}{path}?{query}" if query else f"{runtime.base_url}{path}"

    payload = request_json(
        method=method,
        url=url,
        timeout_seconds=runtime.timeout_seconds,
        json_payload=json_payload,
        allow_not_found=False,
        error_prefix="Mud API request failed",
        non_object_error_message=f"Mud API response for {method} {url} must be a JSON object.",
        opener=opener,
    )
    if payload is None:  # pragma: no cover - guarded by allow_not_found=False
        raise ValueError(
            f"Mud API request failed ({method} {url}): response was unexpectedly empty."
        )
    return payload


def fetch_mud_api_json_anonymous(
    *,
    base_url: str,
    method: str,
    path: str,
    body: dict[str, object] | None,
    timeout_seconds: float = 8.0,
    opener=urlopen,
) -> dict[str, object]:
    """Issue one mud-server API request without session query injection."""
    url = f"{base_url}{path}"
    payload = request_json(
        method=method,
        url=url,
        timeout_seconds=timeout_seconds,
        json_payload=body,
        allow_not_found=False,
        error_prefix="Mud API request failed",
        non_object_error_message=f"Mud API response for {method} {url} must be a JSON object.",
        opener=opener,
    )
    if payload is None:  # pragma: no cover - guarded by allow_not_found=False
        raise ValueError(
            f"Mud API request failed ({method} {url}): response was unexpectedly empty."
        )
    return payload

"""HTTP transport helpers shared by controller and worker flows."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen


def worker_api_error_detail(exc: HTTPError) -> str:
    """Extract best-effort detail from worker HTTP error payload."""
    default_detail = f"HTTP {exc.code}"
    try:
        raw = exc.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        parsed = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return default_detail

    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if detail:
            return str(detail)
    return default_detail


def post_json_with_bearer(
    *,
    base_url: str,
    path: str,
    bearer_token: str,
    timeout_seconds: float,
    payload: dict[str, object],
) -> dict[str, object]:
    """POST JSON with bearer auth and decode object responses."""
    url = f"{base_url.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    request = UrlRequest(
        url=url,
        method="POST",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310  # nosec B310
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ValueError(f"{path} failed: {worker_api_error_detail(exc)}") from exc
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} failed: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{path} response must be a JSON object.")
    return parsed

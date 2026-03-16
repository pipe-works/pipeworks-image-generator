"""Unit tests for mud-server API transport helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from pipeworks.api import mud_api_client


class _FakeResponse:
    """Simple context-manager HTTP response stub with byte payload."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@dataclass(frozen=True, slots=True)
class _Runtime:
    """Minimal runtime payload implementing MudApiRuntime protocol fields."""

    base_url: str
    session_id: str
    timeout_seconds: float


def _http_error(*, url: str, code: int, payload: bytes) -> HTTPError:
    """Build one HTTPError whose body can be consumed by read()."""
    return HTTPError(url=url, code=code, msg="error", hdrs=None, fp=BytesIO(payload))


def test_normalize_base_url_strips_trailing_slashes():
    """Base URL normalization should trim whitespace and trailing slashes."""
    assert mud_api_client.normalize_base_url(" https://mud.example.com/// ") == (
        "https://mud.example.com"
    )


def test_mud_api_http_error_detail_extracts_code_and_detail():
    """HTTP error detail helper should parse JSON payload when present."""
    exc = _http_error(
        url="https://mud.example.com/api",
        code=403,
        payload=b'{"code":"forbidden","detail":"Role denied"}',
    )
    assert mud_api_client.mud_api_http_error_detail(exc) == "forbidden: Role denied"


def test_mud_api_http_error_detail_falls_back_when_payload_unusable():
    """HTTP detail helper should fail closed to an HTTP-status fallback."""
    exc = _http_error(
        url="https://mud.example.com/api",
        code=502,
        payload=b"not-json",
    )
    assert mud_api_client.mud_api_http_error_detail(exc) == "HTTP 502"


def test_request_json_returns_parsed_payload_and_sets_json_headers():
    """request_json should emit JSON request headers and decode object response."""
    captured: dict[str, object] = {}

    def opener(request, timeout):
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["body"] = request.data
        captured["timeout"] = timeout
        return _FakeResponse(b'{"ok": true}')

    payload = mud_api_client.request_json(
        method="POST",
        url="https://mud.example.com/login",
        timeout_seconds=3.5,
        json_payload={"username": "admin"},
        allow_not_found=False,
        error_prefix="Mud API request failed",
        non_object_error_message="response must be object",
        opener=opener,
    )

    assert payload == {"ok": True}
    assert captured["method"] == "POST"
    assert captured["url"] == "https://mud.example.com/login"
    assert captured["timeout"] == 3.5
    assert captured["body"] == b'{"username": "admin"}'
    assert captured["headers"] == {
        "accept": "application/json",
        "content-type": "application/json",
    }


def test_request_json_returns_none_for_allowed_not_found():
    """When allow_not_found=True, HTTP 404 should map to None."""

    def opener(request, timeout):
        raise _http_error(
            url=request.full_url,
            code=404,
            payload=b'{"detail":"missing"}',
        )

    payload = mud_api_client.request_json(
        method="GET",
        url="https://mud.example.com/api/policies",
        timeout_seconds=2.0,
        json_payload=None,
        allow_not_found=True,
        error_prefix="Mud API request failed",
        non_object_error_message="response must be object",
        opener=opener,
    )
    assert payload is None


def test_request_json_raises_value_error_for_http_errors():
    """HTTP failures should include extracted API detail in raised ValueError."""

    def opener(request, timeout):
        raise _http_error(
            url=request.full_url,
            code=401,
            payload=b'{"code":"invalid_credentials","detail":"Bad username/password"}',
        )

    with pytest.raises(ValueError, match="invalid_credentials: Bad username/password"):
        mud_api_client.request_json(
            method="POST",
            url="https://mud.example.com/login",
            timeout_seconds=2.0,
            json_payload={"username": "admin"},
            allow_not_found=False,
            error_prefix="Mud API request failed",
            non_object_error_message="response must be object",
            opener=opener,
        )


def test_request_json_raises_value_error_for_transport_and_decode_failures():
    """Transport errors and malformed JSON should be surfaced as ValueError."""

    def transport_fail_opener(request, timeout):
        raise URLError("connection refused")

    with pytest.raises(ValueError, match="connection refused"):
        mud_api_client.request_json(
            method="GET",
            url="https://mud.example.com/api/policies",
            timeout_seconds=1.0,
            json_payload=None,
            allow_not_found=False,
            error_prefix="Mud API request failed",
            non_object_error_message="response must be object",
            opener=transport_fail_opener,
        )

    def decode_fail_opener(request, timeout):
        return _FakeResponse(b"not-json")

    with pytest.raises(ValueError, match="Mud API request failed"):
        mud_api_client.request_json(
            method="GET",
            url="https://mud.example.com/api/policies",
            timeout_seconds=1.0,
            json_payload=None,
            allow_not_found=False,
            error_prefix="Mud API request failed",
            non_object_error_message="response must be object",
            opener=decode_fail_opener,
        )


def test_request_json_raises_for_non_object_json_payload():
    """Non-dict responses should raise the caller-provided contract message."""

    def opener(request, timeout):
        return _FakeResponse(b'["not","an","object"]')

    with pytest.raises(ValueError, match="must be an object"):
        mud_api_client.request_json(
            method="GET",
            url="https://mud.example.com/api/policies",
            timeout_seconds=1.0,
            json_payload=None,
            allow_not_found=False,
            error_prefix="Mud API request failed",
            non_object_error_message="Response must be an object.",
            opener=opener,
        )


def test_fetch_mud_api_json_injects_session_id_query_parameter():
    """Session-authenticated fetch should append runtime session_id to query."""
    captured: dict[str, object] = {}
    runtime = _Runtime(
        base_url="https://mud.example.com",
        session_id="session-admin-1",
        timeout_seconds=4.0,
    )

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = request.data
        captured["timeout"] = timeout
        return _FakeResponse(b'{"items":[]}')

    payload = mud_api_client.fetch_mud_api_json(
        runtime=runtime,
        method="GET",
        path="/api/policies",
        query_params={"namespace": "image.prompts", "empty": ""},
        json_payload=None,
        opener=opener,
    )

    parsed = urlparse(str(captured["url"]))
    query = parse_qs(parsed.query)
    assert payload == {"items": []}
    assert parsed.path == "/api/policies"
    assert query["namespace"] == ["image.prompts"]
    assert query["session_id"] == ["session-admin-1"]
    assert "empty" not in query
    assert captured["method"] == "GET"
    assert captured["body"] is None
    assert captured["timeout"] == 4.0


def test_fetch_mud_api_json_anonymous_uses_base_url_without_session_query():
    """Anonymous fetch should not append session_id and should send JSON body."""
    captured: dict[str, object] = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = request.data
        captured["timeout"] = timeout
        return _FakeResponse(b'{"session_id":"abc123"}')

    payload = mud_api_client.fetch_mud_api_json_anonymous(
        base_url="https://mud.example.com",
        method="POST",
        path="/login",
        body={"username": "admin", "password": "pw"},
        timeout_seconds=6.0,
        opener=opener,
    )

    parsed = urlparse(str(captured["url"]))
    query = parse_qs(parsed.query)
    assert payload == {"session_id": "abc123"}
    assert parsed.path == "/login"
    assert query == {}
    assert captured["method"] == "POST"
    assert captured["body"] == b'{"username": "admin", "password": "pw"}'
    assert captured["timeout"] == 6.0

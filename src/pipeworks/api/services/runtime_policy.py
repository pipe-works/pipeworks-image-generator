"""Runtime session + policy snippet loading service."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from secrets import token_urlsafe
from threading import RLock

from fastapi import Request, Response

from pipeworks.api.models import RuntimeAuthResponse
from pipeworks.api.runtime_mode import get_runtime_mode
from pipeworks.api.services.prompt_catalog import build_prompt_lookup
from pipeworks.core.config import PipeworksConfig

logger = logging.getLogger(__name__)

_POLICY_API_ROLE_REQUIRED_DETAIL = "Policy API requires admin or superuser role."
_SNIPPET_ALLOWED_ROLES = {"admin", "superuser"}
_SNIPPET_POLICY_TYPES = {
    "prompt",
    "species_block",
    "image_block",
    "clothing_block",
    "descriptor_layer",
    "registry",
    "tone_profile",
}

# Per-type override for the canonical content field that carries the
# prompt-injectable string. Default is ``content.text``; tone_profile
# uses ``content.prompt_block`` per mud-server validation.
_SNIPPET_TEXT_FIELD_OVERRIDES = {
    "tone_profile": "prompt_block",
}

_RUNTIME_SESSION_COOKIE_NAME = "pw_image_runtime_session"
_RUNTIME_SESSION_MAX_AGE_SECONDS = 12 * 60 * 60


@dataclass(slots=True)
class RuntimeBrowserSession:
    """Server-side runtime session binding for browser refresh persistence."""

    session_id: str
    mode_key: str
    server_url: str
    available_worlds: list[dict[str, object]]
    created_at_epoch: int
    updated_at_epoch: int


@dataclass(frozen=True, slots=True)
class MudApiRuntimeConfig:
    """Resolved mud-server API runtime configuration."""

    base_url: str
    session_id: str
    timeout_seconds: float = 8.0


FetchMudApiJson = Callable[..., dict[str, object]]
FetchMudApiJsonAnonymous = Callable[..., dict[str, object]]


class RuntimePolicyService:
    """Handle runtime login session state and canonical policy snippet loading."""

    def __init__(
        self,
        *,
        config: PipeworksConfig,
        default_mud_api_base_url: str,
        fetch_mud_api_json: FetchMudApiJson,
        fetch_mud_api_json_anonymous: FetchMudApiJsonAnonymous,
    ) -> None:
        self._config = config
        self._default_mud_api_base_url = default_mud_api_base_url
        self._fetch_mud_api_json = fetch_mud_api_json
        self._fetch_mud_api_json_anonymous = fetch_mud_api_json_anonymous

        self._runtime_browser_sessions: dict[str, RuntimeBrowserSession] = {}
        self._runtime_browser_sessions_lock = RLock()

    @staticmethod
    def _normalize_server_url_for_binding(value: str | None) -> str:
        return str(value or "").strip().rstrip("/")

    @staticmethod
    def _sanitize_available_worlds(
        world_rows: list[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        if not isinstance(world_rows, list):
            return []
        sanitized: list[dict[str, object]] = []
        for row in world_rows:
            if not isinstance(row, dict):
                continue
            world_id = str(row.get("id") or "").strip()
            if not world_id:
                continue
            normalized_row = dict(row)
            normalized_row["id"] = world_id
            world_name = str(row.get("name") or "").strip()
            if world_name:
                normalized_row["name"] = world_name
            sanitized.append(normalized_row)
        return sanitized

    @staticmethod
    def _runtime_cookie_secure(request: Request) -> bool:
        if request.url.scheme == "https":
            return True
        hostname = str(request.url.hostname or "").strip().lower()
        return hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}

    def set_runtime_session_cookie(
        self, response: Response, *, request: Request, token: str
    ) -> None:
        response.set_cookie(
            key=_RUNTIME_SESSION_COOKIE_NAME,
            value=token,
            max_age=_RUNTIME_SESSION_MAX_AGE_SECONDS,
            httponly=True,
            secure=self._runtime_cookie_secure(request),
            samesite="strict",
            path="/",
        )

    def clear_runtime_session_cookie(self, response: Response, *, request: Request) -> None:
        response.delete_cookie(
            key=_RUNTIME_SESSION_COOKIE_NAME,
            httponly=True,
            secure=self._runtime_cookie_secure(request),
            samesite="strict",
            path="/",
        )

    def _purge_expired_runtime_browser_sessions(self, *, now_epoch: int | None = None) -> None:
        now = int(now_epoch if now_epoch is not None else time.time())
        with self._runtime_browser_sessions_lock:
            expired_tokens = [
                token
                for token, record in self._runtime_browser_sessions.items()
                if now - record.updated_at_epoch >= _RUNTIME_SESSION_MAX_AGE_SECONDS
            ]
            for token in expired_tokens:
                self._runtime_browser_sessions.pop(token, None)

    def store_runtime_browser_session(
        self,
        *,
        mode_key: str,
        server_url: str | None,
        session_id: str,
        available_worlds: list[dict[str, object]] | None,
    ) -> str:
        now = int(time.time())
        token = token_urlsafe(32)
        record = RuntimeBrowserSession(
            session_id=session_id,
            mode_key=mode_key,
            server_url=self._normalize_server_url_for_binding(server_url),
            available_worlds=self._sanitize_available_worlds(available_worlds),
            created_at_epoch=now,
            updated_at_epoch=now,
        )
        with self._runtime_browser_sessions_lock:
            self._runtime_browser_sessions[token] = record
        self._purge_expired_runtime_browser_sessions(now_epoch=now)
        return token

    def pop_runtime_browser_session_by_token(
        self, token: str | None
    ) -> RuntimeBrowserSession | None:
        normalized_token = str(token or "").strip()
        if not normalized_token:
            return None
        with self._runtime_browser_sessions_lock:
            return self._runtime_browser_sessions.pop(normalized_token, None)

    def _resolve_runtime_browser_session(
        self,
        *,
        request: Request,
        mode_key: str,
        server_url: str | None,
    ) -> tuple[str | None, list[dict[str, object]], str | None]:
        self._purge_expired_runtime_browser_sessions()
        token = str(request.cookies.get(_RUNTIME_SESSION_COOKIE_NAME, "")).strip()
        if not token:
            return (None, [], None)
        with self._runtime_browser_sessions_lock:
            record = self._runtime_browser_sessions.get(token)
            if record is None:
                return (None, [], token)
            if (
                record.mode_key != mode_key
                or record.server_url != self._normalize_server_url_for_binding(server_url)
            ):
                self._runtime_browser_sessions.pop(token, None)
                return (None, [], token)
            record.updated_at_epoch = int(time.time())
            return (
                record.session_id,
                [dict(row) for row in record.available_worlds],
                token,
            )

    def _resolve_request_session_id(
        self,
        *,
        request: Request,
        mode_key: str,
        server_url: str | None,
        explicit_session_id: str | None,
    ) -> tuple[str | None, list[dict[str, object]], str | None]:
        normalized_explicit = str(explicit_session_id or "").strip()
        if normalized_explicit:
            return (normalized_explicit, [], None)
        return self._resolve_runtime_browser_session(
            request=request,
            mode_key=mode_key,
            server_url=server_url,
        )

    def resolve_mud_api_runtime_config(
        self,
        *,
        session_id_override: str | None,
        normalize_base_url: Callable[[str | None], str],
        base_url_override: str | None = None,
    ) -> MudApiRuntimeConfig:
        base_url = normalize_base_url(base_url_override or self._default_mud_api_base_url)
        if not base_url:
            raise ValueError("Mud API base URL must not be empty.")

        session_id = str(session_id_override or "").strip()
        if not session_id:
            raise ValueError("No active runtime session. Login with an admin/superuser account.")

        return MudApiRuntimeConfig(base_url=base_url, session_id=session_id, timeout_seconds=8.0)

    @staticmethod
    def extract_available_worlds_from_login_payload(
        payload: dict[str, object],
    ) -> list[dict[str, object]]:
        raw_worlds = payload.get("available_worlds")
        if not isinstance(raw_worlds, list):
            return []

        worlds: list[dict[str, object]] = []
        for world in raw_worlds:
            if not isinstance(world, dict):
                continue
            world_id = str(world.get("id") or "").strip()
            if not world_id:
                continue
            normalized_world = dict(world)
            normalized_world["id"] = world_id
            world_name = str(world.get("name") or "").strip()
            if world_name:
                normalized_world["name"] = world_name
            worlds.append(normalized_world)
        return worlds

    @staticmethod
    def classify_runtime_auth_probe_error(error_detail: str) -> tuple[str, str]:
        if _POLICY_API_ROLE_REQUIRED_DETAIL in error_detail:
            return ("forbidden", "Session is valid but role is not admin/superuser.")
        if "Invalid or expired session" in error_detail or "Invalid session user" in error_detail:
            return ("unauthenticated", "Session is invalid or expired.")
        return ("error", error_detail)

    def probe_runtime_auth(
        self,
        *,
        mode_key: str,
        source_kind: str,
        active_server_url: str | None,
        session_id_override: str | None,
        normalize_base_url: Callable[[str | None], str],
    ) -> RuntimeAuthResponse:
        if source_kind != "server_api":
            return RuntimeAuthResponse(
                mode_key=mode_key,
                source_kind=source_kind,
                active_server_url=active_server_url,
                session_present=False,
                access_granted=False,
                status="error",
                detail="Runtime mode must be server_api.",
                available_worlds=[],
            )

        try:
            runtime = self.resolve_mud_api_runtime_config(
                session_id_override=session_id_override,
                normalize_base_url=normalize_base_url,
                base_url_override=active_server_url,
            )
        except ValueError as exc:
            return RuntimeAuthResponse(
                mode_key=mode_key,
                source_kind=source_kind,
                active_server_url=active_server_url,
                session_present=False,
                access_granted=False,
                status="missing_session",
                detail=str(exc),
                available_worlds=[],
            )

        try:
            self._fetch_mud_api_json(
                runtime=runtime,
                method="GET",
                path="/api/policy-capabilities",
                query_params={},
            )
        except ValueError as exc:
            status, detail = self.classify_runtime_auth_probe_error(str(exc))
            return RuntimeAuthResponse(
                mode_key=mode_key,
                source_kind=source_kind,
                active_server_url=runtime.base_url,
                session_present=True,
                access_granted=False,
                status=status,
                detail=detail,
                available_worlds=[],
            )

        return RuntimeAuthResponse(
            mode_key=mode_key,
            source_kind=source_kind,
            active_server_url=runtime.base_url,
            session_present=True,
            access_granted=True,
            status="authorized",
            detail="Session is authorized for admin/superuser policy APIs.",
            available_worlds=[],
        )

    @staticmethod
    def format_policy_option_label(policy_key: str, variant: str) -> str:
        normalized_key = policy_key.replace("_", " ").replace("-", " ").strip()
        key_label = " ".join(part.capitalize() for part in normalized_key.split()) or policy_key
        variant_label = str(variant or "").strip()
        if variant_label:
            return f"{key_label} ({variant_label})"
        return key_label

    @staticmethod
    def extract_policy_prompt_text(policy_item: dict[str, object]) -> str:
        content = policy_item.get("content")
        if not isinstance(content, dict):
            return ""
        policy_type = str(policy_item.get("policy_type") or "").strip()
        field_name = _SNIPPET_TEXT_FIELD_OVERRIDES.get(policy_type, "text")
        text_value = content.get(field_name)
        if not isinstance(text_value, str):
            return ""
        return text_value.strip()

    def load_policy_prompt_options(
        self,
        *,
        active_server_url: str | None,
        session_id: str | None,
        normalize_base_url: Callable[[str | None], str],
    ) -> list[dict]:
        try:
            runtime = self.resolve_mud_api_runtime_config(
                session_id_override=session_id,
                normalize_base_url=normalize_base_url,
                base_url_override=active_server_url,
            )
        except ValueError:
            return []

        try:
            payload = self._fetch_mud_api_json(
                runtime=runtime,
                method="GET",
                path="/api/policies",
                query_params={},
            )
        except ValueError as exc:
            logger.warning("Unable to load policy snippets from mud-server API: %s", exc)
            return []

        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            return []

        options: list[dict] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            policy_type = str(item.get("policy_type") or "").strip()
            if policy_type not in _SNIPPET_POLICY_TYPES:
                continue

            text_value = self.extract_policy_prompt_text(item)
            if not text_value:
                continue

            policy_id = str(item.get("policy_id") or "").strip()
            variant = str(item.get("variant") or "").strip()
            policy_key = str(item.get("policy_key") or "").strip()
            namespace = str(item.get("namespace") or "").strip()
            if not policy_id or not variant or not policy_key:
                continue

            option_id = f"{policy_id}:{variant}"
            group = namespace or policy_type
            options.append(
                {
                    "id": option_id,
                    "label": self.format_policy_option_label(policy_key, variant),
                    "value": text_value,
                    "group": group,
                    "path": option_id,
                }
            )

        options.sort(key=lambda option: (option.get("group", ""), option.get("label", "")))
        return options

    @staticmethod
    def load_policy_prompt_groups(options: list[dict]) -> list[str]:
        return sorted(
            {str(option.get("group") or "").strip() for option in options if option.get("group")}
        )

    def load_policy_prompts_for_request(
        self,
        *,
        request: Request,
        response: Response | None,
        explicit_session_id: str | None,
        normalize_base_url: Callable[[str | None], str],
    ) -> tuple[list[dict], list[str], RuntimeAuthResponse]:
        state = get_runtime_mode()
        resolved_session_id, cookie_worlds, cookie_token = self._resolve_request_session_id(
            request=request,
            mode_key=state.mode_key,
            server_url=state.active_server_url,
            explicit_session_id=explicit_session_id,
        )

        runtime_auth = self.probe_runtime_auth(
            mode_key=state.mode_key,
            source_kind=state.source_kind,
            active_server_url=state.active_server_url,
            session_id_override=resolved_session_id,
            normalize_base_url=normalize_base_url,
        )

        if cookie_token and runtime_auth.status in {"missing_session", "unauthenticated"}:
            self.pop_runtime_browser_session_by_token(cookie_token)
            if response is not None:
                self.clear_runtime_session_cookie(response, request=request)

        if runtime_auth.access_granted and cookie_worlds:
            runtime_auth = runtime_auth.model_copy(update={"available_worlds": cookie_worlds})

        if not runtime_auth.access_granted:
            return ([], [], runtime_auth)

        policy_prompt_options = self.load_policy_prompt_options(
            active_server_url=state.active_server_url,
            session_id=resolved_session_id,
            normalize_base_url=normalize_base_url,
        )
        policy_prompt_groups = self.load_policy_prompt_groups(policy_prompt_options)
        return (policy_prompt_options, policy_prompt_groups, runtime_auth)

    def load_policy_prompt_lookup(
        self,
        *,
        prompts: dict,
        request: Request,
        normalize_base_url: Callable[[str | None], str],
    ) -> tuple[dict[str, dict], list[dict], RuntimeAuthResponse]:
        options, _, runtime_auth = self.load_policy_prompts_for_request(
            request=request,
            response=None,
            explicit_session_id=None,
            normalize_base_url=normalize_base_url,
        )
        lookup = build_prompt_lookup(prompts, options)
        return lookup, options, runtime_auth

    def fetch_mud_api_json_anonymous(
        self,
        *,
        base_url: str,
        method: str,
        path: str,
        body: dict[str, object] | None,
    ) -> dict[str, object]:
        """Issue one mud-server API request without session query injection."""
        return self._fetch_mud_api_json_anonymous(
            base_url=base_url,
            method=method,
            path=path,
            body=body,
        )

    def sanitize_available_worlds(
        self,
        world_rows: list[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        """Public wrapper around world-row normalization."""
        return self._sanitize_available_worlds(world_rows)

    @property
    def snippet_allowed_roles(self) -> set[str]:
        return set(_SNIPPET_ALLOWED_ROLES)

    @property
    def runtime_session_cookie_name(self) -> str:
        return _RUNTIME_SESSION_COOKIE_NAME

    def login_runtime_session(
        self,
        *,
        mode_key: str,
        server_url: str | None,
        session_id: str,
        available_worlds: list[dict[str, object]] | None,
    ) -> str:
        return self.store_runtime_browser_session(
            mode_key=mode_key,
            server_url=server_url,
            session_id=session_id,
            available_worlds=available_worlds,
        )

    def clear_cookie_session_from_request(self, *, request: Request) -> None:
        session_token = str(request.cookies.get(_RUNTIME_SESSION_COOKIE_NAME, "")).strip()
        if session_token:
            self.pop_runtime_browser_session_by_token(session_token)

    @property
    def runtime_browser_sessions(self) -> dict[str, RuntimeBrowserSession]:
        """Expose in-memory browser session mapping for compatibility tests."""
        return self._runtime_browser_sessions

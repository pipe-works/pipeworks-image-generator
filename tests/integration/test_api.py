"""Integration tests for pipeworks.api.main — FastAPI REST API endpoints.

All tests use the FastAPI TestClient with a mocked ModelManager so that no
real model loading or GPU access occurs.  Tests cover every endpoint:

- ``GET /`` — HTML page serving.
- ``GET /api/config`` — Configuration delivery.
- ``POST /api/generate`` — Batch image generation.
- ``POST /api/prompt/compile`` — Prompt preview.
- ``GET /api/gallery`` — Paginated gallery listing with filters.
- ``GET /api/gallery/{id}`` — Single gallery entry.
- ``GET /api/gallery/{id}/prompt`` — Prompt metadata.
- ``GET /api/gallery/{id}/zip`` — Zip archive download.
- ``POST /api/gallery/favourite`` — Favourite toggling.
- ``DELETE /api/gallery/{id}`` — Image deletion.
- ``GET /api/stats`` — Gallery statistics.
"""

from __future__ import annotations

import json
import os
import threading
import time
from base64 import b64encode
from io import BytesIO
from unittest.mock import patch


class SequenceRandom:
    """Minimal deterministic RNG stub for placeholder integration tests."""

    def __init__(self, picks: list[str]):
        self._picks = list(picks)

    def choice(self, options: list[str]) -> str:
        pick = self._picks.pop(0)
        assert pick in options
        return pick


# ---------------------------------------------------------------------------
# Index page tests.
# ---------------------------------------------------------------------------


class TestIndexPage:
    """Test GET / — main HTML page."""

    def test_index_returns_html(self, test_client):
        """GET / should return 200 with HTML content."""
        resp = test_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Pipe-Works Image Generator" in resp.text


# ---------------------------------------------------------------------------
# Configuration endpoint tests.
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Test GET /api/config — application configuration."""

    def test_config_returns_version(self, test_client):
        """Response should include a version string."""
        resp = test_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data

    def test_config_returns_models(self, test_client):
        """Response should include a models list."""
        resp = test_client.get("/api/config")
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) >= 1

    def test_config_returns_gpu_worker_metadata_without_tokens(self, test_client):
        """GPU worker metadata should be exposed without secret bearer tokens."""
        from pipeworks.api import main as main_module
        from pipeworks.core.config import GpuWorkerConfig

        previous_workers = main_module.config.gpu_workers
        previous_default = main_module.config.default_gpu_worker_id
        main_module.config.gpu_workers = [
            GpuWorkerConfig(id="local", label="Luminal GPU", mode="local", enabled=True),
            GpuWorkerConfig(
                id="remote-1",
                label="Remote GPU 1",
                mode="remote",
                base_url="https://gpu.example.com",
                bearer_token="secret-token",
                enabled=True,
            ),
        ]
        main_module.config.default_gpu_worker_id = "remote-1"
        try:
            resp = test_client.get("/api/config")
            assert resp.status_code == 200
            data = resp.json()
            assert data["default_gpu_worker_id"] == "remote-1"
            assert data["gpu_workers"] == [
                {"id": "local", "label": "Luminal GPU", "mode": "local", "enabled": True},
                {"id": "remote-1", "label": "Remote GPU 1", "mode": "remote", "enabled": True},
            ]
            assert "bearer_token" not in json.dumps(data)
        finally:
            main_module.config.gpu_workers = previous_workers
            main_module.config.default_gpu_worker_id = previous_default

    def test_config_exposes_z_image_prompt_limit(self, test_client):
        """Z-Image should expose its Qwen-based 512-token prompt budget."""
        resp = test_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        z_image = next(model for model in data["models"] if model["id"] == "z-image-turbo")
        assert z_image["max_prompt_tokens"] == 512

    def test_config_exposes_flux2_klein_model(self, test_client):
        """FLUX.2-klein-4B should be listed with support metadata."""
        resp = test_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        flux_model = next(model for model in data["models"] if model["id"] == "flux-2-klein-4b")
        assert flux_model["hf_id"] == "black-forest-labs/FLUX.2-klein-4B"
        assert flux_model["max_prompt_tokens"] == 512
        assert flux_model["default_guidance"] == 0.0
        assert flux_model["min_guidance"] == 0.0
        assert flux_model["max_guidance"] == 0.0
        assert flux_model["supports_negative_prompt"] is False
        assert "is_available" in flux_model
        assert "unavailable_reason" in flux_model

    def test_config_exposes_negative_prompt_support_per_model(self, test_client):
        """Negative prompt support metadata should match actual model capability."""
        resp = test_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        models_by_id = {model["id"]: model for model in data["models"]}

        assert models_by_id["sd-v1-5"]["supports_negative_prompt"] is True
        assert models_by_id["sdxl-1-0"]["supports_negative_prompt"] is True
        assert models_by_id["z-image-turbo"]["supports_negative_prompt"] is False
        assert models_by_id["flux-2-klein-4b"]["supports_negative_prompt"] is False

    def test_config_returns_prompts(self, test_client):
        """Response should include all three prompt categories."""
        resp = test_client.get("/api/config")
        data = resp.json()
        assert "prepend_prompts" in data
        assert "automated_prompts" in data
        assert "append_prompts" in data

    def test_config_returns_split_prompt_libraries(self, test_client):
        """Response should expose the three split prompt-library files."""
        resp = test_client.get("/api/config")
        data = resp.json()
        assert "prepend_library" in data
        assert "main_library" in data
        assert "append_library" in data

    def test_config_returns_policy_prompt_options(self, test_client):
        """Without runtime login, policy-backed snippet options should be empty."""
        resp = test_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "policy_prompt_options" in data
        assert isinstance(data["policy_prompt_options"], list)
        assert data["policy_prompt_options"] == []
        assert data["runtime_auth"]["status"] in {"missing_session", "unauthenticated"}

    def test_config_returns_policy_prompt_groups(self, test_client):
        """Without runtime login, policy snippet groups should be empty."""
        resp = test_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "policy_prompt_groups" in data
        assert isinstance(data["policy_prompt_groups"], list)
        assert data["policy_prompt_groups"] == []

    def test_config_returns_policy_prompt_slot_kinds(self, test_client):
        """Without runtime login, policy snippet slot kinds should be empty."""
        resp = test_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "policy_prompt_slot_kinds" in data
        assert isinstance(data["policy_prompt_slot_kinds"], list)
        assert data["policy_prompt_slot_kinds"] == []

    def test_runtime_mode_switch_accepts_server_url_override(self, test_client):
        """Runtime mode endpoint should accept explicit dev/prod URL overrides."""
        resp = test_client.post(
            "/api/runtime-mode",
            json={"mode_key": "server_prod", "server_url": "https://mud.example.com/"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["mode_key"] == "server_prod"
        assert payload["active_server_url"] == "https://mud.example.com"

    def test_runtime_auth_reports_missing_session_without_login(self, test_client):
        """Runtime auth endpoint should fail closed until a runtime session exists."""
        resp = test_client.get("/api/runtime-auth")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["session_present"] is False
        assert payload["access_granted"] is False
        assert payload["status"] in {"missing_session", "unauthenticated"}

    def test_policy_prompts_load_from_canonical_api_after_runtime_login(self, test_client):
        """Snippet dropdown payload should come from canonical policy APIs after login."""

        def _fake_login_fetch(*, base_url, method, path, body):
            assert base_url == "http://127.0.0.1:18000"
            assert method == "POST"
            assert path == "/login"
            assert body == {"username": "admin", "password": "pw"}
            return {
                "session_id": "session-admin-1",
                "role": "admin",
                "available_worlds": [{"id": "pipeworks_web", "name": "Pipeworks Web"}],
            }

        def _fake_authenticated_fetch(*, runtime, method, path, query_params, json_payload=None):
            assert runtime.base_url == "http://127.0.0.1:18000"
            assert runtime.session_id == "session-admin-1"
            assert method == "GET"
            assert json_payload is None
            assert query_params == {}
            if path == "/api/policy-capabilities":
                return {
                    "allowed_policy_types": ["prompt", "species_block", "registry"],
                    "allowed_statuses": ["draft", "active"],
                }
            if path == "/api/policies":
                return {
                    "items": [
                        {
                            "policy_id": "prompt:image.prompts.creatures:goblin_workshop",
                            "policy_type": "prompt",
                            "namespace": "image.prompts.creatures",
                            "policy_key": "goblin_workshop",
                            "variant": "v1",
                            "content": {"text": "A goblin workshop scene."},
                        },
                        {
                            "policy_id": "species_block:image.blocks.species:goblin",
                            "policy_type": "species_block",
                            "namespace": "image.blocks.species",
                            "policy_key": "goblin",
                            "variant": "v2",
                            "content": {"text": "A goblin of pipe-works canon."},
                        },
                        {
                            "policy_id": "registry:image.registries:species_registry",
                            "policy_type": "registry",
                            "namespace": "image.registries",
                            "policy_key": "species_registry",
                            "variant": "v1",
                            "content": {"references": []},
                        },
                    ]
                }
            raise AssertionError(f"Unexpected path: {path}")

        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous", side_effect=_fake_login_fetch
            ),
            patch("pipeworks.api.main._fetch_mud_api_json", side_effect=_fake_authenticated_fetch),
        ):
            login_resp = test_client.post(
                "/api/runtime-login",
                json={"username": "admin", "password": "pw"},
            )
            assert login_resp.status_code == 200
            assert login_resp.json()["success"] is True

            snippets_resp = test_client.get("/api/policy-prompts")
            assert snippets_resp.status_code == 200
            payload = snippets_resp.json()
            assert payload["runtime_auth"]["status"] == "authorized"
            assert payload["runtime_auth"]["access_granted"] is True
            assert payload["policy_prompt_groups"] == [
                "image.blocks.species",
                "image.prompts.creatures",
            ]

            options_by_id = {option["id"]: option for option in payload["policy_prompt_options"]}
            assert "prompt:image.prompts.creatures:goblin_workshop:v1" in options_by_id
            assert "species_block:image.blocks.species:goblin:v2" in options_by_id
            assert "registry:image.registries:species_registry:v1" not in options_by_id
            assert (
                options_by_id["species_block:image.blocks.species:goblin:v2"]["value"]
                == "A goblin of pipe-works canon."
            )
            # slot_kinds default to [namespace] when mud-server omits the field.
            assert options_by_id["species_block:image.blocks.species:goblin:v2"]["slot_kinds"] == [
                "image.blocks.species"
            ]
            assert payload["policy_prompt_slot_kinds"] == [
                "image.blocks.species",
                "image.prompts.creatures",
            ]

    def test_policy_prompts_honor_explicit_slot_kinds_field(self, test_client):
        """Top-level mud-server slot_kinds should be surfaced verbatim to clients."""

        def _fake_login_fetch(*, base_url, method, path, body):
            return {
                "session_id": "session-admin-1",
                "role": "admin",
                "available_worlds": [{"id": "pipeworks_web", "name": "Pipeworks Web"}],
            }

        def _fake_authenticated_fetch(*, runtime, method, path, query_params, json_payload=None):
            if path == "/api/policy-capabilities":
                return {
                    "allowed_policy_types": ["tone_profile"],
                    "allowed_statuses": ["draft", "active"],
                }
            if path == "/api/policies":
                return {
                    "items": [
                        {
                            "policy_id": "tone_profile:image.tone_profiles:ledger_engraving",
                            "policy_type": "tone_profile",
                            "namespace": "image.tone_profiles",
                            "policy_key": "ledger_engraving",
                            "variant": "v1",
                            "slot_kinds": ["atmosphere", "lighting"],
                            "content": {"prompt_block": "Etched ledger lines."},
                        },
                    ]
                }
            raise AssertionError(f"Unexpected path: {path}")

        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous", side_effect=_fake_login_fetch
            ),
            patch("pipeworks.api.main._fetch_mud_api_json", side_effect=_fake_authenticated_fetch),
        ):
            login_resp = test_client.post(
                "/api/runtime-login",
                json={"username": "admin", "password": "pw"},
            )
            assert login_resp.status_code == 200

            snippets_resp = test_client.get("/api/policy-prompts")
            assert snippets_resp.status_code == 200
            payload = snippets_resp.json()
            options_by_id = {option["id"]: option for option in payload["policy_prompt_options"]}
            tone_id = "tone_profile:image.tone_profiles:ledger_engraving:v1"
            assert options_by_id[tone_id]["slot_kinds"] == ["atmosphere", "lighting"]
            assert payload["policy_prompt_slot_kinds"] == ["atmosphere", "lighting"]

    def test_tone_profile_snippet_uses_prompt_block_field(self, test_client):
        """tone_profile snippets pull text from canonical ``content.prompt_block``."""

        def _fake_login_fetch(*, base_url, method, path, body):
            return {
                "session_id": "session-admin-1",
                "role": "admin",
                "available_worlds": [{"id": "pipeworks_web", "name": "Pipeworks Web"}],
            }

        def _fake_authenticated_fetch(*, runtime, method, path, query_params, json_payload=None):
            if path == "/api/policy-capabilities":
                return {
                    "allowed_policy_types": ["prompt", "tone_profile"],
                    "allowed_statuses": ["draft", "active"],
                }
            if path == "/api/policies":
                return {
                    "items": [
                        {
                            "policy_id": "tone_profile:image.tone_profiles:ledger_engraving",
                            "policy_type": "tone_profile",
                            "namespace": "image.tone_profiles",
                            "policy_key": "ledger_engraving",
                            "variant": "v1",
                            "content": {
                                "name": "ledger_engraving_v1",
                                "prompt_block": "Linocut style with muted sepia palette.",
                            },
                        },
                        {
                            "policy_id": "tone_profile:image.tone_profiles:no_payload",
                            "policy_type": "tone_profile",
                            "namespace": "image.tone_profiles",
                            "policy_key": "no_payload",
                            "variant": "v1",
                            "content": {"name": "no_payload_v1"},
                        },
                    ]
                }
            raise AssertionError(f"Unexpected path: {path}")

        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous", side_effect=_fake_login_fetch
            ),
            patch("pipeworks.api.main._fetch_mud_api_json", side_effect=_fake_authenticated_fetch),
        ):
            login_resp = test_client.post(
                "/api/runtime-login",
                json={"username": "admin", "password": "pw"},
            )
            assert login_resp.status_code == 200

            snippets_resp = test_client.get("/api/policy-prompts")
            assert snippets_resp.status_code == 200
            payload = snippets_resp.json()

            options_by_id = {option["id"]: option for option in payload["policy_prompt_options"]}
            tone_id = "tone_profile:image.tone_profiles:ledger_engraving:v1"
            assert tone_id in options_by_id
            assert options_by_id[tone_id]["value"] == "Linocut style with muted sepia palette."
            # tone_profile without a prompt_block field is silently dropped.
            assert "tone_profile:image.tone_profiles:no_payload:v1" not in options_by_id

    def test_location_snippet_uses_default_text_field(self, test_client):
        """location snippets pull text from canonical ``content.text``.

        Confirms the canonical ``location`` policy type flows through the
        prompt-snippet pipeline using the default text-bearing field, with
        no per-type override required, and that locations missing
        ``content.text`` are silently dropped.
        """

        def _fake_login_fetch(*, base_url, method, path, body):
            return {
                "session_id": "session-admin-1",
                "role": "admin",
                "available_worlds": [{"id": "pipeworks_web", "name": "Pipeworks Web"}],
            }

        def _fake_authenticated_fetch(*, runtime, method, path, query_params, json_payload=None):
            if path == "/api/policy-capabilities":
                return {
                    "allowed_policy_types": ["prompt", "location"],
                    "allowed_statuses": ["draft", "active"],
                }
            if path == "/api/policies":
                return {
                    "items": [
                        {
                            "policy_id": "location:image.locations.environment:cozy_inn",
                            "policy_type": "location",
                            "namespace": "image.locations.environment",
                            "policy_key": "cozy_inn",
                            "variant": "v1",
                            "content": {
                                "text": ("A warm timber-beamed inn lit by hearth firelight."),
                            },
                        },
                        {
                            "policy_id": "location:image.locations.environment:no_payload",
                            "policy_type": "location",
                            "namespace": "image.locations.environment",
                            "policy_key": "no_payload",
                            "variant": "v1",
                            "content": {"description": "missing canonical text field"},
                        },
                    ]
                }
            raise AssertionError(f"Unexpected path: {path}")

        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous", side_effect=_fake_login_fetch
            ),
            patch("pipeworks.api.main._fetch_mud_api_json", side_effect=_fake_authenticated_fetch),
        ):
            login_resp = test_client.post(
                "/api/runtime-login",
                json={"username": "admin", "password": "pw"},
            )
            assert login_resp.status_code == 200

            snippets_resp = test_client.get("/api/policy-prompts")
            assert snippets_resp.status_code == 200
            payload = snippets_resp.json()

            options_by_id = {option["id"]: option for option in payload["policy_prompt_options"]}
            location_id = "location:image.locations.environment:cozy_inn:v1"
            assert location_id in options_by_id
            assert (
                options_by_id[location_id]["value"]
                == "A warm timber-beamed inn lit by hearth firelight."
            )
            # location without a text field is silently dropped.
            assert "location:image.locations.environment:no_payload:v1" not in options_by_id

    def test_disable_http_cache_adds_no_cache_headers(self, test_client):
        """No-cache headers should be emitted when local dev mode enables them."""
        from pipeworks.api import main as main_module

        previous = main_module.config.disable_http_cache
        main_module.config.disable_http_cache = True
        try:
            resp = test_client.get("/api/config")
            assert resp.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
            assert resp.headers["pragma"] == "no-cache"
            assert resp.headers["expires"] == "0"
        finally:
            main_module.config.disable_http_cache = previous


# ---------------------------------------------------------------------------
# GPU settings endpoint tests.
# ---------------------------------------------------------------------------


class TestGpuSettings:
    """Test runtime-editable GPU settings endpoints."""

    def test_get_gpu_settings_defaults_to_local_mode(self, test_client):
        """GPU settings endpoint should default to local-only mode."""
        resp = test_client.get("/api/gpu-settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["use_remote_gpu"] is False
        assert data["default_gpu_worker_id"] == "local"
        assert data["has_bearer_token"] is False

    def test_update_gpu_settings_enables_remote_and_updates_api_config(self, test_client):
        """Saving remote settings should update worker list exposed by /api/config."""
        save_resp = test_client.post(
            "/api/gpu-settings",
            json={
                "use_remote_gpu": True,
                "remote_base_url": "https://gpu-worker.example",
                "bearer_token": "test-token",
                "remote_label": "Remote Worker",
            },
        )
        assert save_resp.status_code == 200
        payload = save_resp.json()
        assert payload["use_remote_gpu"] is True
        assert payload["has_bearer_token"] is True

        config_resp = test_client.get("/api/config")
        assert config_resp.status_code == 200
        config_payload = config_resp.json()
        workers_by_id = {worker["id"]: worker for worker in config_payload["gpu_workers"]}
        assert "remote-ts" in workers_by_id
        assert workers_by_id["remote-ts"]["label"] == "Remote Worker"
        assert "bearer_token" not in json.dumps(config_payload)

    def test_update_gpu_settings_generates_token_when_missing(self, test_client):
        """Saving remote settings without a token should generate one."""
        test_client.post(
            "/api/gpu-settings",
            json={
                "use_remote_gpu": False,
            },
        )
        resp = test_client.post(
            "/api/gpu-settings",
            json={
                "use_remote_gpu": True,
                "remote_base_url": "https://gpu-worker.example",
                "bearer_token": None,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["use_remote_gpu"] is True
        assert data["has_bearer_token"] is True
        assert isinstance(data.get("generated_bearer_token"), str)
        assert len(data["generated_bearer_token"]) > 20

    def test_gpu_settings_test_connection_uses_saved_token_when_not_provided(self, test_client):
        """Health test endpoint should fall back to saved token when form token is blank."""
        test_client.post(
            "/api/gpu-settings",
            json={
                "use_remote_gpu": True,
                "remote_base_url": "https://gpu-worker.example",
                "bearer_token": "saved-token",
            },
        )

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"success": true, "status": "ok"}'

        def _fake_urlopen(request, timeout):
            assert request.full_url == "https://gpu-worker.example/api/worker/health"
            assert request.get_header("Authorization") == "Bearer saved-token"
            assert timeout == 8
            return _FakeResponse()

        with patch("pipeworks.api.main.urlopen", side_effect=_fake_urlopen):
            resp = test_client.post(
                "/api/gpu-settings/test",
                json={
                    "remote_base_url": "https://gpu-worker.example",
                    "bearer_token": None,
                    "timeout_seconds": 8,
                },
            )

        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_update_gpu_settings_writes_runtime_file_with_strict_permissions(self, test_client):
        """Persisted runtime GPU settings should be owner-only on POSIX hosts."""
        if os.name != "posix":
            return

        from pipeworks.api import main as main_module

        resp = test_client.post(
            "/api/gpu-settings",
            json={
                "use_remote_gpu": True,
                "remote_base_url": "https://gpu-worker.example",
                "bearer_token": "test-token",
            },
        )
        assert resp.status_code == 200
        assert main_module.GPU_SETTINGS_DB.exists() is True
        mode = main_module.GPU_SETTINGS_DB.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# Generation endpoint tests.
# ---------------------------------------------------------------------------


class TestGenerate:
    """Test POST /api/generate — image generation."""

    def _make_generate_payload(self, **overrides) -> dict:
        """Build a valid v3 generate request payload with optional overrides."""
        sections_override = overrides.pop("sections", None)
        payload = {
            "model_id": "z-image-turbo",
            "prompt_schema_version": 3,
            "sections": sections_override
            or [
                {
                    "label": "Subject",
                    "mode": "manual",
                    "manual_text": "A goblin workshop.",
                }
            ],
            "aspect_ratio_id": "1:1",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 0.0,
            "batch_size": 1,
        }
        payload.update(overrides)
        return payload

    def test_generate_success(self, test_client):
        """A valid generate request should return 200 with image data."""
        resp = test_client.post("/api/generate", json=self._make_generate_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["images"]) == 1
        assert "compiled_prompt" in data

    def test_generate_success_includes_compute_target_metadata(self, test_client):
        """Local generation should persist compute target id/label metadata."""
        resp = test_client.post("/api/generate", json=self._make_generate_payload())
        assert resp.status_code == 200
        image = resp.json()["images"][0]
        assert image["compute_target_id"] == "local"
        assert image["compute_target_label"] == "Luminal GPU"

    def test_generate_remote_worker_success(self, test_client, mock_model_manager):
        """Remote worker mode should save returned images to local gallery metadata."""
        from PIL import Image

        settings_resp = test_client.post(
            "/api/gpu-settings",
            json={
                "use_remote_gpu": True,
                "remote_label": "Remote A",
                "remote_base_url": "https://worker.example.com",
                "bearer_token": "worker-token",
            },
        )
        assert settings_resp.status_code == 200

        buffer = BytesIO()
        Image.new("RGB", (1024, 1024), color=(1, 2, 3)).save(buffer, format="PNG")
        png_payload = b64encode(buffer.getvalue()).decode("utf-8")

        with patch(
            "pipeworks.api.main._post_json_with_bearer",
            return_value={
                "success": True,
                "cancelled": False,
                "completed_count": 1,
                "results": [{"index": 0, "seed": 123, "png_base64": png_payload}],
            },
        ) as worker_post:
            resp = test_client.post(
                "/api/generate",
                json=self._make_generate_payload(
                    gpu_worker_id="remote-ts",
                    generation_id="gen-remote-success",
                    seed=123,
                ),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["completed_count"] == 1
        assert data["images"][0]["compute_target_id"] == "remote-ts"
        assert data["images"][0]["compute_target_label"] == "Remote A"
        assert mock_model_manager.generate.call_count == 0
        worker_post.assert_called_once()

    def test_generate_remote_worker_unreachable_returns_clear_error(self, test_client):
        """Remote worker transport failures should fail fast with worker label detail."""
        settings_resp = test_client.post(
            "/api/gpu-settings",
            json={
                "use_remote_gpu": True,
                "remote_label": "Remote B",
                "remote_base_url": "https://worker.example.com",
                "bearer_token": "worker-token",
            },
        )
        assert settings_resp.status_code == 200

        with patch(
            "pipeworks.api.main._post_json_with_bearer",
            side_effect=ValueError("timed out"),
        ):
            resp = test_client.post(
                "/api/generate",
                json=self._make_generate_payload(gpu_worker_id="remote-ts"),
            )

        assert resp.status_code == 502
        assert "Remote B" in resp.json()["detail"]
        gallery_resp = test_client.get("/api/gallery")
        assert gallery_resp.status_code == 200
        assert gallery_resp.json()["total"] == 0

    def test_generate_batch(self, test_client):
        """Batch generation should return the correct number of images."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(batch_size=3),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["images"]) == 3

    def test_generate_seeds_increment(self, test_client):
        """Each image in a batch should have an incrementing seed."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(batch_size=3, seed=100),
        )
        data = resp.json()
        seeds = [img["seed"] for img in data["images"]]
        assert seeds == [100, 101, 102]

    def test_generate_unknown_model(self, test_client):
        """An unknown model_id should return 400."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(model_id="nonexistent-model"),
        )
        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["detail"]

    def test_generate_unavailable_flux_model_returns_503(self, test_client):
        """Unsupported runtime models should fail cleanly instead of 500ing."""
        with patch(
            "pipeworks.api.main.get_model_runtime_support",
            return_value=(False, "Flux2KleinPipeline missing"),
        ):
            resp = test_client.post(
                "/api/generate",
                json={
                    "model_id": "flux-2-klein-4b",
                    "prompt_schema_version": 3,
                    "sections": [
                        {
                            "label": "Subject",
                            "mode": "manual",
                            "manual_text": "A test scene.",
                        }
                    ],
                    "aspect_ratio_id": "1:1",
                    "width": 1024,
                    "height": 1024,
                    "steps": 28,
                    "guidance": 4.0,
                    "batch_size": 1,
                },
            )
        assert resp.status_code == 503
        assert "Flux2KleinPipeline" in resp.json()["detail"]

    def test_generate_manual_blank_section_text(self, test_client):
        """Sections with empty text should be silently dropped from compiled prompt."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                sections=[
                    {"label": "Subject", "mode": "manual", "manual_text": ""},
                ],
            ),
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_generate_automated_section_with_valid_preset(self, test_client):
        """Automated mode with a valid preset id should succeed."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                sections=[
                    {
                        "label": "Scene",
                        "mode": "automated",
                        "automated_prompt_id": "goblin-workshop",
                    }
                ],
            ),
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_generate_automated_section_missing_id(self, test_client):
        """Automated mode without preset id should generate without that section's text."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                sections=[
                    {"label": "Scene", "mode": "automated", "automated_prompt_id": None},
                    {"label": "Subject", "mode": "manual", "manual_text": "A goblin."},
                ],
            ),
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_generate_rejects_legacy_v2_payload(self, test_client):
        """Legacy v2 payloads should be rejected by Pydantic validation."""
        legacy_payload = {
            "model_id": "z-image-turbo",
            "prompt_schema_version": 2,
            "subject_mode": "manual",
            "manual_subject": "A legacy scene.",
            "aspect_ratio_id": "1:1",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 0.0,
            "batch_size": 1,
        }
        resp = test_client.post("/api/generate", json=legacy_payload)
        assert resp.status_code == 422

    def test_generate_dynamic_v3_payload(self, test_client):
        """A v3 dynamic-section generate payload should produce a successful batch."""
        resp = test_client.post(
            "/api/generate",
            json={
                "model_id": "z-image-turbo",
                "prompt_schema_version": 3,
                "sections": [
                    {
                        "label": "Subject",
                        "mode": "manual",
                        "manual_text": "A goblin workshop.",
                    },
                    {
                        "label": "Tone",
                        "mode": "manual",
                        "manual_text": "Sepia.",
                    },
                ],
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
                "batch_size": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["images"]) == 1
        compiled = data["compiled_prompt"]
        assert compiled.index("Subject:") < compiled.index("Tone:")
        assert "A goblin workshop." in compiled
        assert "Sepia." in compiled

    def test_generate_invalid_batch_size_zero(self, test_client):
        """batch_size of 0 should return 400."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(batch_size=0),
        )
        assert resp.status_code == 400

    def test_generate_invalid_batch_size_too_large(self, test_client):
        """batch_size above 1000 should return 400."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(batch_size=1001),
        )
        assert resp.status_code == 400

    def test_generate_batch_size_at_upper_limit(self, test_client):
        """batch_size of 1000 should be accepted."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(batch_size=1000),
        )
        assert resp.status_code == 200

    def test_generate_with_automated_section_resolves_preset(self, test_client):
        """An automated section with a valid preset id should resolve into compiled prompt."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                sections=[
                    {
                        "label": "Scene",
                        "mode": "automated",
                        "automated_prompt_id": "goblin-workshop",
                    }
                ],
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "clockwork automaton" in data["compiled_prompt"].lower()

    def test_generate_automated_section_can_use_append_library_prompt(self, test_client):
        """Section automation should accept a prompt sourced from append.json."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                sections=[
                    {
                        "label": "Quality",
                        "mode": "automated",
                        "automated_prompt_id": "high-detail",
                    }
                ],
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "award-winning quality" in data["compiled_prompt"].lower()

    def test_generate_multiple_manual_sections(self, test_client):
        """Multiple manual sections should each appear in compiled prompt in order."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                sections=[
                    {"label": "Setting", "mode": "manual", "manual_text": "Ink sketch style."},
                    {"label": "Details", "mode": "manual", "manual_text": "High contrast."},
                ],
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        compiled = data["compiled_prompt"]
        assert "Ink sketch style." in compiled
        assert "High contrast." in compiled
        assert compiled.index("Setting:") < compiled.index("Details:")

    def test_generate_with_scheduler(self, test_client):
        """Scheduler should be passed through and stored in gallery metadata."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(scheduler="dpmpp-2m-karras"),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["images"][0]["scheduler"] == "dpmpp-2m-karras"

    def test_generate_without_scheduler(self, test_client):
        """Omitting scheduler should store None in gallery metadata."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["images"][0]["scheduler"] is None

    def test_generate_batch_reexpands_placeholders_for_each_image(
        self,
        test_client,
        mock_model_manager,
    ):
        """Each image in a batch should get a fresh placeholder expansion."""
        with patch(
            "pipeworks.api.prompt_builder._PLACEHOLDER_RANDOM",
            SequenceRandom(["red", "blue", "green"]),
        ):
            resp = test_client.post(
                "/api/generate",
                json=self._make_generate_payload(
                    sections=[
                        {
                            "label": "Subject",
                            "mode": "manual",
                            "manual_text": "A {red|blue|green} automaton.",
                        }
                    ],
                    batch_size=3,
                    seed=100,
                ),
            )

        assert resp.status_code == 200
        data = resp.json()
        compiled_prompts = [image["compiled_prompt"] for image in data["images"]]

        assert "A red automaton." in compiled_prompts[0]
        assert "A blue automaton." in compiled_prompts[1]
        assert "A green automaton." in compiled_prompts[2]
        assert data["compiled_prompt"] == compiled_prompts[0]

        generate_prompts = [
            call.kwargs["prompt"] for call in mock_model_manager.generate.call_args_list[-3:]
        ]
        assert generate_prompts == compiled_prompts

    def test_generate_expands_negative_prompt_placeholders(
        self,
        test_client,
        mock_model_manager,
    ):
        """Negative prompt placeholders should be expanded during generation."""
        with (
            patch(
                "pipeworks.api.main.get_model_runtime_support",
                return_value=(True, None),
            ),
            patch(
                "pipeworks.api.prompt_builder._PLACEHOLDER_RANDOM",
                SequenceRandom(["red", "blur"]),
            ),
        ):
            resp = test_client.post(
                "/api/generate",
                json=self._make_generate_payload(
                    model_id="flux-2-klein-4b",
                    negative_prompt="{red|blue} tint, {blur|noise}",
                ),
            )

        assert resp.status_code == 200
        assert mock_model_manager.generate.call_args.kwargs["negative_prompt"] == "red tint, blur"
        assert resp.json()["images"][0]["negative_prompt"] == "red tint, blur"

    def test_generate_cancel_returns_partial_batch(
        self,
        test_client,
        mock_model_manager,
    ):
        """Cancelling a batch should return completed images only."""
        generation_id = "gen-cancel-test"

        def _slow_generate(**kwargs):
            time.sleep(0.05)
            from PIL import Image

            return Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(255, 0, 0))

        mock_model_manager.generate.side_effect = _slow_generate

        cancel_response = {}

        def _cancel_batch():
            time.sleep(0.01)
            cancel_response["resp"] = test_client.post(
                "/api/generate/cancel",
                json={"generation_id": generation_id},
            )

        cancel_thread = threading.Thread(target=_cancel_batch)
        cancel_thread.start()
        try:
            resp = test_client.post(
                "/api/generate",
                json=self._make_generate_payload(
                    batch_size=3,
                    generation_id=generation_id,
                ),
            )
        finally:
            cancel_thread.join()

        assert cancel_response["resp"].status_code == 200
        data = resp.json()
        assert resp.status_code == 200
        assert data["cancelled"] is True
        assert data["completed_count"] == 1
        assert len(data["images"]) == 1

    def test_generate_cancel_forwards_to_remote_worker(self, test_client):
        """Cancel endpoint should forward cancellation to active remote worker."""
        settings_resp = test_client.post(
            "/api/gpu-settings",
            json={
                "use_remote_gpu": True,
                "remote_label": "Remote C",
                "remote_base_url": "https://worker.example.com",
                "bearer_token": "worker-token",
            },
        )
        assert settings_resp.status_code == 200

        generation_id = "gen-remote-cancel"
        worker_generate_started = threading.Event()
        cancel_forwarded = threading.Event()
        responses: dict[str, object] = {}

        def _fake_worker_post(*, path, payload, **kwargs):
            if path == "/api/worker/generate-batch":
                worker_generate_started.set()
                time.sleep(0.08)
                if cancel_forwarded.is_set():
                    return {"success": True, "cancelled": True, "completed_count": 0, "results": []}
                return {"success": True, "cancelled": False, "completed_count": 0, "results": []}
            if path == "/api/worker/generate/cancel":
                assert payload == {"generation_id": generation_id}
                cancel_forwarded.set()
                return {"success": True, "generation_id": generation_id, "status": "cancelling"}
            raise AssertionError(path)

        def _run_generate():
            responses["generate"] = test_client.post(
                "/api/generate",
                json=self._make_generate_payload(
                    gpu_worker_id="remote-ts",
                    generation_id=generation_id,
                ),
            )

        with patch("pipeworks.api.main._post_json_with_bearer", side_effect=_fake_worker_post):
            generate_thread = threading.Thread(target=_run_generate)
            generate_thread.start()
            worker_generate_started.wait(timeout=0.5)
            cancel_resp = test_client.post(
                "/api/generate/cancel",
                json={"generation_id": generation_id},
            )
            generate_thread.join()

        assert cancel_resp.status_code == 200
        assert cancel_forwarded.is_set()
        generate_resp = responses["generate"]
        assert generate_resp.status_code == 200
        assert generate_resp.json()["cancelled"] is True

    def test_generation_status_reports_live_model_download_phase(
        self,
        test_client,
        mock_model_manager,
    ):
        """Status endpoint should expose first-run Luminal download/load progress."""
        generation_id = "gen-status-test"
        load_started = threading.Event()
        allow_load_to_finish = threading.Event()
        responses = {}

        def _slow_load_model(hf_id):
            load_started.set()
            allow_load_to_finish.wait(timeout=0.5)
            mock_model_manager.current_model_id = hf_id

        mock_model_manager.load_model.side_effect = _slow_load_model

        def _run_generate():
            responses["generate"] = test_client.post(
                "/api/generate",
                json=self._make_generate_payload(
                    model_id="sd-v1-5",
                    generation_id=generation_id,
                ),
            )

        generate_thread = threading.Thread(target=_run_generate)
        generate_thread.start()
        try:
            assert load_started.wait(timeout=0.5) is True
            status_resp = test_client.get(f"/api/generate/status/{generation_id}")
            assert status_resp.status_code == 200
            payload = status_resp.json()
            assert payload["generation_id"] == generation_id
            assert payload["model_id"] == "sd-v1-5"
            assert payload["worker_label"] == "Luminal GPU"
            assert payload["cache_miss"] is True
            assert payload["phase"] in {"downloading_model", "loading_model"}
            assert "Luminal" in payload["message"]
        finally:
            allow_load_to_finish.set()
            generate_thread.join()

        assert responses["generate"].status_code == 200

    def test_generation_status_unknown_id_returns_404(self, test_client):
        """Status endpoint should fail closed for unknown generation ids."""
        resp = test_client.get("/api/generate/status/missing-generation")
        assert resp.status_code == 404

    def test_cancel_generation_unknown_id_returns_404(self, test_client):
        """Cancelling a non-existent batch should return 404."""
        resp = test_client.post(
            "/api/generate/cancel",
            json={"generation_id": "missing-generation"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Worker endpoint tests.
# ---------------------------------------------------------------------------


class TestWorkerEndpoints:
    """Test internal worker API endpoints."""

    def test_worker_health_requires_bearer_token(self, test_client):
        """Worker health endpoint should reject unauthenticated requests."""
        resp = test_client.get("/api/worker/health")
        assert resp.status_code == 401

    def test_worker_health_success_with_valid_token(self, test_client):
        """Worker health endpoint should return ok for valid bearer token."""
        from pipeworks.api import main as main_module

        previous_tokens = list(main_module.config.worker_api_bearer_tokens)
        main_module.config.worker_api_bearer_tokens = ["worker-secret"]
        try:
            resp = test_client.get(
                "/api/worker/health",
                headers={"Authorization": "Bearer worker-secret"},
            )
        finally:
            main_module.config.worker_api_bearer_tokens = previous_tokens

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["status"] == "ok"

    def test_worker_generate_batch_success_and_no_gallery_write(self, test_client):
        """Worker batch endpoint should return PNG payloads without gallery persistence."""
        from pipeworks.api import main as main_module

        previous_tokens = list(main_module.config.worker_api_bearer_tokens)
        main_module.config.worker_api_bearer_tokens = ["worker-secret"]
        payload = {
            "generation_id": "worker-gen-1",
            "hf_id": "Tongyi-MAI/Z-Image-Turbo",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 0.0,
            "jobs": [
                {
                    "index": 0,
                    "seed": 42,
                    "prompt": "A test image.",
                    "negative_prompt": None,
                }
            ],
        }
        try:
            resp = test_client.post(
                "/api/worker/generate-batch",
                headers={"Authorization": "Bearer worker-secret"},
                json=payload,
            )
        finally:
            main_module.config.worker_api_bearer_tokens = previous_tokens

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["completed_count"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["index"] == 0
        assert data["results"][0]["seed"] == 42
        assert isinstance(data["results"][0]["png_base64"], str)

        gallery_resp = test_client.get("/api/gallery")
        assert gallery_resp.status_code == 200
        assert gallery_resp.json()["total"] == 0

    def test_worker_cancel_unknown_generation_returns_404(self, test_client):
        """Worker cancel endpoint should return 404 for unknown generation IDs."""
        from pipeworks.api import main as main_module

        previous_tokens = list(main_module.config.worker_api_bearer_tokens)
        main_module.config.worker_api_bearer_tokens = ["worker-secret"]
        try:
            resp = test_client.post(
                "/api/worker/generate/cancel",
                headers={"Authorization": "Bearer worker-secret"},
                json={"generation_id": "missing"},
            )
        finally:
            main_module.config.worker_api_bearer_tokens = previous_tokens

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Prompt compilation endpoint tests.
# ---------------------------------------------------------------------------


class TestPromptCompile:
    """Test POST /api/prompt/compile — dynamic-section prompt preview."""

    def _make_compile_payload(self, **overrides) -> dict:
        sections_override = overrides.pop("sections", None)
        payload = {
            "model_id": "z-image-turbo",
            "prompt_schema_version": 3,
            "sections": sections_override
            or [
                {"label": "Subject", "mode": "manual", "manual_text": "A test scene."},
            ],
            "aspect_ratio_id": "1:1",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 0.0,
        }
        payload.update(overrides)
        return payload

    def test_compile_rejects_legacy_v2_payload(self, test_client):
        """Legacy v2 compile payloads should be rejected by Pydantic validation."""
        resp = test_client.post(
            "/api/prompt/compile",
            json={
                "model_id": "z-image-turbo",
                "prompt_schema_version": 2,
                "subject_mode": "manual",
                "manual_subject": "A legacy scene.",
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
            },
        )
        assert resp.status_code == 422

    def test_compile_dynamic_v3_emits_sections_in_submitted_order(self, test_client):
        """v3 compile should emit dynamic section blocks in submitted order."""
        resp = test_client.post(
            "/api/prompt/compile",
            json={
                "model_id": "z-image-turbo",
                "prompt_schema_version": 3,
                "sections": [
                    {
                        "label": "Tone",
                        "mode": "manual",
                        "manual_text": "Sepia and grit.",
                    },
                    {
                        "label": "Species",
                        "mode": "manual",
                        "manual_text": "A goblin inventor.",
                    },
                ],
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        compiled = data["compiled_prompt"]
        assert compiled.index("Tone:") < compiled.index("Species:")
        assert "Sepia and grit." in compiled
        assert "A goblin inventor." in compiled
        assert [s["label"] for s in data["token_counts"]["sections"]] == ["Tone", "Species"]
        assert all(s["tokens"] > 0 for s in data["token_counts"]["sections"])

    def test_compile_dynamic_v3_drops_empty_sections_from_compiled_text(self, test_client):
        """v3 compile should silently omit sections with empty resolved text."""
        resp = test_client.post(
            "/api/prompt/compile",
            json={
                "model_id": "z-image-turbo",
                "prompt_schema_version": 3,
                "sections": [
                    {
                        "label": "Tone",
                        "mode": "manual",
                        "manual_text": "",
                    },
                    {
                        "label": "Species",
                        "mode": "manual",
                        "manual_text": "A goblin inventor.",
                    },
                ],
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Tone:" not in data["compiled_prompt"]
        assert "Species:\nA goblin inventor." in data["compiled_prompt"]
        labels = [s["label"] for s in data["token_counts"]["sections"]]
        assert labels == ["Tone", "Species"]

    def test_compile_returns_flux2_token_counts(self, test_client):
        """Prompt preview should return tokenizer-backed token counts for FLUX.2-klein-4B."""
        resp = test_client.post(
            "/api/prompt/compile",
            json=self._make_compile_payload(
                model_id="flux-2-klein-4b",
                sections=[
                    {
                        "label": "Subject",
                        "mode": "manual",
                        "manual_text": "A moonlit machine garden.",
                    },
                    {
                        "label": "Style",
                        "mode": "manual",
                        "manual_text": "Ink wash style and soft bloom.",
                    },
                ],
                steps=28,
                guidance=4.0,
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        counts = data["token_counts"]
        assert counts["method"] == "tokenizer"
        assert all(s["tokens"] > 0 for s in counts["sections"])
        assert counts["total"] >= max(s["tokens"] for s in counts["sections"])

    def test_compile_expands_placeholders_once_per_request(self, test_client):
        """Prompt preview should expand placeholder groups in the response text."""
        with patch(
            "pipeworks.api.prompt_builder._PLACEHOLDER_RANDOM",
            SequenceRandom(["blue", "fog"]),
        ):
            resp = test_client.post(
                "/api/prompt/compile",
                json=self._make_compile_payload(
                    sections=[
                        {
                            "label": "Subject",
                            "mode": "manual",
                            "manual_text": "A {red|blue} automaton.",
                        },
                        {
                            "label": "Atmosphere",
                            "mode": "manual",
                            "manual_text": "Wrapped in {smoke|fog}.",
                        },
                    ],
                ),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "A blue automaton." in data["compiled_prompt"]
        assert "Wrapped in fog." in data["compiled_prompt"]

    def test_compile_automated_section_with_preset_id(self, test_client):
        """Automated section with a valid preset id should resolve its preset value."""
        resp = test_client.post(
            "/api/prompt/compile",
            json=self._make_compile_payload(
                sections=[
                    {
                        "label": "Scene",
                        "mode": "automated",
                        "automated_prompt_id": "goblin-workshop",
                    }
                ],
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "goblin" in data["compiled_prompt"].lower()

    def test_compile_automated_section_can_use_append_library_prompt(self, test_client):
        """Section automation should accept prompts sourced from append.json."""
        resp = test_client.post(
            "/api/prompt/compile",
            json=self._make_compile_payload(
                sections=[
                    {
                        "label": "Quality",
                        "mode": "automated",
                        "automated_prompt_id": "high-detail",
                    }
                ],
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "award-winning quality" in data["compiled_prompt"].lower()


# ---------------------------------------------------------------------------
# Gallery endpoint tests.
# ---------------------------------------------------------------------------


class TestGallery:
    """Test gallery listing, filtering, and pagination."""

    def test_empty_gallery(self, test_client):
        """An empty gallery should return zero images."""
        resp = test_client.get("/api/gallery")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["images"] == []

    def test_gallery_after_generation(self, test_client):
        """After generating an image, the gallery should contain it."""
        test_client.post(
            "/api/generate",
            json={
                "model_id": "z-image-turbo",
                "prompt_schema_version": 3,
                "sections": [
                    {"label": "Subject", "mode": "manual", "manual_text": "test"},
                ],
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
                "batch_size": 1,
            },
        )

        resp = test_client.get("/api/gallery")
        data = resp.json()
        assert data["total"] == 1

    def test_gallery_pagination(self, test_client, sample_gallery):
        """Pagination should return the correct slice of images."""
        resp = test_client.get("/api/gallery?page=1&per_page=2")
        data = resp.json()
        assert data["total"] == 5
        assert len(data["images"]) == 2
        assert data["page"] == 1
        assert data["pages"] == 3  # ceil(5 / 2) = 3

    def test_gallery_favourites_filter(self, test_client, sample_gallery):
        """Favourites filter should return only favourited images."""
        resp = test_client.get("/api/gallery?favourites_only=true")
        data = resp.json()
        # Only the first entry is favourited in sample_gallery.
        assert data["total"] == 1
        assert data["images"][0]["is_favourite"] is True

    def test_gallery_model_filter(self, test_client, sample_gallery):
        """Model filter should return only images from the specified model."""
        resp = test_client.get("/api/gallery?model_id=z-image-turbo")
        data = resp.json()
        assert data["total"] == 5  # All sample images use z-image-turbo.

    def test_gallery_model_filter_no_match(self, test_client, sample_gallery):
        """Model filter with no matching images should return empty list."""
        resp = test_client.get("/api/gallery?model_id=nonexistent")
        data = resp.json()
        assert data["total"] == 0

    def test_gallery_prunes_missing_files_from_counts(
        self,
        test_client,
        sample_gallery,
        tmp_gallery_dir,
        test_config,
    ):
        """Missing files on disk should be removed from gallery totals automatically."""
        missing_entry = sample_gallery[0]
        missing_path = tmp_gallery_dir / missing_entry["filename"]
        assert missing_path.exists()

        missing_path.unlink()

        resp = test_client.get("/api/gallery")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total"] == 4
        assert all(image["id"] != missing_entry["id"] for image in data["images"])

        persisted_gallery = json.loads(test_config.gallery_db.read_text())
        assert len(persisted_gallery) == 4
        assert all(entry["id"] != missing_entry["id"] for entry in persisted_gallery)

    def test_gallery_model_filter_counts_only_existing_matching_images(
        self,
        test_client,
        sample_gallery_mixed_models,
        tmp_gallery_dir,
    ):
        """Model-filtered totals should ignore manually deleted files."""
        missing_entry = next(
            entry for entry in sample_gallery_mixed_models if entry["model_id"] == "z-image-turbo"
        )
        (tmp_gallery_dir / missing_entry["filename"]).unlink()

        overall_resp = test_client.get("/api/gallery")
        overall_data = overall_resp.json()
        assert overall_data["total"] == 4

        model_resp = test_client.get("/api/gallery?model_id=z-image-turbo")
        model_data = model_resp.json()
        assert model_data["total"] == 2
        assert all(image["model_id"] == "z-image-turbo" for image in model_data["images"])

        other_model_resp = test_client.get("/api/gallery?model_id=sdxl-base")
        other_model_data = other_model_resp.json()
        assert other_model_data["total"] == 2

    def test_gallery_clamps_page_after_collection_shrinks(self, test_client, sample_gallery):
        """Pagination should clamp to the last valid page after deletions."""
        first_page = test_client.get("/api/gallery?page=3&per_page=2")
        assert first_page.status_code == 200
        assert first_page.json()["page"] == 3

        test_client.delete(f"/api/gallery/{sample_gallery[0]['id']}")
        test_client.delete(f"/api/gallery/{sample_gallery[1]['id']}")

        resp = test_client.get("/api/gallery?page=3&per_page=2")
        data = resp.json()
        assert data["total"] == 3
        assert data["pages"] == 2
        assert data["page"] == 2
        assert len(data["images"]) == 1


class TestGalleryEntry:
    """Test single gallery entry retrieval."""

    def test_get_existing_entry(self, test_client, sample_gallery):
        """GET /api/gallery/{id} should return the correct entry."""
        entry_id = sample_gallery[0]["id"]
        resp = test_client.get(f"/api/gallery/{entry_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == entry_id

    def test_get_nonexistent_entry(self, test_client, sample_gallery):
        """GET /api/gallery/{id} with an unknown ID should return 404."""
        resp = test_client.get("/api/gallery/nonexistent-id")
        assert resp.status_code == 404


class TestGalleryPrompt:
    """Test prompt metadata retrieval."""

    def test_get_prompt_metadata(self, test_client, sample_gallery):
        """GET /api/gallery/{id}/prompt should return prompt details."""
        entry_id = sample_gallery[0]["id"]
        resp = test_client.get(f"/api/gallery/{entry_id}/prompt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == entry_id
        assert "compiled_prompt" in data
        assert "seed" in data

    def test_get_prompt_nonexistent(self, test_client, sample_gallery):
        """GET /api/gallery/{id}/prompt with unknown ID should return 404."""
        resp = test_client.get("/api/gallery/nonexistent-id/prompt")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Zip download endpoint tests.
# ---------------------------------------------------------------------------


class TestZipDownload:
    """Test GET /api/gallery/{id}/zip — zip archive download."""

    def test_zip_download_success(self, test_client, sample_gallery):
        """GET /api/gallery/{id}/zip should return a valid zip archive."""
        import io
        import zipfile

        image = sample_gallery[0]
        resp = test_client.get(f"/api/gallery/{image['id']}/zip")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            id_short = image["id"][:8]
            assert f"pipeworks_{id_short}.png" in names
            assert f"pipeworks_{id_short}_metadata.json" in names

    def test_zip_download_not_found(self, test_client, sample_gallery):
        """GET /api/gallery/{id}/zip should 404 for unknown image IDs."""
        resp = test_client.get("/api/gallery/does-not-exist/zip")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Favourite endpoint tests.
# ---------------------------------------------------------------------------


class TestFavourite:
    """Test POST /api/gallery/favourite — favourite toggling."""

    def test_set_favourite(self, test_client, sample_gallery):
        """Should mark an image as favourite."""
        entry_id = sample_gallery[1]["id"]  # Not favourited initially.
        resp = test_client.post(
            "/api/gallery/favourite",
            json={"image_id": entry_id, "is_favourite": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["is_favourite"] is True

        # Verify the change persisted in the gallery.
        check = test_client.get(f"/api/gallery/{entry_id}")
        assert check.json()["is_favourite"] is True

    def test_unset_favourite(self, test_client, sample_gallery):
        """Should unmark a favourited image."""
        entry_id = sample_gallery[0]["id"]  # Favourited initially.
        resp = test_client.post(
            "/api/gallery/favourite",
            json={"image_id": entry_id, "is_favourite": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_favourite"] is False

    def test_favourite_nonexistent(self, test_client, sample_gallery):
        """Favouriting a nonexistent image should return 404."""
        resp = test_client.post(
            "/api/gallery/favourite",
            json={"image_id": "nonexistent-id", "is_favourite": True},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete endpoint tests.
# ---------------------------------------------------------------------------


class TestDelete:
    """Test DELETE /api/gallery/{id} — image deletion."""

    def test_delete_existing(self, test_client, sample_gallery):
        """Deleting an existing image should remove it from the gallery."""
        entry_id = sample_gallery[0]["id"]
        resp = test_client.delete(f"/api/gallery/{entry_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["deleted"] == entry_id

        # Verify it's gone from the gallery.
        check = test_client.get(f"/api/gallery/{entry_id}")
        assert check.status_code == 404

    def test_delete_reduces_count(self, test_client, sample_gallery):
        """Gallery total should decrease after deletion."""
        entry_id = sample_gallery[0]["id"]
        test_client.delete(f"/api/gallery/{entry_id}")

        resp = test_client.get("/api/gallery")
        assert resp.json()["total"] == 4

    def test_delete_updates_model_filtered_count(self, test_client, sample_gallery_mixed_models):
        """Deleting a filtered-model image should reduce that filtered total."""
        entry_id = next(
            entry["id"] for entry in sample_gallery_mixed_models if entry["model_id"] == "sdxl-base"
        )

        delete_resp = test_client.delete(f"/api/gallery/{entry_id}")
        assert delete_resp.status_code == 200

        filtered_resp = test_client.get("/api/gallery?model_id=sdxl-base")
        filtered_data = filtered_resp.json()
        assert filtered_data["total"] == 1
        assert all(entry["model_id"] == "sdxl-base" for entry in filtered_data["images"])

    def test_delete_nonexistent(self, test_client, sample_gallery):
        """Deleting a nonexistent image should return 404."""
        resp = test_client.delete("/api/gallery/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Stats endpoint tests.
# ---------------------------------------------------------------------------


class TestBulkDelete:
    """Test POST /api/gallery/bulk-delete — bulk image deletion."""

    def test_bulk_delete_success(self, test_client, sample_gallery):
        """Deleting 2 of 5 images should leave 3 remaining."""
        ids_to_delete = [sample_gallery[0]["id"], sample_gallery[1]["id"]]

        resp = test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": ids_to_delete},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert set(data["deleted"]) == set(ids_to_delete)
        assert data["not_found"] == []

        # Verify gallery count decreased.
        gallery_resp = test_client.get("/api/gallery")
        assert gallery_resp.json()["total"] == 3

    def test_bulk_delete_partial_not_found(self, test_client, sample_gallery):
        """Some IDs missing should delete what exists and report not_found."""
        valid_id = sample_gallery[0]["id"]
        fake_id = "nonexistent-id-12345"

        resp = test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": [valid_id, fake_id]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert valid_id in data["deleted"]
        assert fake_id in data["not_found"]

        # Gallery should have 4 remaining.
        gallery_resp = test_client.get("/api/gallery")
        assert gallery_resp.json()["total"] == 4

    def test_bulk_delete_empty_list_rejected(self, test_client, sample_gallery):
        """An empty image_ids list should be rejected by Pydantic validation."""
        resp = test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": []},
        )
        assert resp.status_code == 422

    def test_bulk_delete_removes_files(self, test_client, sample_gallery, tmp_gallery_dir):
        """PNG files should be removed from disk after bulk delete."""
        entry = sample_gallery[0]
        filepath = tmp_gallery_dir / entry["filename"]
        assert filepath.exists()

        test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": [entry["id"]]},
        )

        assert not filepath.exists()

    def test_bulk_delete_all(self, test_client, sample_gallery):
        """Deleting all images should result in an empty gallery."""
        all_ids = [e["id"] for e in sample_gallery]

        resp = test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": all_ids},
        )
        assert resp.status_code == 200
        assert len(resp.json()["deleted"]) == 5

        gallery_resp = test_client.get("/api/gallery")
        assert gallery_resp.json()["total"] == 0


class TestStats:
    """Test GET /api/stats — gallery statistics."""

    def test_stats_empty_gallery(self, test_client):
        """Stats on an empty gallery should show all zeros."""
        resp = test_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 0
        assert data["total_favourites"] == 0
        assert data["model_counts"] == {}

    def test_stats_with_gallery(self, test_client, sample_gallery):
        """Stats should reflect the sample gallery contents."""
        resp = test_client.get("/api/stats")
        data = resp.json()
        assert data["total_images"] == 5
        assert data["total_favourites"] == 1  # Only first entry is favourited.
        assert data["model_counts"]["z-image-turbo"] == 5

    def test_stats_exclude_entries_for_missing_files(
        self,
        test_client,
        sample_gallery_mixed_models,
        tmp_gallery_dir,
    ):
        """Stats should ignore metadata whose image files were removed manually."""
        removed_entry = sample_gallery_mixed_models[0]
        (tmp_gallery_dir / removed_entry["filename"]).unlink()

        resp = test_client.get("/api/stats")
        data = resp.json()

        assert data["total_images"] == 4
        assert data["model_counts"]["z-image-turbo"] == 2
        assert data["model_counts"]["sdxl-base"] == 2


# ---------------------------------------------------------------------------
# Gallery runs endpoint tests.
# ---------------------------------------------------------------------------


class TestGalleryRuns:
    """Test GET /api/gallery/runs — run-grouped gallery listing."""

    def test_runs_empty_gallery(self, test_client):
        """An empty gallery should return zero runs."""
        resp = test_client.get("/api/gallery/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 0
        assert data["total_images"] == 0
        assert data["runs"] == []

    def test_runs_grouping(self, test_client, sample_gallery_with_runs):
        """Images should be grouped by batch_seed into distinct runs."""
        resp = test_client.get("/api/gallery/runs")
        assert resp.status_code == 200
        data = resp.json()

        # 3 runs: batch_seed 1000 (4 images), 2000 (2 images), 3000 (1 image)
        assert data["total_runs"] == 3
        assert data["total_images"] == 7

        seeds = [r["batch_seed"] for r in data["runs"]]
        # Newest first: Run A (today) then Run B and C (yesterday).
        assert seeds[0] == 1000

    def test_runs_pagination(self, test_client, sample_gallery_with_runs):
        """Pagination should work on runs, not individual images."""
        resp = test_client.get("/api/gallery/runs?per_page=1&page=1")
        data = resp.json()

        assert data["total_runs"] == 3
        assert data["pages"] == 3
        assert len(data["runs"]) == 1

    def test_runs_model_filter(self, test_client, sample_gallery_with_runs):
        """Model filter should exclude runs from other models."""
        resp = test_client.get("/api/gallery/runs?model_id=sdxl-1-0")
        data = resp.json()

        assert data["total_runs"] == 1
        assert data["runs"][0]["model_id"] == "sdxl-1-0"

    def test_runs_thumbnail_limit(self, test_client, sample_gallery_with_runs):
        """Thumbnail limit should cap images returned but not total_images."""
        resp = test_client.get("/api/gallery/runs?thumbnail_limit=2")
        data = resp.json()

        # Run A has 4 images but should return only 2 thumbnails.
        run_a = next(r for r in data["runs"] if r["batch_seed"] == 1000)
        assert run_a["total_images"] == 4
        assert run_a["thumbnail_count"] == 2
        assert len(run_a["images"]) == 2

    def test_runs_include_date_field(self, test_client, sample_gallery_with_runs):
        """Each run should include a date string."""
        resp = test_client.get("/api/gallery/runs")
        data = resp.json()

        for run in data["runs"]:
            assert "date" in run
            assert len(run["date"]) == 10  # YYYY-MM-DD


class TestRunZipDownload:
    """Test GET /api/gallery/runs/{batch_seed}/zip — bulk run zip download."""

    def test_run_zip_contains_all_images(self, test_client, sample_gallery_with_runs):
        """The zip should contain PNG + metadata for every image in the run."""
        import io
        import zipfile

        resp = test_client.get("/api/gallery/runs/1000/zip")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            # Run A has 4 images, so 4 PNGs + 4 metadata JSONs = 8 files.
            assert len(names) == 8

            png_count = sum(1 for n in names if n.endswith(".png"))
            json_count = sum(1 for n in names if n.endswith("_metadata.json"))
            assert png_count == 4
            assert json_count == 4

    def test_run_zip_404_for_unknown_seed(self, test_client, sample_gallery_with_runs):
        """A non-existent batch_seed should return 404."""
        resp = test_client.get("/api/gallery/runs/999999/zip")
        assert resp.status_code == 404

    def test_run_zip_metadata_format(self, test_client, sample_gallery_with_runs):
        """Metadata in the run zip should match the per-image zip format."""
        import io
        import zipfile

        resp = test_client.get("/api/gallery/runs/1000/zip")

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            json_files = [n for n in zf.namelist() if n.endswith("_metadata.json")]
            metadata = json.loads(zf.read(json_files[0]))

        # Same top-level structure as per-image zip.
        assert "id" in metadata
        assert "model" in metadata
        assert "prompt" in metadata
        assert "generation" in metadata
        assert "batch" in metadata
        assert "created_at" in metadata

    def test_run_zip_content_disposition(self, test_client, sample_gallery_with_runs):
        """The filename in Content-Disposition should include the batch_seed."""
        resp = test_client.get("/api/gallery/runs/1000/zip")
        assert "pipeworks_run_1000.zip" in resp.headers["content-disposition"]

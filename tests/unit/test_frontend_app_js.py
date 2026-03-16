"""Frontend module smoke tests for app composition and API wiring."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "pipeworks" / "static" / "js"


def _read_js(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_app_bootstraps_feature_modules() -> None:
    """Root app should compose feature modules from app/*.mjs."""
    script = _read_js("app.js")

    assert 'from "./app/api-client.mjs"' in script
    assert 'from "./app/runtime-gpu-controller.mjs"' in script
    assert 'from "./app/prompt-composer.mjs"' in script
    assert 'from "./app/generation-flow.mjs"' in script
    assert 'from "./app/gallery-manager.mjs"' in script


def test_api_client_defines_runtime_and_gpu_routes() -> None:
    """API client module should centralize runtime and GPU endpoints."""
    script = _read_js("app/api-client.mjs")

    assert "/api/runtime-mode" in script
    assert "/api/runtime-auth" in script
    assert "/api/runtime-login" in script
    assert "/api/policy-prompts" in script
    assert "/api/gpu-settings" in script
    assert "/api/gpu-settings/test" in script
    assert "/api/generate" in script


def test_prompt_composer_payload_includes_gpu_worker_id() -> None:
    """Prompt composer payload should include selected worker identifier."""
    script = _read_js("app/prompt-composer.mjs")
    assert "gpu_worker_id: getSelectedGpuWorker()?.id || null" in script

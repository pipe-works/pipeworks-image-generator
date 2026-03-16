"""Frontend script smoke tests for generation payload wiring."""

from __future__ import annotations

from pathlib import Path


def test_app_js_includes_gpu_worker_in_generate_payload() -> None:
    """Generate payload should include selected gpu_worker_id."""
    script_path = (
        Path(__file__).resolve().parents[2] / "src" / "pipeworks" / "static" / "js" / "app.js"
    )
    script = script_path.read_text(encoding="utf-8")

    assert "gpu_worker_id: getSelectedGpuWorker()?.id || null" in script
    assert "selectedGpuWorkerLabel()" in script
    assert "/api/gpu-settings" in script
    assert "/api/gpu-settings/test" in script

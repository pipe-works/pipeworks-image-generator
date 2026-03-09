"""Unit tests for the CLI entry point in :mod:`pipeworks.api.main`."""

from __future__ import annotations

import uvicorn

from pipeworks.api import main as main_module


def test_main_uses_img_gen_log_prefix(monkeypatch) -> None:
    """CLI startup should prepend `img-gen` to uvicorn default/access logs."""
    captured: dict = {}

    def _fake_run(app_target: str, **kwargs) -> None:
        captured["app_target"] = app_target
        captured["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    main_module.main()

    assert captured["app_target"] == "pipeworks.api.main:app"
    log_config = captured["kwargs"]["log_config"]
    assert log_config["formatters"]["default"]["fmt"].startswith("img-gen ")
    assert log_config["formatters"]["access"]["fmt"].startswith("img-gen ")

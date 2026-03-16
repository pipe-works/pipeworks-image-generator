"""Unit tests for prompt catalog loading."""

from __future__ import annotations

import json

from pipeworks.api.services.prompt_catalog import load_prompt_catalog


def test_load_prompt_catalog_ignores_legacy_prompts_json(tmp_path):
    """Only split prompt files should be used for catalog loading."""
    (tmp_path / "prepend.json").write_text(
        json.dumps([{"id": "new-prepend", "label": "New Pre", "value": "new pre"}]),
        encoding="utf-8",
    )
    (tmp_path / "main.json").write_text(
        json.dumps([{"id": "new-main", "label": "New Main", "value": "new main"}]),
        encoding="utf-8",
    )
    (tmp_path / "append.json").write_text(
        json.dumps([{"id": "new-append", "label": "New Post", "value": "new post"}]),
        encoding="utf-8",
    )

    legacy_payload = {
        "prepend_prompts": [{"id": "legacy-prepend", "label": "Legacy Pre", "value": "old pre"}],
        "automated_prompts": [{"id": "legacy-main", "label": "Legacy Main", "value": "old main"}],
        "append_prompts": [{"id": "legacy-append", "label": "Legacy Post", "value": "old post"}],
    }
    (tmp_path / "prompts.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    prompts = load_prompt_catalog(data_dir=tmp_path)

    assert [item["id"] for item in prompts["prepend_library"]] == ["new-prepend"]
    assert [item["id"] for item in prompts["main_library"]] == ["new-main"]
    assert [item["id"] for item in prompts["append_library"]] == ["new-append"]

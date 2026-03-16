"""Unit tests for prompt catalog loading and legacy fallback warnings."""

from __future__ import annotations

import json
import logging

from pipeworks.api.services import deprecations
from pipeworks.api.services.deprecations import LEGACY_PROMPTS_JSON_FALLBACK_WARNING
from pipeworks.api.services.prompt_catalog import load_prompt_catalog


def test_load_prompt_catalog_warns_when_prompts_json_fallback_used(tmp_path, caplog):
    """Using legacy prompts.json fallback should emit one startup warning."""
    deprecations._WARNED_KEYS.clear()

    legacy_payload = {
        "prepend_prompts": [{"id": "legacy-prepend", "label": "Legacy Pre", "value": "old pre"}],
        "automated_prompts": [{"id": "legacy-main", "label": "Legacy Main", "value": "old main"}],
        "append_prompts": [{"id": "legacy-append", "label": "Legacy Post", "value": "old post"}],
    }
    (tmp_path / "prompts.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        prompts = load_prompt_catalog(data_dir=tmp_path)
        _ = load_prompt_catalog(data_dir=tmp_path)

    assert [item["id"] for item in prompts["prepend_library"]] == ["legacy-prepend"]
    assert [item["id"] for item in prompts["main_library"]] == ["legacy-main"]
    assert [item["id"] for item in prompts["append_library"]] == ["legacy-append"]
    matching = [
        record.message
        for record in caplog.records
        if LEGACY_PROMPTS_JSON_FALLBACK_WARNING in record.message
    ]
    assert len(matching) == 1

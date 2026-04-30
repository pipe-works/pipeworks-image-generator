"""Tests for model-aware prompt token counting."""

from __future__ import annotations

from unittest.mock import patch

from pipeworks.core.prompt_token_counter import PromptTokenCounter, _load_tokenizer


class _FakeTokenizer:
    def __init__(self, *, with_chat_template: bool = True) -> None:
        self.chat_template = "fake-template" if with_chat_template else None
        self.applied_prompts: list[str] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        prompt = messages[0]["content"]
        self.applied_prompts.append(prompt)
        return f"<chat>{prompt}</chat>"

    def __call__(self, text: str, **kwargs) -> dict:
        normalized = text.replace("<chat>", "").replace("</chat>", "")
        return {"input_ids": normalized.split()}


def test_count_dynamic_prompt_sections_uses_chat_template_for_total_only(test_config):
    """Per-slot counts should be plain tokenizer counts; total should use chat wrapping."""
    tokenizer = _FakeTokenizer()
    counter = PromptTokenCounter(test_config)

    with patch("pipeworks.core.prompt_token_counter._load_tokenizer", return_value=tokenizer):
        counts = counter.count_dynamic_prompt_sections(
            hf_id="black-forest-labs/FLUX.2-klein-4B",
            sections=[
                {"label": "Style", "text": "ink sketch"},
                {"label": "Subject", "text": "a brass automaton"},
                {"label": "Lighting", "text": "dramatic lighting"},
            ],
            compiled_prompt="ink sketch a brass automaton dramatic lighting",
        )

    assert [entry["label"] for entry in counts["sections"]] == ["Style", "Subject", "Lighting"]
    assert [entry["tokens"] for entry in counts["sections"]] == [2, 3, 2]
    assert counts["total"] == 7
    assert counts["method"] == "tokenizer"
    assert tokenizer.applied_prompts == ["ink sketch a brass automaton dramatic lighting"]


def test_count_dynamic_prompt_sections_falls_back_to_heuristic_when_tokenizer_load_fails(
    test_config,
):
    """A tokenizer load failure should not break prompt preview token counts."""
    counter = PromptTokenCounter(test_config)

    with patch(
        "pipeworks.core.prompt_token_counter._load_tokenizer",
        side_effect=RuntimeError("offline"),
    ):
        counts = counter.count_dynamic_prompt_sections(
            hf_id="black-forest-labs/FLUX.2-klein-4B",
            sections=[
                {"label": "Subject", "text": "a brass automaton"},
            ],
            compiled_prompt="a brass automaton",
        )

    assert counts["method"] == "heuristic"
    assert counts["total"] > 0
    assert counts["sections"][0]["label"] == "Subject"


def test_load_tokenizer_uses_tokenizer_subfolder():
    """Diffusers repos should load tokenizers from their tokenizer subfolder."""
    fake_tokenizer = object()

    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=fake_tokenizer,
    ) as auto_from_pretrained:
        _load_tokenizer.cache_clear()
        tokenizer = _load_tokenizer("Tongyi-MAI/Z-Image-Turbo", "/tmp/models")

    assert tokenizer is fake_tokenizer
    auto_from_pretrained.assert_called_once_with(
        "Tongyi-MAI/Z-Image-Turbo",
        cache_dir="/tmp/models",
        subfolder="tokenizer",
        use_fast=True,
    )


def test_load_tokenizer_falls_back_to_explicit_qwen_class():
    """Known Qwen-backed repos should fall back to explicit tokenizer classes."""
    fake_tokenizer = object()

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=ValueError("bad root config"),
        ) as auto_from_pretrained,
        patch(
            "transformers.Qwen2TokenizerFast.from_pretrained",
            return_value=fake_tokenizer,
        ) as qwen_fast_from_pretrained,
    ):
        _load_tokenizer.cache_clear()
        tokenizer = _load_tokenizer("black-forest-labs/FLUX.2-klein-4B", "/tmp/models")

    assert tokenizer is fake_tokenizer
    auto_from_pretrained.assert_called_once()
    qwen_fast_from_pretrained.assert_called_once_with(
        "black-forest-labs/FLUX.2-klein-4B",
        cache_dir="/tmp/models",
        subfolder="tokenizer",
    )

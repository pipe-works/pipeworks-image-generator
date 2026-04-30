"""Model-aware prompt token counting helpers.

This module centralises prompt token counting so the frontend can display
token budgets that match the real tokenizer path used by the underlying
model.  For chat-tuned tokenizers such as Qwen, total prompt counting applies
the tokenizer's chat template before tokenisation because the diffusion
pipeline does the same internally.
"""

from __future__ import annotations

import inspect
import logging
import os
from functools import lru_cache
from typing import Any, cast

from pipeworks.core.config import PipeworksConfig

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_TOKENIZER_SUBFOLDER = "tokenizer"
_EXPLICIT_TOKENIZER_CLASSES: dict[str, tuple[str, ...]] = {
    "Tongyi-MAI/Z-Image-Turbo": ("Qwen2Tokenizer", "Qwen2TokenizerFast"),
    "black-forest-labs/FLUX.2-klein-4B": ("Qwen2TokenizerFast", "Qwen2Tokenizer"),
}


def _estimate_tokens(text: str) -> int:
    """Fallback token estimate used when a real tokenizer is unavailable."""
    stripped = text.strip()
    if not stripped:
        return 0
    return (len(stripped) + 3) // 4


@lru_cache(maxsize=8)
def _load_tokenizer(hf_id: str, cache_dir: str):
    """Load and cache a tokenizer for a HuggingFace model ID."""
    import transformers
    from transformers import AutoTokenizer

    load_kwargs = {
        "cache_dir": cache_dir,
        "subfolder": _TOKENIZER_SUBFOLDER,
    }

    try:
        return AutoTokenizer.from_pretrained(
            hf_id,
            use_fast=True,
            **load_kwargs,
        )
    except Exception as auto_exc:
        last_exc = auto_exc
        for class_name in _EXPLICIT_TOKENIZER_CLASSES.get(hf_id, ()):
            tokenizer_cls = getattr(transformers, class_name, None)
            if tokenizer_cls is None:
                continue
            try:
                return tokenizer_cls.from_pretrained(hf_id, **load_kwargs)
            except Exception as explicit_exc:
                last_exc = explicit_exc
        raise last_exc


class PromptTokenCounter:
    """Count prompt tokens using the model's real tokenizer when possible."""

    def __init__(self, config: PipeworksConfig) -> None:
        self._config = config

    def count_dynamic_prompt_sections(
        self,
        *,
        hf_id: str | None,
        sections: list[dict[str, str]],
        compiled_prompt: str,
    ) -> dict[str, Any]:
        """Return per-slot token counts for the v3 dynamic-section schema.

        The returned dict mirrors the v2 shape's ``total`` and ``method``
        fields and adds a ``sections`` list with one entry per submitted
        slot in submitted order: ``{"label", "tokens"}``. Front-end token
        counters render off the list rather than fixed section names.
        """
        tokenizer = self._get_tokenizer(hf_id)
        if tokenizer is None:
            return {
                "sections": [
                    {
                        "label": section.get("label", "Policy"),
                        "tokens": _estimate_tokens(section.get("text", "")),
                    }
                    for section in sections
                ],
                "total": _estimate_tokens(compiled_prompt),
                "method": "heuristic",
            }

        return {
            "sections": [
                {
                    "label": section.get("label", "Policy"),
                    "tokens": self._count_text(tokenizer, section.get("text", "")),
                }
                for section in sections
            ],
            "total": self._count_text(tokenizer, compiled_prompt, apply_chat_template=True),
            "method": "tokenizer",
        }

    def _get_tokenizer(self, hf_id: str | None):
        if not hf_id:
            return None

        try:
            return _load_tokenizer(hf_id, str(self._config.models_dir))
        except Exception as exc:
            logger.warning(
                "Falling back to heuristic token counts for '%s': %s",
                hf_id,
                exc,
            )
            return None

    def _count_text(self, tokenizer, text: str, *, apply_chat_template: bool = False) -> int:
        stripped = text.strip()
        if not stripped:
            return 0

        token_input = stripped
        if apply_chat_template and getattr(tokenizer, "chat_template", None):
            token_input = self._apply_chat_template(tokenizer, stripped)

        encoded = tokenizer(
            token_input,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        input_ids = encoded.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)

    @staticmethod
    def _apply_chat_template(tokenizer, prompt: str) -> str:
        """Apply the tokenizer's user chat template to mirror pipeline input."""
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        signature = inspect.signature(tokenizer.apply_chat_template)
        if "enable_thinking" in signature.parameters:
            kwargs["enable_thinking"] = False

        return cast(
            str,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                **kwargs,
            ),
        )

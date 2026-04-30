"""Dynamic-section prompt compilation for the Image Generator (schema v3).

A compiled prompt is built from a curator-ordered list of labelled
sections. Each section is emitted in submitted order as a labelled block;
sections whose text resolves to empty are silently skipped. Block headers
use the section's ``label`` verbatim, with a trailing colon.

Usage::

    sections = [
        {"label": "Subject", "text": "A goblin inventor."},
        {"label": "Tone", "text": "Sepia and grit."},
    ]
    compiled = build_dynamic_prompt(sections)
"""

from __future__ import annotations

import random
import re

_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")
_PLACEHOLDER_RANDOM = random.SystemRandom()


def expand_prompt_placeholders(
    text: str,
    *,
    rng: random.Random | random.SystemRandom | None = None,
) -> str:
    """Expand ``{a|b|c}`` placeholders using an isolated RNG.

    Each placeholder is replaced independently. Empty options are ignored and
    malformed placeholders are left unchanged. Placeholder expansion is
    intentionally isolated from the module-level ``random`` state so prompt
    variation does not perturb unrelated randomness elsewhere in the app.
    """
    if not text:
        return text

    resolved_rng = rng or _PLACEHOLDER_RANDOM

    def _replace(match: re.Match[str]) -> str:
        options = [option.strip() for option in match.group(1).split("|")]
        valid_options = [option for option in options if option]
        if not valid_options:
            return match.group(0)
        return resolved_rng.choice(valid_options)

    expanded = text
    for _ in range(10):
        updated = _PLACEHOLDER_PATTERN.sub(_replace, expanded)
        if updated == expanded:
            break
        expanded = updated
    return expanded


def resolve_dynamic_prompt_variants(
    sections: list[dict[str, str]],
    *,
    rng: random.Random | random.SystemRandom | None = None,
) -> list[dict[str, str]]:
    """Expand placeholders for the dynamic v3 prompt schema."""
    return [
        {
            "label": section.get("label", "Policy"),
            "text": expand_prompt_placeholders(section.get("text", ""), rng=rng),
        }
        for section in sections
    ]


def build_dynamic_prompt(
    sections: list[dict[str, str]],
    *,
    expand_placeholders: bool = True,
    rng: random.Random | random.SystemRandom | None = None,
) -> str:
    """Build a prompt from a curator-ordered list of labelled sections.

    Sections are emitted in submitted order as labelled blocks. Sections
    whose resolved text is empty are silently skipped. Block headers use
    the section's ``label`` verbatim, with a trailing colon.
    """
    resolved_sections = (
        resolve_dynamic_prompt_variants(sections, rng=rng) if expand_placeholders else sections
    )

    parts: list[str] = []
    for section in resolved_sections:
        text = (section.get("text") or "").strip()
        if not text:
            continue
        label = (section.get("label") or "Policy").strip() or "Policy"
        parts.append(f"{label}:\n{text}")

    return "\n\n".join(parts)

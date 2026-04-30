"""Tests for pipeworks.api.prompt_builder — dynamic-section prompt compilation."""

from __future__ import annotations

import random

from pipeworks.api.prompt_builder import (
    build_dynamic_prompt,
    expand_prompt_placeholders,
    resolve_dynamic_prompt_variants,
)


class SequenceRandom:
    """Minimal deterministic RNG stub for placeholder expansion tests."""

    def __init__(self, picks: list[str]):
        self._picks = list(picks)

    def choice(self, options: list[str]) -> str:
        pick = self._picks.pop(0)
        assert pick in options
        return pick


class TestPromptPlaceholderExpansion:
    """Test random placeholder expansion within prompt text."""

    def test_expand_prompt_placeholders_replaces_each_group(self):
        """Each placeholder group should be expanded independently."""
        result = expand_prompt_placeholders(
            "A {red|blue} dragon in a {forest|cave}.",
            rng=SequenceRandom(["blue", "cave"]),
        )
        assert result == "A blue dragon in a cave."

    def test_expand_prompt_placeholders_preserves_global_random_state(self):
        """Placeholder expansion should not mutate module-level random state."""
        random.seed(12345)
        before = random.getstate()

        expand_prompt_placeholders("A {red|blue|green} banner.")

        after = random.getstate()
        assert after == before

    def test_expand_prompt_placeholders_returns_input_when_no_groups(self):
        """Plain text without placeholders should round-trip unchanged."""
        text = "Plain text with no placeholders."
        assert expand_prompt_placeholders(text) == text

    def test_expand_prompt_placeholders_leaves_empty_groups_alone(self):
        """Groups containing only empty options should be left as-is."""
        result = expand_prompt_placeholders("Before {|||} after.")
        assert result == "Before {|||} after."


class TestDynamicPromptBuilder:
    """Tests for the v3 dynamic ordered-section prompt schema."""

    def test_build_dynamic_prompt_emits_sections_in_submitted_order(self):
        """Section blocks should appear in the submitted order with their labels."""
        result = build_dynamic_prompt(
            [
                {"label": "Tone", "text": "Sepia and grit."},
                {"label": "Species", "text": "A goblin inventor."},
                {"label": "Lighting", "text": "Soft warm lamplight."},
            ],
            expand_placeholders=False,
        )
        assert result.index("Tone:") < result.index("Species:")
        assert result.index("Species:") < result.index("Lighting:")
        assert "A goblin inventor." in result
        assert "Soft warm lamplight." in result

    def test_build_dynamic_prompt_skips_empty_text(self):
        """Sections with empty resolved text should be silently omitted."""
        result = build_dynamic_prompt(
            [
                {"label": "Tone", "text": ""},
                {"label": "Species", "text": "A goblin."},
                {"label": "Setting", "text": "   "},
            ],
            expand_placeholders=False,
        )
        assert "Tone:" not in result
        assert "Setting:" not in result
        assert "Species:\nA goblin." in result

    def test_build_dynamic_prompt_falls_back_to_default_label(self):
        """Sections with empty/whitespace labels should render as 'Policy'."""
        result = build_dynamic_prompt(
            [
                {"label": "  ", "text": "First line."},
                {"label": "", "text": "Second line."},
            ],
            expand_placeholders=False,
        )
        assert result.count("Policy:") == 2
        assert "First line." in result
        assert "Second line." in result

    def test_build_dynamic_prompt_supports_duplicate_labels(self):
        """Labels are not required to be unique; both blocks should appear."""
        result = build_dynamic_prompt(
            [
                {"label": "Policy", "text": "alpha"},
                {"label": "Policy", "text": "beta"},
            ],
            expand_placeholders=False,
        )
        assert "alpha" in result
        assert "beta" in result
        assert "alpha\n\nPolicy:\nbeta" in result

    def test_build_dynamic_prompt_expands_placeholders(self):
        """Placeholder groups in section text expand once per section."""
        result = build_dynamic_prompt(
            [
                {"label": "Subject", "text": "A {goblin|human} inventor."},
                {"label": "Tone", "text": "{warm|cold}."},
            ],
            rng=SequenceRandom(["human", "warm"]),
        )
        assert "A human inventor." in result
        assert "warm." in result

    def test_resolve_dynamic_prompt_variants(self):
        """Per-section placeholder expansion should resolve each section once."""
        resolved = resolve_dynamic_prompt_variants(
            [
                {"label": "Subject", "text": "{goblin|human}"},
                {"label": "Setting", "text": "{city|coast}"},
            ],
            rng=SequenceRandom(["goblin", "coast"]),
        )
        assert resolved[0] == {"label": "Subject", "text": "goblin"}
        assert resolved[1] == {"label": "Setting", "text": "coast"}

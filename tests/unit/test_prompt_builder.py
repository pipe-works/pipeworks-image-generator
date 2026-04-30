"""Tests for pipeworks.api.prompt_builder — three-part prompt compilation.

Tests cover:
- Full prompt with all three parts (prepend, scene, append) in template mode.
- Empty prepend/append are silently omitted.
- Whitespace-only values are treated as empty.
- Fixed boilerplate sections are present in template mode, absent in manual mode.
- Double newline delimiters between sections.
- Main scene header is always included.
- Manual prepend mode omits style boilerplate.
- Manual append mode omits mood and colour boilerplate.
"""

from __future__ import annotations

import random

from pipeworks.api.prompt_builder import (
    _COLOUR_BOILERPLATE,
    _MOOD_BOILERPLATE,
    _STYLE_BOILERPLATE,
    build_dynamic_prompt,
    build_prompt,
    build_structured_prompt,
    expand_prompt_placeholders,
    resolve_dynamic_prompt_variants,
    resolve_prompt_variants,
    resolve_structured_prompt_variants,
)


class SequenceRandom:
    """Minimal deterministic RNG stub for placeholder expansion tests."""

    def __init__(self, picks: list[str]):
        self._picks = list(picks)

    def choice(self, options: list[str]) -> str:
        pick = self._picks.pop(0)
        assert pick in options
        return pick


class TestBuildPromptFullParts:
    """Test prompt compilation with all three user parts provided (template mode)."""

    def test_all_parts_present(self):
        """Compiled prompt should contain all three user-provided parts."""
        result = build_prompt(
            prepend_value="Oil painting style.",
            main_scene="A goblin workshop.",
            append_value="8K resolution.",
        )
        assert "Oil painting style." in result
        assert "A goblin workshop." in result
        assert "8K resolution." in result

    def test_contains_style_boilerplate(self):
        """Template mode should include the Ledgerfall style boilerplate."""
        result = build_prompt("", "A scene.", "")
        assert _STYLE_BOILERPLATE in result

    def test_contains_mood_boilerplate(self):
        """Template mode should include the mood boilerplate."""
        result = build_prompt("", "A scene.", "")
        assert _MOOD_BOILERPLATE in result

    def test_contains_colour_boilerplate(self):
        """Template mode should include the colour palette boilerplate."""
        result = build_prompt("", "A scene.", "")
        assert _COLOUR_BOILERPLATE in result

    def test_main_scene_header_present(self):
        """The 'Main Scene:' header should always appear before the scene."""
        result = build_prompt("", "A scene.", "")
        assert "Main Scene:" in result

    def test_sections_separated_by_double_newlines(self):
        """All sections should be separated by double newlines."""
        result = build_prompt("Prepend.", "Scene.", "Append.")
        # Each section boundary should have exactly "\n\n".
        parts = result.split("\n\n")
        # With all 3 user parts + 3 boilerplate + header = 7 sections.
        assert len(parts) == 7


class TestBuildPromptEmptyParts:
    """Test prompt compilation when prepend and/or append are empty."""

    def test_empty_prepend_omitted(self):
        """An empty prepend should not produce a blank section."""
        result = build_prompt("", "A scene.", "Append.")
        # The result should start with the boilerplate, not an empty section.
        assert not result.startswith("\n")
        # Verify the structure: boilerplate, header, scene, mood, append, colour.
        parts = result.split("\n\n")
        assert parts[0] == _STYLE_BOILERPLATE

    def test_empty_append_omitted(self):
        """An empty append should not produce a blank section."""
        result = build_prompt("Prepend.", "A scene.", "")
        # Should end with colour boilerplate, not an empty section.
        assert result.strip().endswith(_COLOUR_BOILERPLATE)
        parts = result.split("\n\n")
        assert parts[-1] == _COLOUR_BOILERPLATE

    def test_both_empty(self):
        """When both prepend and append are empty, only fixed sections remain."""
        result = build_prompt("", "A scene.", "")
        parts = result.split("\n\n")
        # boilerplate, header, scene, mood, colour = 5 sections.
        assert len(parts) == 5

    def test_whitespace_only_prepend_treated_as_empty(self):
        """Whitespace-only prepend should be omitted."""
        result = build_prompt("   \t\n  ", "A scene.", "")
        parts = result.split("\n\n")
        # Same as empty — 5 sections.
        assert len(parts) == 5

    def test_whitespace_only_append_treated_as_empty(self):
        """Whitespace-only append should be omitted."""
        result = build_prompt("", "A scene.", "   \n  ")
        parts = result.split("\n\n")
        assert len(parts) == 5

    def test_scene_is_stripped(self):
        """Main scene should be stripped of leading/trailing whitespace."""
        result = build_prompt("", "  A padded scene.  ", "")
        assert "A padded scene." in result
        assert "  A padded scene.  " not in result

    def test_empty_scene_omits_main_scene_header(self):
        """An empty scene should not produce a blank Main Scene section."""
        result = build_prompt("", "", "")
        assert "Main Scene:" not in result
        assert "\n\n\n\n" not in result


class TestBuildPromptManualPrepend:
    """Test prompt compilation when prepend_mode is 'manual'."""

    def test_manual_prepend_omits_style_boilerplate(self):
        """Manual prepend mode should NOT include style boilerplate."""
        result = build_prompt(
            "My custom style.",
            "A scene.",
            "",
            prepend_mode="manual",
        )
        assert _STYLE_BOILERPLATE not in result
        assert "My custom style." in result

    def test_manual_prepend_keeps_mood_and_colour(self):
        """Manual prepend mode should still include mood and colour boilerplate."""
        result = build_prompt(
            "My custom style.",
            "A scene.",
            "",
            prepend_mode="manual",
        )
        assert _MOOD_BOILERPLATE in result
        assert _COLOUR_BOILERPLATE in result

    def test_manual_prepend_empty_omits_style(self):
        """Empty manual prepend should omit both the value and style boilerplate."""
        result = build_prompt("", "A scene.", "", prepend_mode="manual")
        assert _STYLE_BOILERPLATE not in result
        parts = result.split("\n\n")
        assert parts[0] == "Main Scene:"


class TestBuildPromptManualAppend:
    """Test prompt compilation when append_mode is 'manual'."""

    def test_manual_append_omits_mood_and_colour(self):
        """Manual append mode should NOT include mood or colour boilerplate."""
        result = build_prompt(
            "",
            "A scene.",
            "Cinematic lighting.",
            append_mode="manual",
        )
        assert _MOOD_BOILERPLATE not in result
        assert _COLOUR_BOILERPLATE not in result
        assert "Cinematic lighting." in result

    def test_manual_append_keeps_style_boilerplate(self):
        """Manual append mode should still include style boilerplate."""
        result = build_prompt(
            "",
            "A scene.",
            "Cinematic lighting.",
            append_mode="manual",
        )
        assert _STYLE_BOILERPLATE in result

    def test_manual_append_empty_omits_mood_and_colour(self):
        """Empty manual append should omit mood and colour boilerplate."""
        result = build_prompt("", "A scene.", "", append_mode="manual")
        assert _MOOD_BOILERPLATE not in result
        assert _COLOUR_BOILERPLATE not in result
        # Should end with the scene text.
        parts = result.split("\n\n")
        assert parts[-1] == "A scene."


class TestBuildPromptBothManual:
    """Test prompt compilation when both prepend and append are manual."""

    def test_both_manual_omits_all_boilerplate_and_header(self):
        """Both manual should produce only user text, no boilerplate or header."""
        result = build_prompt(
            "Custom style.",
            "A dragon.",
            "High contrast.",
            prepend_mode="manual",
            append_mode="manual",
        )
        assert _STYLE_BOILERPLATE not in result
        assert _MOOD_BOILERPLATE not in result
        assert _COLOUR_BOILERPLATE not in result
        assert "Main Scene:" not in result
        assert "Custom style." in result
        assert "A dragon." in result
        assert "High contrast." in result

    def test_both_manual_section_count(self):
        """Both manual with all parts should have 3 sections."""
        result = build_prompt(
            "Style.",
            "Scene.",
            "Modifier.",
            prepend_mode="manual",
            append_mode="manual",
        )
        parts = result.split("\n\n")
        # style, scene, modifier = 3 sections.
        assert len(parts) == 3

    def test_both_manual_empty_prepend_and_append(self):
        """Both manual with empty prepend/append should be scene only."""
        result = build_prompt(
            "",
            "Just a scene.",
            "",
            prepend_mode="manual",
            append_mode="manual",
        )
        parts = result.split("\n\n")
        # scene only = 1 section.
        assert len(parts) == 1
        assert parts[0] == "Just a scene."

    def test_both_manual_all_empty_returns_empty_string(self):
        """All-empty manual prompts should compile to an empty string."""
        result = build_prompt(
            "",
            "",
            "",
            prepend_mode="manual",
            append_mode="manual",
        )
        assert result == ""

    def test_main_scene_header_present_when_one_template(self):
        """Main Scene header should still appear if either mode is template."""
        result = build_prompt(
            "Custom.",
            "A scene.",
            "",
            prepend_mode="manual",
            append_mode="template",
        )
        assert "Main Scene:" in result


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

    def test_resolve_prompt_variants_expands_all_sections(self):
        """Prepend, scene, and append should all be expanded once."""
        prepend_value, main_scene, append_value = resolve_prompt_variants(
            "Style {ink|wash}",
            "A {red|blue} automaton",
            "With {fog|sparks}",
            rng=SequenceRandom(["wash", "red", "sparks"]),
        )
        assert prepend_value == "Style wash"
        assert main_scene == "A red automaton"
        assert append_value == "With sparks"

    def test_build_prompt_can_expand_placeholders(self):
        """build_prompt should expand placeholders by default."""
        result = build_prompt(
            "Style {ink|wash}",
            "A {red|blue} automaton.",
            "With {fog|sparks}.",
            prepend_mode="manual",
            append_mode="manual",
            rng=SequenceRandom(["wash", "blue", "fog"]),
        )
        assert "Style wash" in result
        assert "A blue automaton." in result
        assert "With fog." in result


class TestStructuredPromptBuilder:
    """Tests for the Subject/Setting/Details/Lighting/Atmosphere prompt schema."""

    def test_build_structured_prompt_omits_empty_sections(self):
        """Only non-empty section blocks should be emitted."""
        result = build_structured_prompt(
            {
                "subject": "A goblin inventor.",
                "setting": "",
                "details": "Oily hands and brass tools.",
                "lighting": "",
                "atmosphere": "Hushed and focused.",
            },
            expand_placeholders=False,
        )
        assert "Subject:" in result
        assert "Details:" in result
        assert "Atmosphere:" in result
        assert "Setting:" not in result
        assert "Lighting:" not in result

    def test_build_structured_prompt_expands_placeholders(self):
        """Structured prompt building should expand placeholders by default."""
        result = build_structured_prompt(
            {
                "subject": "A {goblin|human} inventor.",
                "setting": "In a {workshop|laboratory}.",
                "details": "",
                "lighting": "",
                "atmosphere": "",
            },
            rng=SequenceRandom(["human", "workshop"]),
        )
        assert "A human inventor." in result
        assert "In a workshop." in result

    def test_resolve_structured_prompt_variants(self):
        """Per-section placeholder expansion should resolve each section once."""
        resolved = resolve_structured_prompt_variants(
            {
                "subject": "{goblin|human}",
                "setting": "{city|coast}",
                "details": "{tool|ledger}",
                "lighting": "{warm|cold}",
                "atmosphere": "{calm|tense}",
            },
            rng=SequenceRandom(["goblin", "coast", "tool", "warm", "tense"]),
        )
        assert resolved["subject"] == "goblin"
        assert resolved["setting"] == "coast"
        assert resolved["details"] == "tool"
        assert resolved["lighting"] == "warm"
        assert resolved["atmosphere"] == "tense"


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
        # Both sections render with the fallback label.
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
        # Distinct blocks separated by double-newline.
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

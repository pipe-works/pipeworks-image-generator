"""Tests for pipeworks.api.prompt_builder — three-part prompt compilation.

Tests cover:
- Full prompt with all three parts (prepend, scene, append).
- Empty prepend/append are silently omitted.
- Whitespace-only values are treated as empty.
- Fixed boilerplate sections are always present.
- Double newline delimiters between sections.
- Main scene header is always included.
"""

from __future__ import annotations

from pipeworks.api.prompt_builder import (
    _COLOUR_BOILERPLATE,
    _MOOD_BOILERPLATE,
    _STYLE_BOILERPLATE,
    build_prompt,
)


class TestBuildPromptFullParts:
    """Test prompt compilation with all three user parts provided."""

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
        """Compiled prompt should always contain the Ledgerfall style boilerplate."""
        result = build_prompt("", "A scene.", "")
        assert _STYLE_BOILERPLATE in result

    def test_contains_mood_boilerplate(self):
        """Compiled prompt should always contain the mood boilerplate."""
        result = build_prompt("", "A scene.", "")
        assert _MOOD_BOILERPLATE in result

    def test_contains_colour_boilerplate(self):
        """Compiled prompt should always contain the colour palette boilerplate."""
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

"""Tests for delimiter behavior in UI handlers."""

import pytest

from pipeworks.core.prompt_builder import PromptBuilder
from pipeworks.ui.handlers.prompt import build_combined_prompt
from pipeworks.ui.models import SegmentConfig, UIState


@pytest.fixture
def setup_test_files(tmp_path):
    """Create test files for delimiter testing."""
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    # Create a test file with multiple lines
    test_file = inputs_dir / "test.txt"
    test_file.write_text("line1\nline2\nline3\n")

    return inputs_dir


@pytest.fixture
def test_state(setup_test_files):
    """Create a test UI state with prompt builder (minimal, no model loading)."""
    state = UIState()
    state.prompt_builder = PromptBuilder(setup_test_files)
    # Don't call initialize_ui_state - we only need prompt_builder for these tests
    return state


class TestDelimiterLabelsAndMapping:
    """Test delimiter label to value mapping."""

    def test_delimiter_label_default(self):
        """Test default delimiter is a descriptive label."""
        segment = SegmentConfig()
        assert segment.delimiter == "Space ( )"

    def test_get_delimiter_value_space(self):
        """Test space delimiter label converts correctly."""
        segment = SegmentConfig(delimiter="Space ( )")
        assert segment.get_delimiter_value() == " "

    def test_get_delimiter_value_comma(self):
        """Test comma delimiter label converts correctly."""
        segment = SegmentConfig(delimiter="Comma (,)")
        assert segment.get_delimiter_value() == ","

    def test_get_delimiter_value_period_space(self):
        """Test period-space delimiter label converts correctly."""
        segment = SegmentConfig(delimiter="Period-Space (. )")
        assert segment.get_delimiter_value() == ". "

    def test_get_delimiter_value_none(self):
        """Test none delimiter label converts to empty string."""
        segment = SegmentConfig(delimiter="None (no separator)")
        assert segment.get_delimiter_value() == ""


class TestSegmentDelimiterAppending:
    """Test that each segment appends its delimiter at the end."""

    def test_text_only_segment_appends_delimiter(self, test_state):
        """Test text-only segment appends its delimiter."""
        seg1 = SegmentConfig(text="beautiful landscape", delimiter="Comma (,)")
        seg2 = SegmentConfig()  # Empty
        seg3 = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, seg1, seg2, seg3, seg1, seg2, seg3, test_state
        )

        # Should have delimiter appended: "beautiful landscape,"
        assert result == "beautiful landscape,beautiful landscape,beautiful landscape,"

    def test_file_only_segment_appends_delimiter(self, test_state):
        """Test file-only segment appends its delimiter."""
        seg1 = SegmentConfig(file="test.txt", mode="Specific Line", line=1, delimiter="Period (.)")
        seg2 = SegmentConfig()  # Empty
        seg3 = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, seg1, seg2, seg3, seg1, seg2, seg3, test_state
        )

        # Should be "line1.line1.line1."
        assert result == "line1.line1.line1."

    def test_text_and_file_segment_appends_delimiter(self, test_state):
        """Test text+file segment appends delimiter at end."""
        seg1 = SegmentConfig(
            text="photo of",
            file="test.txt",
            mode="Specific Line",
            line=1,
            delimiter="Comma-Space (, )",
        )
        seg2 = SegmentConfig()  # Empty
        seg3 = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, seg1, seg2, seg3, seg1, seg2, seg3, test_state
        )

        # Should be "photo of line1, photo of line1, photo of line1, "
        assert result == "photo of line1, photo of line1, photo of line1, "


class TestSegmentConcatenation:
    """Test that segments are concatenated without additional delimiters."""

    def test_two_segments_concatenate_directly(self, test_state):
        """Test two segments with different delimiters concatenate directly."""
        seg1 = SegmentConfig(text="beautiful", delimiter="Comma (,)")
        seg2 = SegmentConfig(text="landscape", delimiter="Period (.)")
        seg3 = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, seg3, seg3, seg3, seg3, seg3, seg3, test_state
        )

        # Should be: "beautiful," + "landscape." with NO space or delimiter between
        assert result == "beautiful,landscape."

    def test_three_segments_different_delimiters(self, test_state):
        """Test three segments each with different delimiters."""
        seg1 = SegmentConfig(text="A", delimiter="Comma (,)")
        seg2 = SegmentConfig(text="B", delimiter="Period (.)")
        seg3 = SegmentConfig(text="C", delimiter="None (no separator)")
        empty = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, empty, empty, empty, empty, empty, empty, test_state
        )

        # Should be: "A," + "B." + "C" (no delimiter on last one)
        assert result == "A,B.C"


class TestEmptyDelimiter:
    """Test empty delimiter (no separator) option."""

    def test_empty_delimiter_no_separator(self, test_state):
        """Test empty delimiter adds no separator."""
        seg1 = SegmentConfig(text="beautiful", delimiter="None (no separator)")
        seg2 = SegmentConfig(text="landscape", delimiter="None (no separator)")
        seg3 = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, seg3, seg3, seg3, seg3, seg3, seg3, test_state
        )

        # Should concatenate directly with no separators
        assert result == "beautifullandscape"

    def test_mixed_empty_and_comma_delimiters(self, test_state):
        """Test mixing empty and comma delimiters."""
        seg1 = SegmentConfig(text="word1", delimiter="None (no separator)")
        seg2 = SegmentConfig(text="word2", delimiter="Comma (,)")
        seg3 = SegmentConfig(text="word3", delimiter="None (no separator)")
        empty = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, empty, empty, empty, empty, empty, empty, test_state
        )

        # Should be: "word1" + "word2," + "word3"
        assert result == "word1word2,word3"


class TestMultiLineFileDelimiters:
    """Test delimiters with multi-line file operations."""

    def test_line_range_uses_delimiter_within_file(self, test_state):
        """Test Line Range mode uses delimiter to join lines within file."""
        seg1 = SegmentConfig(
            file="test.txt",
            mode="Line Range",
            line=1,
            range_end=3,
            delimiter="Comma-Space (, )",
        )
        seg2 = SegmentConfig()  # Empty
        seg3 = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, seg1, seg2, seg3, seg1, seg2, seg3, test_state
        )

        # Lines joined with ", " AND delimiter appended: "line1, line2, line3, "
        # Repeated 3 times
        assert result == "line1, line2, line3, line1, line2, line3, line1, line2, line3, "

    def test_all_lines_uses_delimiter_within_file(self, test_state):
        """Test All Lines mode uses delimiter to join lines."""
        seg1 = SegmentConfig(file="test.txt", mode="All Lines", delimiter="Period (.)")
        seg2 = SegmentConfig()  # Empty
        seg3 = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, seg1, seg2, seg3, seg1, seg2, seg3, test_state
        )

        # Lines joined with "." AND delimiter appended: "line1.line2.line3."
        # Repeated 3 times
        assert result == "line1.line2.line3.line1.line2.line3.line1.line2.line3."


class TestRealWorldScenarios:
    """Test real-world prompt building scenarios."""

    def test_tags_style_prompt(self, test_state):
        """Test building a tags-style prompt with commas."""
        seg1 = SegmentConfig(text="beautiful landscape", delimiter="Comma-Space (, )")
        seg2 = SegmentConfig(text="sunset", delimiter="Comma-Space (, )")
        seg3 = SegmentConfig(text="8k uhd", delimiter="None (no separator)")
        empty = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, empty, empty, empty, empty, empty, empty, test_state
        )

        # Should create: "beautiful landscape, sunset, 8k uhd"
        assert result == "beautiful landscape, sunset, 8k uhd"

    def test_sentence_style_prompt(self, test_state):
        """Test building a sentence-style prompt with periods."""
        seg1 = SegmentConfig(text="A wizard in robes", delimiter="Period-Space (. )")
        seg2 = SegmentConfig(text="Standing in forest", delimiter="Period-Space (. )")
        seg3 = SegmentConfig(text="Magical atmosphere", delimiter="None (no separator)")
        empty = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, empty, empty, empty, empty, empty, empty, test_state
        )

        # Should create: "A wizard in robes. Standing in forest. Magical atmosphere"
        assert result == "A wizard in robes. Standing in forest. Magical atmosphere"


class TestBackwardCompatibility:
    """Test that old behavior still works when using space delimiter."""

    def test_space_delimiter_prevents_word_collision(self, test_state):
        """Test space delimiter still works as before."""
        seg1 = SegmentConfig(text="beautiful", delimiter="Space ( )")
        seg2 = SegmentConfig(text="landscape", delimiter="Space ( )")
        seg3 = SegmentConfig()  # Empty

        result = build_combined_prompt(
            seg1, seg2, seg3, seg3, seg3, seg3, seg3, seg3, seg3, test_state
        )

        # Should be: "beautiful " + "landscape " = "beautiful landscape "
        assert result == "beautiful landscape "
        assert "beautifullandscape" not in result  # No collision

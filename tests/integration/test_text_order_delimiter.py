"""Integration tests for text_order and delimiter features."""

import pytest

from pipeworks.core.prompt_builder import PromptBuilder
from pipeworks.ui.handlers.prompt import build_combined_prompt
from pipeworks.ui.models import SegmentConfig, UIState


@pytest.fixture
def test_prompt_builder(test_inputs_dir):
    """Create a PromptBuilder with test inputs directory."""
    return PromptBuilder(test_inputs_dir)


@pytest.fixture
def initialized_state(test_prompt_builder):
    """Create an initialized UI state with prompt builder."""
    state = UIState()
    state.prompt_builder = test_prompt_builder
    return state


@pytest.fixture
def empty_segments():
    """Create 9 empty segments."""
    return [SegmentConfig() for _ in range(9)]


class TestTextOrderFeature:
    """Tests for text_order functionality."""

    def test_text_first_order(self, initialized_state, test_inputs_dir, empty_segments):
        """Test text_first order places text before file content."""
        # Create test file
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("robed")

        start_1 = SegmentConfig(
            text="wizard",
            file="test.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Comma-Space (, )",  # Use label format
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        # New behavior: space between text and file, delimiter at end
        assert result == "wizard robed, "

    def test_file_first_order(self, initialized_state, test_inputs_dir, empty_segments):
        """Test file_first order places file content before text."""
        # Create test file
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("robed")

        start_1 = SegmentConfig(
            text="wizard",
            file="test.txt",
            mode="Random Line",
            text_order="file_first",
            delimiter="Comma-Space (, )",  # Use label format
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        # New behavior: space between file and text, delimiter at end
        assert result == "robed wizard, "

    def test_text_only_unchanged(self, initialized_state, empty_segments):
        """Test text-only segments work unchanged."""
        start_1 = SegmentConfig(
            text="wizard", text_order="text_first", delimiter="Comma-Space (, )"
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        # New behavior: delimiter appended to text
        assert result == "wizard, "

    def test_file_only_unchanged(self, initialized_state, test_inputs_dir, empty_segments):
        """Test file-only segments work unchanged."""
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("robed")

        start_1 = SegmentConfig(
            file="test.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Comma-Space (, )",
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        # New behavior: delimiter appended to file content
        assert result == "robed, "


class TestDelimiterFeature:
    """Tests for delimiter functionality."""

    def test_default_delimiter_comma_space(
        self, initialized_state, test_inputs_dir, empty_segments
    ):
        """Test default delimiter ', ' (comma-space)."""
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("robed")

        start_1 = SegmentConfig(
            text="wizard",
            file="test.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Comma-Space (, )",  # Default
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        assert result == "wizard robed, "

    def test_delimiter_period_space(self, initialized_state, test_inputs_dir, empty_segments):
        """Test delimiter '. ' (period-space) for sentences."""
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("wearing robes")

        start_1 = SegmentConfig(
            text="A wizard",
            file="test.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Period-Space (. )",
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        assert result == "A wizard wearing robes. "

    def test_delimiter_single_space(self, initialized_state, test_inputs_dir, empty_segments):
        """Test delimiter ' ' (single space)."""
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("robed")

        start_1 = SegmentConfig(
            text="wizard",
            file="test.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Space ( )",
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        assert result == "wizard robed "

    def test_delimiter_comma_only(self, initialized_state, test_inputs_dir, empty_segments):
        """Test delimiter ',' (comma only)."""
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("robed")

        start_1 = SegmentConfig(
            text="wizard",
            file="test.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Comma (,)",
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        assert result == "wizard robed,"

    def test_delimiter_period_only(self, initialized_state, test_inputs_dir, empty_segments):
        """Test delimiter '.' (period only)."""
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("robed")

        start_1 = SegmentConfig(
            text="wizard",
            file="test.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Period (.)",
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        assert result == "wizard robed."


class TestMultipleSegments:
    """Tests for multiple segments with different settings."""

    def test_different_delimiters_per_segment(
        self, initialized_state, test_inputs_dir, empty_segments
    ):
        """Test that each segment can have its own delimiter."""
        # Create test files
        (test_inputs_dir / "file1.txt").write_text("robed")
        (test_inputs_dir / "file2.txt").write_text("old")

        start_1 = SegmentConfig(
            text="wizard",
            file="file1.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Period-Space (. )",  # Period-space
        )

        start_2 = SegmentConfig(
            text="castle",
            file="file2.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Comma-Space (, )",  # Comma-space
        )

        result = build_combined_prompt(
            start_1, start_2, *empty_segments[2:], state=initialized_state
        )

        # New behavior: space between text and file, delimiter appended
        # Segment 1: "wizard robed. " Segment 2: "castle old, "
        assert result == "wizard robed. castle old, "

    def test_mixed_text_order_per_segment(self, initialized_state, test_inputs_dir, empty_segments):
        """Test that each segment can have its own text_order."""
        (test_inputs_dir / "file1.txt").write_text("robed")
        (test_inputs_dir / "file2.txt").write_text("ancient")

        start_1 = SegmentConfig(
            text="wizard",
            file="file1.txt",
            mode="Random Line",
            text_order="text_first",  # wizard robed
            delimiter="Comma-Space (, )",
        )

        start_2 = SegmentConfig(
            text="castle",
            file="file2.txt",
            mode="Random Line",
            text_order="file_first",  # ancient castle
            delimiter="Comma-Space (, )",
        )

        result = build_combined_prompt(
            start_1, start_2, *empty_segments[2:], state=initialized_state
        )

        # New behavior: space within segments, delimiter at end
        assert result == "wizard robed, ancient castle, "


class TestBackwardCompatibility:
    """Tests for backward compatibility with defaults."""

    def test_defaults_match_original_behavior(
        self, initialized_state, test_inputs_dir, empty_segments
    ):
        """Test that default values work correctly."""
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("robed")

        # Using defaults (text_first, "Space ( )")
        start_1 = SegmentConfig(
            text="wizard",
            file="test.txt",
            mode="Random Line",
            # text_order defaults to "text_first"
            # delimiter defaults to "Space ( )"
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        # New behavior: text first, space between, space delimiter at end
        assert result == "wizard robed "

    def test_empty_segments_still_skipped(self, initialized_state, empty_segments):
        """Test that empty segments are still skipped."""
        result = build_combined_prompt(*empty_segments, state=initialized_state)

        assert result == ""


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_file_read_failure_falls_back_to_text(
        self, initialized_state, test_inputs_dir, empty_segments
    ):
        """Test that if file doesn't exist, falls back to text."""
        start_1 = SegmentConfig(
            text="wizard",
            file="nonexistent.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Comma-Space (, )",
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        # Should fall back to text only with delimiter
        assert result == "wizard, "

    def test_whitespace_stripped(self, initialized_state, test_inputs_dir, empty_segments):
        """Test that whitespace is properly stripped."""
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("  robed  ")

        start_1 = SegmentConfig(
            text="  wizard  ",
            file="test.txt",
            mode="Random Line",
            text_order="text_first",
            delimiter="Comma-Space (, )",
        )

        result = build_combined_prompt(start_1, *empty_segments[1:], state=initialized_state)

        # Whitespace stripped, then combined with space and delimiter
        assert result == "wizard robed, "

    def test_all_file_modes_work_with_text_and_delimiter(
        self, initialized_state, test_inputs_dir, empty_segments
    ):
        """Test that all file modes work with text_order and delimiter."""
        test_file = test_inputs_dir / "test.txt"
        test_file.write_text("line1\nline2\nline3")

        # Test Specific Line - delimiter goes at END, not between text and file
        cfg = SegmentConfig(
            text="prefix", file="test.txt", mode="Specific Line", line=2, delimiter="Comma (,)"
        )
        result = build_combined_prompt(cfg, *empty_segments[1:], state=initialized_state)
        assert result == "prefix line2,"

        # Test Line Range - delimiter used within file lines AND at end
        cfg = SegmentConfig(
            text="prefix",
            file="test.txt",
            mode="Line Range",
            line=1,
            range_end=2,
            delimiter="Comma (,)",
        )
        result = build_combined_prompt(cfg, *empty_segments[1:], state=initialized_state)
        assert result == "prefix line1,line2,"

        # Test All Lines - delimiter used within file lines AND at end
        cfg = SegmentConfig(
            text="prefix", file="test.txt", mode="All Lines", delimiter="Comma (,)"
        )
        result = build_combined_prompt(cfg, *empty_segments[1:], state=initialized_state)
        assert result == "prefix line1,line2,line3,"

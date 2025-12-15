"""Integration tests for UI components."""

import pytest
from unittest.mock import Mock, patch
import gradio as gr

from pipeworks.ui.components import (
    SegmentUI,
    update_mode_visibility,
    create_three_segments,
)
from pipeworks.ui.models import SegmentConfig


class TestSegmentUI:
    """Integration tests for SegmentUI component."""

    def test_segment_ui_creation(self):
        """Test that SegmentUI can be created."""
        with patch('gradio.Group'), \
             patch('gradio.Markdown') as MockMarkdown, \
             patch('gradio.Textbox') as MockTextbox, \
             patch('gradio.Dropdown') as MockDropdown, \
             patch('gradio.Checkbox') as MockCheckbox, \
             patch('gradio.Number') as MockNumber, \
             patch('gradio.State') as MockState, \
             patch('gradio.Row'):

            segment = SegmentUI("Test", ["(None)", "file1.txt"])

            assert segment.name == "Test"
            assert segment.title is not None
            assert segment.text is not None
            assert segment.file is not None

    def test_get_input_components(self):
        """Test that get_input_components returns correct components."""
        with patch('gradio.Group'), \
             patch('gradio.Markdown'), \
             patch('gradio.Textbox'), \
             patch('gradio.Dropdown'), \
             patch('gradio.Checkbox'), \
             patch('gradio.Number'), \
             patch('gradio.State'), \
             patch('gradio.Row'):

            segment = SegmentUI("Test", ["(None)"])
            inputs = segment.get_input_components()

            assert len(inputs) == 8  # text, path_state, file, mode, line, range_end, count, dynamic

    def test_get_output_components(self):
        """Test that get_output_components returns correct components."""
        with patch('gradio.Group'), \
             patch('gradio.Markdown'), \
             patch('gradio.Textbox'), \
             patch('gradio.Dropdown'), \
             patch('gradio.Checkbox'), \
             patch('gradio.Number'), \
             patch('gradio.State'), \
             patch('gradio.Row'):

            segment = SegmentUI("Test", ["(None)"])
            outputs = segment.get_output_components()

            assert len(outputs) == 4  # title, path_display, file, path_state

    def test_get_navigation_components(self):
        """Test that get_navigation_components returns correct tuple."""
        with patch('gradio.Group'), \
             patch('gradio.Markdown'), \
             patch('gradio.Textbox'), \
             patch('gradio.Dropdown'), \
             patch('gradio.Checkbox'), \
             patch('gradio.Number'), \
             patch('gradio.State'), \
             patch('gradio.Row'):

            segment = SegmentUI("Test", ["(None)"])
            file, path_state, path_display = segment.get_navigation_components()

            assert file is segment.file
            assert path_state is segment.path_state
            assert path_display is segment.path_display

    def test_get_mode_visibility_outputs(self):
        """Test that get_mode_visibility_outputs returns correct tuple."""
        with patch('gradio.Group'), \
             patch('gradio.Markdown'), \
             patch('gradio.Textbox'), \
             patch('gradio.Dropdown'), \
             patch('gradio.Checkbox'), \
             patch('gradio.Number'), \
             patch('gradio.State'), \
             patch('gradio.Row'):

            segment = SegmentUI("Test", ["(None)"])
            line, range_end, count = segment.get_mode_visibility_outputs()

            assert line is segment.line
            assert range_end is segment.range_end
            assert count is segment.count

    def test_values_to_config(self):
        """Test that values_to_config creates correct SegmentConfig."""
        config = SegmentUI.values_to_config(
            "test text",
            "subfolder",
            "file.txt",
            "Specific Line",
            5,
            10,
            3,
            True
        )

        assert isinstance(config, SegmentConfig)
        assert config.text == "test text"
        assert config.path == "subfolder"
        assert config.file == "file.txt"
        assert config.mode == "Specific Line"
        assert config.line == 5
        assert config.range_end == 10
        assert config.count == 3
        assert config.dynamic is True

    def test_values_to_config_handles_none_numbers(self):
        """Test that values_to_config handles None number values."""
        config = SegmentUI.values_to_config(
            "text",
            "",
            "(None)",
            "Random Line",
            None,  # line
            None,  # range_end
            None,  # count
            False
        )

        assert config.line == 1  # Default
        assert config.range_end == 1  # Default
        assert config.count == 1  # Default

    def test_format_title_unconfigured(self):
        """Test format_title for unconfigured segment."""
        title = SegmentUI.format_title("Test", "(None)", "Random Line", False)

        assert "Test" in title
        assert "color" not in title  # No color styling

    def test_format_title_configured(self):
        """Test format_title for configured segment."""
        title = SegmentUI.format_title("Test", "file.txt", "Random Line", False)

        assert "Test" in title
        assert "#22c55e" in title  # Green color
        assert "Random Line" in title

    def test_format_title_configured_with_dynamic(self):
        """Test format_title for configured segment with dynamic."""
        title = SegmentUI.format_title("Test", "file.txt", "Specific Line", True)

        assert "Test" in title
        assert "#22c55e" in title
        assert "Specific Line" in title
        assert "Dynamic" in title

    def test_format_title_folder_not_configured(self):
        """Test format_title treats folder as unconfigured."""
        title = SegmentUI.format_title("Test", "📁 folder", "Random Line", False)

        assert "Test" in title
        assert "color" not in title  # No color styling


class TestUpdateModeVisibility:
    """Tests for update_mode_visibility function."""

    def test_random_line_hides_all(self):
        """Test that Random Line mode hides all number inputs."""
        line, range_end, count = update_mode_visibility("Random Line")

        # Should return gr.update() objects
        assert line == gr.update(visible=False)
        assert range_end == gr.update(visible=False)
        assert count == gr.update(visible=False)

    def test_specific_line_shows_line_only(self):
        """Test that Specific Line mode shows line input only."""
        line, range_end, count = update_mode_visibility("Specific Line")

        assert line == gr.update(visible=True)
        assert range_end == gr.update(visible=False)
        assert count == gr.update(visible=False)

    def test_line_range_shows_line_and_range(self):
        """Test that Line Range mode shows line and range_end."""
        line, range_end, count = update_mode_visibility("Line Range")

        assert line == gr.update(visible=True)
        assert range_end == gr.update(visible=True)
        assert count == gr.update(visible=False)

    def test_random_multiple_shows_count_only(self):
        """Test that Random Multiple mode shows count input only."""
        line, range_end, count = update_mode_visibility("Random Multiple")

        assert line == gr.update(visible=False)
        assert range_end == gr.update(visible=False)
        assert count == gr.update(visible=True)

    def test_all_lines_hides_all(self):
        """Test that All Lines mode hides all number inputs."""
        line, range_end, count = update_mode_visibility("All Lines")

        assert line == gr.update(visible=False)
        assert range_end == gr.update(visible=False)
        assert count == gr.update(visible=False)


class TestCreateThreeSegments:
    """Tests for create_three_segments function."""

    def test_creates_three_segments(self):
        """Test that three SegmentUI instances are created."""
        with patch('gradio.Group'), \
             patch('gradio.Markdown'), \
             patch('gradio.Textbox'), \
             patch('gradio.Dropdown'), \
             patch('gradio.Checkbox'), \
             patch('gradio.Number'), \
             patch('gradio.State'), \
             patch('gradio.Row'):

            start, middle, end = create_three_segments(["(None)", "file.txt"])

            assert isinstance(start, SegmentUI)
            assert isinstance(middle, SegmentUI)
            assert isinstance(end, SegmentUI)

    def test_segments_have_correct_names(self):
        """Test that segments have correct names."""
        with patch('gradio.Group'), \
             patch('gradio.Markdown'), \
             patch('gradio.Textbox'), \
             patch('gradio.Dropdown'), \
             patch('gradio.Checkbox'), \
             patch('gradio.Number'), \
             patch('gradio.State'), \
             patch('gradio.Row'):

            start, middle, end = create_three_segments(["(None)"])

            assert start.name == "Start"
            assert middle.name == "Middle"
            assert end.name == "End"

    def test_segments_receive_same_choices(self):
        """Test that all segments receive the same initial choices."""
        choices = ["(None)", "file1.txt", "file2.txt"]

        with patch('gradio.Group'), \
             patch('gradio.Markdown'), \
             patch('gradio.Textbox'), \
             patch('gradio.Dropdown') as MockDropdown, \
             patch('gradio.Checkbox'), \
             patch('gradio.Number'), \
             patch('gradio.State'), \
             patch('gradio.Row'):

            start, middle, end = create_three_segments(choices)

            # Each segment should have created a Dropdown with the choices
            # (called multiple times per segment for mode and file dropdowns)
            assert MockDropdown.call_count >= 6  # 2 dropdowns × 3 segments

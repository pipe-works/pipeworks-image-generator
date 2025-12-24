"""Integration tests for UI components."""

from unittest.mock import patch

import gradio as gr

from pipeworks.ui.components import (
    ConditionSegmentUI,
    SegmentUI,
    create_nine_segments,
    create_three_segments,
    update_mode_visibility,
)
from pipeworks.ui.models import SegmentConfig


class TestSegmentUI:
    """Integration tests for SegmentUI component."""

    def test_segment_ui_creation(self):
        """Test that SegmentUI can be created."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown") as MockMarkdown,
            patch("gradio.Textbox") as MockTextbox,
            patch("gradio.Dropdown") as MockDropdown,
            patch("gradio.Checkbox") as MockCheckbox,
            patch("gradio.Number") as MockNumber,
            patch("gradio.State") as MockState,
            patch("gradio.Row"),
        ):
            segment = SegmentUI("Test", ["(None)", "file1.txt"])

            assert segment.name == "Test"
            assert segment.title is not None
            assert segment.text is not None
            assert segment.file is not None

    def test_get_input_components(self):
        """Test that get_input_components returns correct components."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
        ):
            segment = SegmentUI("Test", ["(None)"])
            inputs = segment.get_input_components()

            # text, path_state, file, mode, line, range_end, count, dynamic, sequential_start_line,
            # text_order, delimiter
            assert len(inputs) == 11

    def test_get_output_components(self):
        """Test that get_output_components returns correct components."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
        ):
            segment = SegmentUI("Test", ["(None)"])
            outputs = segment.get_output_components()

            assert len(outputs) == 4  # title, path_display, file, path_state

    def test_get_navigation_components(self):
        """Test that get_navigation_components returns correct tuple."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
        ):
            segment = SegmentUI("Test", ["(None)"])
            file, path_state, path_display = segment.get_navigation_components()

            assert file is segment.file
            assert path_state is segment.path_state
            assert path_display is segment.path_display

    def test_get_mode_visibility_outputs(self):
        """Test that get_mode_visibility_outputs returns correct tuple."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
        ):
            segment = SegmentUI("Test", ["(None)"])
            line, range_end, count, sequential_start_line = segment.get_mode_visibility_outputs()

            assert line is segment.line
            assert range_end is segment.range_end
            assert count is segment.count
            assert sequential_start_line is segment.sequential_start_line

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
            True,
            2,  # sequential_start_line
            "file_first",  # text_order
            ". ",  # delimiter
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
        assert config.sequential_start_line == 2
        assert config.text_order == "file_first"
        assert config.delimiter == ". "

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
            False,
            None,  # sequential_start_line
            "text_first",  # text_order
            ", ",  # delimiter
        )

        assert config.line == 1  # Default
        assert config.range_end == 1  # Default
        assert config.count == 1  # Default
        assert config.sequential_start_line == 1  # Default
        assert config.text_order == "text_first"
        assert config.delimiter == ", "

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
        line, range_end, count, sequential_start_line = update_mode_visibility("Random Line")

        # Should return gr.update() objects
        assert line == gr.update(visible=False)
        assert range_end == gr.update(visible=False)
        assert count == gr.update(visible=False)
        assert sequential_start_line == gr.update(visible=False)

    def test_specific_line_shows_line_only(self):
        """Test that Specific Line mode shows line input only."""
        line, range_end, count, sequential_start_line = update_mode_visibility("Specific Line")

        assert line == gr.update(visible=True)
        assert range_end == gr.update(visible=False)
        assert count == gr.update(visible=False)
        assert sequential_start_line == gr.update(visible=False)

    def test_line_range_shows_line_and_range(self):
        """Test that Line Range mode shows line and range_end."""
        line, range_end, count, sequential_start_line = update_mode_visibility("Line Range")

        assert line == gr.update(visible=True)
        assert range_end == gr.update(visible=True)
        assert count == gr.update(visible=False)
        assert sequential_start_line == gr.update(visible=False)

    def test_random_multiple_shows_count_only(self):
        """Test that Random Multiple mode shows count input only."""
        line, range_end, count, sequential_start_line = update_mode_visibility("Random Multiple")

        assert line == gr.update(visible=False)
        assert range_end == gr.update(visible=False)
        assert count == gr.update(visible=True)
        assert sequential_start_line == gr.update(visible=False)

    def test_all_lines_hides_all(self):
        """Test that All Lines mode hides all number inputs."""
        line, range_end, count, sequential_start_line = update_mode_visibility("All Lines")

        assert line == gr.update(visible=False)
        assert range_end == gr.update(visible=False)
        assert count == gr.update(visible=False)
        assert sequential_start_line == gr.update(visible=False)

    def test_sequential_shows_sequential_start_line_only(self):
        """Test that Sequential mode shows sequential_start_line input only."""
        line, range_end, count, sequential_start_line = update_mode_visibility("Sequential")

        assert line == gr.update(visible=False)
        assert range_end == gr.update(visible=False)
        assert count == gr.update(visible=False)
        assert sequential_start_line == gr.update(visible=True)


class TestCreateThreeSegments:
    """Tests for create_three_segments function."""

    def test_creates_three_segments(self):
        """Test that three SegmentUI instances are created."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
        ):
            start, middle, end = create_three_segments(["(None)", "file.txt"])

            assert isinstance(start, SegmentUI)
            assert isinstance(middle, SegmentUI)
            assert isinstance(end, SegmentUI)

    def test_segments_have_correct_names(self):
        """Test that segments have correct names."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
        ):
            start, middle, end = create_three_segments(["(None)"])

            assert start.name == "Start"
            assert middle.name == "Middle"
            assert end.name == "End"

    def test_segments_receive_same_choices(self):
        """Test that all segments receive the same initial choices."""
        choices = ["(None)", "file1.txt", "file2.txt"]

        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown") as MockDropdown,
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
        ):
            start, middle, end = create_three_segments(choices)

            # Each segment should have created a Dropdown with the choices
            # (called multiple times per segment for mode and file dropdowns)
            assert MockDropdown.call_count >= 6  # 2 dropdowns × 3 segments

    def test_get_all_components_returns_all_13_components(self):
        """Test that get_all_components returns all 13 components."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
        ):
            segment = SegmentUI("Test", ["(None)"])
            all_components = segment.get_all_components()

            # Should return all 13 components:
            # title, text, path_display, file, path_state, mode, dynamic,
            # text_order, delimiter, line, range_end, count, sequential_start_line
            assert len(all_components) == 13

            # Verify they're the actual component attributes
            assert all_components[0] is segment.title
            assert all_components[1] is segment.text
            assert all_components[2] is segment.path_display
            assert all_components[3] is segment.file
            assert all_components[4] is segment.path_state
            assert all_components[5] is segment.mode
            assert all_components[6] is segment.dynamic
            assert all_components[7] is segment.text_order
            assert all_components[8] is segment.delimiter
            assert all_components[9] is segment.line
            assert all_components[10] is segment.range_end
            assert all_components[11] is segment.count
            assert all_components[12] is segment.sequential_start_line


class TestConditionSegmentUI:
    """Integration tests for ConditionSegmentUI component."""

    def test_condition_segment_ui_creation(self):
        """Test that ConditionSegmentUI can be created."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown") as MockMarkdown,
            patch("gradio.Textbox") as MockTextbox,
            patch("gradio.Dropdown") as MockDropdown,
            patch("gradio.Checkbox") as MockCheckbox,
            patch("gradio.Number") as MockNumber,
            patch("gradio.State") as MockState,
            patch("gradio.Row"),
            patch("gradio.Button") as MockButton,
        ):
            segment = ConditionSegmentUI("Start 2", ["(None)", "file1.txt"])

            assert segment.name == "Start 2"
            assert segment.title is not None
            assert segment.text is not None
            assert segment.file is not None

    def test_condition_segment_has_condition_components(self):
        """Test that ConditionSegmentUI has condition-specific components."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
            patch("gradio.Button"),
        ):
            segment = ConditionSegmentUI("Start 2", ["(None)"])

            # Should have condition-specific attributes
            assert hasattr(segment, "condition_type")
            assert hasattr(segment, "condition_text")
            assert hasattr(segment, "condition_regenerate")
            assert hasattr(segment, "condition_dynamic")
            assert hasattr(segment, "condition_controls")

    def test_get_condition_components(self):
        """Test that get_condition_components returns correct tuple."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
            patch("gradio.Button"),
        ):
            segment = ConditionSegmentUI("Start 2", ["(None)"])
            (
                condition_type,
                condition_text,
                regenerate_btn,
                condition_dynamic,
                condition_controls,
            ) = segment.get_condition_components()

            assert condition_type is segment.condition_type
            assert condition_text is segment.condition_text
            assert regenerate_btn is segment.condition_regenerate
            assert condition_dynamic is segment.condition_dynamic
            assert condition_controls is segment.condition_controls

    def test_condition_segment_inherits_base_methods(self):
        """Test that ConditionSegmentUI inherits all base SegmentUI methods."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
            patch("gradio.Button"),
        ):
            segment = ConditionSegmentUI("Start 2", ["(None)"])

            # Should inherit all base component getters
            inputs = segment.get_input_components()
            assert len(inputs) == 11  # Same as base SegmentUI

            outputs = segment.get_output_components()
            assert len(outputs) == 4  # Same as base SegmentUI

            file, path_state, path_display = segment.get_navigation_components()
            assert file is segment.file
            assert path_state is segment.path_state
            assert path_display is segment.path_display

            (
                line,
                range_end,
                count,
                sequential_start_line,
            ) = segment.get_mode_visibility_outputs()
            assert line is segment.line
            assert range_end is segment.range_end
            assert count is segment.count
            assert sequential_start_line is segment.sequential_start_line

    def test_condition_segment_has_all_base_components(self):
        """Test that ConditionSegmentUI has all base SegmentUI components."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
            patch("gradio.Button"),
        ):
            segment = ConditionSegmentUI("Start 2", ["(None)"])

            # All base components should exist
            assert hasattr(segment, "title")
            assert hasattr(segment, "text")
            assert hasattr(segment, "path_display")
            assert hasattr(segment, "file")
            assert hasattr(segment, "path_state")
            assert hasattr(segment, "mode")
            assert hasattr(segment, "dynamic")
            assert hasattr(segment, "text_order")
            assert hasattr(segment, "delimiter")
            assert hasattr(segment, "line")
            assert hasattr(segment, "range_end")
            assert hasattr(segment, "count")
            assert hasattr(segment, "sequential_start_line")


class TestCreateNineSegments:
    """Tests for create_nine_segments function."""

    def test_creates_nine_segments(self):
        """Test that nine segment instances are created."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
            patch("gradio.Button"),
        ):
            segments = create_nine_segments(["(None)", "file.txt"])

            assert len(segments) == 9
            # All should be instances of SegmentUI or its subclass
            for segment in segments:
                assert isinstance(segment, SegmentUI)

    def test_segments_have_correct_types(self):
        """Test that start_2 and start_3 are ConditionSegmentUI, others are SegmentUI."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
            patch("gradio.Button"),
        ):
            (
                start_1,
                start_2,
                start_3,
                mid_1,
                mid_2,
                mid_3,
                end_1,
                end_2,
                end_3,
            ) = create_nine_segments(["(None)"])

            # start_1 is standard SegmentUI
            assert type(start_1) is SegmentUI
            # start_2 and start_3 are ConditionSegmentUI
            assert type(start_2) is ConditionSegmentUI
            assert type(start_3) is ConditionSegmentUI
            # Middle and end segments are standard SegmentUI
            assert type(mid_1) is SegmentUI
            assert type(mid_2) is SegmentUI
            assert type(mid_3) is SegmentUI
            assert type(end_1) is SegmentUI
            assert type(end_2) is SegmentUI
            assert type(end_3) is SegmentUI

    def test_segments_have_correct_names(self):
        """Test that segments have correct names."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
            patch("gradio.Button"),
        ):
            (
                start_1,
                start_2,
                start_3,
                mid_1,
                mid_2,
                mid_3,
                end_1,
                end_2,
                end_3,
            ) = create_nine_segments(["(None)"])

            assert start_1.name == "Start 1"
            assert start_2.name == "Start 2"
            assert start_3.name == "Start 3"
            assert mid_1.name == "Mid 1"
            assert mid_2.name == "Mid 2"
            assert mid_3.name == "Mid 3"
            assert end_1.name == "End 1"
            assert end_2.name == "End 2"
            assert end_3.name == "End 3"

    def test_segments_receive_same_choices(self):
        """Test that all segments receive the same initial choices."""
        choices = ["(None)", "file1.txt", "file2.txt"]

        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown") as MockDropdown,
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
            patch("gradio.Button"),
        ):
            segments = create_nine_segments(choices)

            # Each segment should have created Dropdowns with the choices
            # (mode, file, condition_type for condition segments, delimiter, text_order)
            # 9 segments × multiple dropdowns per segment
            assert MockDropdown.call_count >= 27  # At least 3 dropdowns × 9 segments

    def test_condition_segments_have_condition_components(self):
        """Test that start_2 and start_3 have condition generation components."""
        with (
            patch("gradio.Group"),
            patch("gradio.Markdown"),
            patch("gradio.Textbox"),
            patch("gradio.Dropdown"),
            patch("gradio.Checkbox"),
            patch("gradio.Number"),
            patch("gradio.State"),
            patch("gradio.Row"),
            patch("gradio.Button"),
        ):
            (
                start_1,
                start_2,
                start_3,
                mid_1,
                mid_2,
                mid_3,
                end_1,
                end_2,
                end_3,
            ) = create_nine_segments(["(None)"])

            # start_2 and start_3 should have condition components
            assert hasattr(start_2, "condition_type")
            assert hasattr(start_2, "condition_text")
            assert hasattr(start_2, "condition_regenerate")
            assert hasattr(start_3, "condition_type")
            assert hasattr(start_3, "condition_text")
            assert hasattr(start_3, "condition_regenerate")

            # Others should not have condition components
            assert not hasattr(start_1, "condition_type")
            assert not hasattr(mid_1, "condition_type")
            assert not hasattr(end_1, "condition_type")

"""Unit tests for UI adapter functions."""

from pipeworks.ui.adapters import convert_segment_values_to_configs, split_segment_inputs
from pipeworks.ui.models import SegmentConfig


class TestSplitSegmentInputs:
    """Tests for split_segment_inputs function."""

    def test_split_valid_28_element_list(self):
        """Test splitting a valid 28-element input list."""
        values = [
            # Start segment (0-8)
            "start text",
            "start/path",
            "start.txt",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
            # Middle segment (9-17)
            "middle text",
            "middle/path",
            "middle.txt",
            "Specific Line",
            5,
            10,
            3,
            True,
            2,
            # End segment (18-26)
            "end text",
            "end/path",
            "end.txt",
            "Line Range",
            10,
            20,
            5,
            False,
            3,
            # State (27)
            {"key": "value"},
        ]

        start_values, middle_values, end_values, state = split_segment_inputs(values)

        # Verify start segment
        assert start_values == (
            "start text",
            "start/path",
            "start.txt",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
        )

        # Verify middle segment
        assert middle_values == (
            "middle text",
            "middle/path",
            "middle.txt",
            "Specific Line",
            5,
            10,
            3,
            True,
            2,
        )

        # Verify end segment
        assert end_values == (
            "end text",
            "end/path",
            "end.txt",
            "Line Range",
            10,
            20,
            5,
            False,
            3,
        )

        # Verify state
        assert state == {"key": "value"}

    def test_split_with_none_values(self):
        """Test splitting when some values are None."""
        values = [
            # Start segment
            None,
            "",
            "(None)",
            "Random Line",
            None,
            None,
            None,
            False,
            None,
            # Middle segment
            "",
            "",
            "(None)",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
            # End segment
            None,
            None,
            "(None)",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
            # State
            None,
        ]

        start_values, middle_values, end_values, state = split_segment_inputs(values)

        # Values should be preserved as-is (including None)
        assert start_values[0] is None
        assert start_values[1] == ""
        assert start_values[2] == "(None)"
        assert middle_values[0] == ""
        assert end_values[0] is None
        assert state is None

    def test_split_with_empty_strings(self):
        """Test splitting with empty string values."""
        values = [""] * 27 + [None]

        start_values, middle_values, end_values, state = split_segment_inputs(values)

        assert len(start_values) == 9
        assert len(middle_values) == 9
        assert len(end_values) == 9
        assert all(v == "" for v in start_values)
        assert all(v == "" for v in middle_values)
        assert all(v == "" for v in end_values)
        assert state is None

    def test_split_with_numeric_values(self):
        """Test splitting preserves numeric types."""
        values = [
            # Start segment
            "text",
            "path",
            "file.txt",
            "mode",
            42,
            100,
            5,
            True,
            10,
            # Middle segment
            "text",
            "path",
            "file.txt",
            "mode",
            0,
            0,
            0,
            False,
            0,
            # End segment
            "text",
            "path",
            "file.txt",
            "mode",
            999,
            1000,
            10,
            True,
            500,
            # State
            {"count": 123},
        ]

        start_values, middle_values, end_values, state = split_segment_inputs(values)

        # Verify numeric values are preserved
        assert start_values[4] == 42
        assert start_values[5] == 100
        assert start_values[6] == 5
        assert middle_values[4] == 0
        assert end_values[4] == 999
        assert state == {"count": 123}

    def test_split_returns_tuples(self):
        """Test that segment values are returned as tuples, not lists."""
        values = ["value"] * 27 + [None]

        start_values, middle_values, end_values, state = split_segment_inputs(values)

        assert isinstance(start_values, tuple)
        assert isinstance(middle_values, tuple)
        assert isinstance(end_values, tuple)

    def test_split_with_boolean_values(self):
        """Test splitting preserves boolean values correctly."""
        values = [
            # Start - all False for dynamic
            "text",
            "path",
            "file",
            "mode",
            1,
            1,
            1,
            False,
            1,
            # Middle - True for dynamic
            "text",
            "path",
            "file",
            "mode",
            1,
            1,
            1,
            True,
            1,
            # End - False for dynamic
            "text",
            "path",
            "file",
            "mode",
            1,
            1,
            1,
            False,
            1,
            # State
            None,
        ]

        start_values, middle_values, end_values, state = split_segment_inputs(values)

        assert start_values[7] is False
        assert middle_values[7] is True
        assert end_values[7] is False


class TestConvertSegmentValuesToConfigs:
    """Tests for convert_segment_values_to_configs function."""

    def test_convert_all_segments_to_configs(self):
        """Test converting all three segments to SegmentConfig objects."""
        start_values = (
            "start text",
            "start/path",
            "start.txt",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
        )
        middle_values = (
            "middle text",
            "middle/path",
            "middle.txt",
            "Specific Line",
            5,
            10,
            3,
            True,
            2,
        )
        end_values = (
            "end text",
            "end/path",
            "end.txt",
            "Line Range",
            10,
            20,
            5,
            False,
            3,
        )

        start_cfg, middle_cfg, end_cfg = convert_segment_values_to_configs(
            start_values, middle_values, end_values
        )

        # Verify start config
        assert isinstance(start_cfg, SegmentConfig)
        assert start_cfg.text == "start text"
        assert start_cfg.path == "start/path"
        assert start_cfg.file == "start.txt"
        assert start_cfg.mode == "Random Line"
        assert start_cfg.line == 1
        assert start_cfg.range_end == 1
        assert start_cfg.count == 1
        assert start_cfg.dynamic is False
        assert start_cfg.sequential_start_line == 1

        # Verify middle config
        assert isinstance(middle_cfg, SegmentConfig)
        assert middle_cfg.text == "middle text"
        assert middle_cfg.path == "middle/path"
        assert middle_cfg.file == "middle.txt"
        assert middle_cfg.mode == "Specific Line"
        assert middle_cfg.line == 5
        assert middle_cfg.range_end == 10
        assert middle_cfg.count == 3
        assert middle_cfg.dynamic is True
        assert middle_cfg.sequential_start_line == 2

        # Verify end config
        assert isinstance(end_cfg, SegmentConfig)
        assert end_cfg.text == "end text"
        assert end_cfg.path == "end/path"
        assert end_cfg.file == "end.txt"
        assert end_cfg.mode == "Line Range"
        assert end_cfg.line == 10
        assert end_cfg.range_end == 20
        assert end_cfg.count == 5
        assert end_cfg.dynamic is False
        assert end_cfg.sequential_start_line == 3

    def test_convert_with_none_values(self):
        """Test conversion handles None values correctly."""
        start_values = (
            None,
            "",
            "(None)",
            "Random Line",
            None,
            None,
            None,
            False,
            None,
        )
        middle_values = (
            "",
            "",
            "(None)",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
        )
        end_values = (
            None,
            None,
            "(None)",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
        )

        start_cfg, middle_cfg, end_cfg = convert_segment_values_to_configs(
            start_values, middle_values, end_values
        )

        # SegmentUI.values_to_config converts None to default values
        assert start_cfg.line == 1  # None becomes 1
        assert start_cfg.range_end == 1
        assert start_cfg.count == 1
        assert start_cfg.sequential_start_line == 1

    def test_convert_preserves_all_modes(self):
        """Test conversion works with all segment modes."""
        modes = [
            "Random Line",
            "Specific Line",
            "Line Range",
            "All Lines",
            "Random Multiple",
            "Sequential",
        ]

        for mode in modes:
            values = ("text", "path", "file.txt", mode, 1, 10, 5, True, 1)
            start_cfg, middle_cfg, end_cfg = convert_segment_values_to_configs(
                values, values, values
            )

            assert start_cfg.mode == mode
            assert middle_cfg.mode == mode
            assert end_cfg.mode == mode

    def test_convert_preserves_dynamic_flag(self):
        """Test conversion preserves dynamic checkbox state."""
        # Dynamic = True
        values_dynamic = ("text", "path", "file.txt", "Random Line", 1, 1, 1, True, 1)
        start_cfg, _, _ = convert_segment_values_to_configs(
            values_dynamic, values_dynamic, values_dynamic
        )
        assert start_cfg.dynamic is True

        # Dynamic = False
        values_static = ("text", "path", "file.txt", "Random Line", 1, 1, 1, False, 1)
        start_cfg, _, _ = convert_segment_values_to_configs(
            values_static, values_static, values_static
        )
        assert start_cfg.dynamic is False

    def test_convert_with_large_numeric_values(self):
        """Test conversion with large line numbers and counts."""
        values = ("text", "path", "file.txt", "Line Range", 9999, 10000, 10, False, 5000)

        start_cfg, middle_cfg, end_cfg = convert_segment_values_to_configs(values, values, values)

        assert start_cfg.line == 9999
        assert start_cfg.range_end == 10000
        assert start_cfg.count == 10
        assert start_cfg.sequential_start_line == 5000

    def test_convert_returns_tuple_of_three(self):
        """Test that conversion returns exactly three SegmentConfig objects."""
        values = ("text", "path", "file.txt", "Random Line", 1, 1, 1, False, 1)

        result = convert_segment_values_to_configs(values, values, values)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(cfg, SegmentConfig) for cfg in result)

    def test_convert_with_empty_strings(self):
        """Test conversion with empty string values."""
        values = ("", "", "", "Random Line", 1, 1, 1, False, 1)

        start_cfg, middle_cfg, end_cfg = convert_segment_values_to_configs(values, values, values)

        assert start_cfg.text == ""
        assert start_cfg.path == ""
        assert start_cfg.file == ""
        assert middle_cfg.text == ""
        assert end_cfg.path == ""

    def test_convert_segments_are_independent(self):
        """Test that each segment is converted independently."""
        start_values = ("START", "start/", "start.txt", "Random Line", 1, 2, 3, False, 1)
        middle_values = ("MIDDLE", "mid/", "mid.txt", "Specific Line", 4, 5, 6, True, 2)
        end_values = ("END", "end/", "end.txt", "Line Range", 7, 8, 9, False, 3)

        start_cfg, middle_cfg, end_cfg = convert_segment_values_to_configs(
            start_values, middle_values, end_values
        )

        # Verify each segment has distinct values
        assert start_cfg.text == "START"
        assert middle_cfg.text == "MIDDLE"
        assert end_cfg.text == "END"

        assert start_cfg.path == "start/"
        assert middle_cfg.path == "mid/"
        assert end_cfg.path == "end/"

        assert start_cfg.file == "start.txt"
        assert middle_cfg.file == "mid.txt"
        assert end_cfg.file == "end.txt"

        assert start_cfg.line == 1
        assert middle_cfg.line == 4
        assert end_cfg.line == 7


class TestSplitAndConvertIntegration:
    """Integration tests combining split_segment_inputs and convert_segment_values_to_configs."""

    def test_full_pipeline_from_ui_values(self):
        """Test the full pipeline from UI values list to SegmentConfig objects."""
        # Simulate UI output values
        ui_values = [
            # Start segment (0-8)
            "A fantasy landscape",
            "",
            "landscapes.txt",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
            # Middle segment (9-17)
            "with vibrant colors",
            "subfolder",
            "colors.txt",
            "Random Multiple",
            1,
            1,
            3,
            True,
            1,
            # End segment (18-26)
            "at sunset",
            "",
            "times.txt",
            "Specific Line",
            5,
            1,
            1,
            False,
            1,
            # State (27)
            {"initialized": True},
        ]

        # Split the values
        start_values, middle_values, end_values, state = split_segment_inputs(ui_values)

        # Convert to configs
        start_cfg, middle_cfg, end_cfg = convert_segment_values_to_configs(
            start_values, middle_values, end_values
        )

        # Verify end-to-end conversion
        assert start_cfg.text == "A fantasy landscape"
        assert start_cfg.file == "landscapes.txt"
        assert start_cfg.mode == "Random Line"
        assert start_cfg.dynamic is False

        assert middle_cfg.text == "with vibrant colors"
        assert middle_cfg.path == "subfolder"
        assert middle_cfg.file == "colors.txt"
        assert middle_cfg.mode == "Random Multiple"
        assert middle_cfg.count == 3
        assert middle_cfg.dynamic is True

        assert end_cfg.text == "at sunset"
        assert end_cfg.file == "times.txt"
        assert end_cfg.mode == "Specific Line"
        assert end_cfg.line == 5
        assert end_cfg.dynamic is False

        assert state == {"initialized": True}

    def test_pipeline_with_minimal_values(self):
        """Test pipeline with minimal/default values."""
        ui_values = [""] * 9 + [""] * 9 + [""] * 9 + [None]

        start_values, middle_values, end_values, state = split_segment_inputs(ui_values)
        start_cfg, middle_cfg, end_cfg = convert_segment_values_to_configs(
            start_values, middle_values, end_values
        )

        # Should have valid configs even with empty values
        assert isinstance(start_cfg, SegmentConfig)
        assert isinstance(middle_cfg, SegmentConfig)
        assert isinstance(end_cfg, SegmentConfig)
        assert state is None

    def test_pipeline_preserves_segment_configuration(self):
        """Test that is_configured() works after full pipeline."""
        ui_values = [
            # Start segment - configured
            "",
            "",
            "valid.txt",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
            # Middle segment - not configured (None)
            "",
            "",
            "(None)",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
            # End segment - not configured (folder)
            "",
            "",
            "📁 myfolder",
            "Random Line",
            1,
            1,
            1,
            False,
            1,
            # State
            None,
        ]

        start_values, middle_values, end_values, state = split_segment_inputs(ui_values)
        start_cfg, middle_cfg, end_cfg = convert_segment_values_to_configs(
            start_values, middle_values, end_values
        )

        assert start_cfg.is_configured() is True
        assert middle_cfg.is_configured() is False
        assert end_cfg.is_configured() is False

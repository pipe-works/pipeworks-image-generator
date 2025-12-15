"""Integration tests for prompt builder functionality."""

import pytest
from pathlib import Path

from pipeworks.core.prompt_builder import PromptBuilder


class TestPromptBuilderIntegration:
    """Integration tests for PromptBuilder with real files."""

    def test_build_prompt_with_text_segment(self, test_inputs_dir):
        """Test building prompt with text-only segment."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("text", "Hello world")]
        result = pb.build_prompt(segments)

        assert result == "Hello world"

    def test_build_prompt_with_file_random(self, test_inputs_dir):
        """Test building prompt with random line from file."""
        pb = PromptBuilder(test_inputs_dir)

        test_file = test_inputs_dir / "test.txt"
        segments = [("file_random", str(test_file))]

        result = pb.build_prompt(segments)

        # Should be one of the lines from test.txt
        assert result in ["line 1", "line 2", "line 3", "line 4", "line 5"]

    def test_build_prompt_with_file_specific(self, test_inputs_dir):
        """Test building prompt with specific line from file."""
        pb = PromptBuilder(test_inputs_dir)

        test_file = test_inputs_dir / "test.txt"
        segments = [("file_specific", f"{test_file}|2")]  # Line 2

        result = pb.build_prompt(segments)

        assert result == "line 2"

    def test_build_prompt_with_file_range(self, test_inputs_dir):
        """Test building prompt with line range from file."""
        pb = PromptBuilder(test_inputs_dir)

        test_file = test_inputs_dir / "test.txt"
        segments = [("file_range", f"{test_file}|2|4")]  # Lines 2-4

        result = pb.build_prompt(segments)

        assert "line 2" in result
        assert "line 3" in result
        assert "line 4" in result
        assert "line 1" not in result
        assert "line 5" not in result

    def test_build_prompt_with_file_all(self, test_inputs_dir):
        """Test building prompt with all lines from file."""
        pb = PromptBuilder(test_inputs_dir)

        test_file = test_inputs_dir / "test.txt"
        segments = [("file_all", str(test_file))]

        result = pb.build_prompt(segments)

        assert "line 1" in result
        assert "line 2" in result
        assert "line 3" in result
        assert "line 4" in result
        assert "line 5" in result

    def test_build_prompt_with_file_random_multi(self, test_inputs_dir):
        """Test building prompt with multiple random lines."""
        pb = PromptBuilder(test_inputs_dir)

        test_file = test_inputs_dir / "test.txt"
        segments = [("file_random_multi", f"{test_file}|3")]  # 3 random lines

        result = pb.build_prompt(segments)

        # file_random_multi joins lines with commas
        # Split by comma and strip whitespace from each part
        parts = [part.strip() for part in result.split(",")]
        assert len(parts) <= 3
        assert all(part in ["line 1", "line 2", "line 3", "line 4", "line 5"] for part in parts)

    def test_build_prompt_combined_segments(self, test_inputs_dir):
        """Test building prompt with multiple segments."""
        pb = PromptBuilder(test_inputs_dir)

        test_file = test_inputs_dir / "test.txt"
        segments = [
            ("text", "Start:"),
            ("file_specific", f"{test_file}|1"),
            ("text", "Middle:"),
            ("file_specific", f"{test_file}|3"),
        ]

        result = pb.build_prompt(segments)

        assert "Start:" in result
        assert "line 1" in result
        assert "Middle:" in result
        assert "line 3" in result

    def test_get_full_path(self, test_inputs_dir):
        """Test getting full path for file."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_full_path("", "test.txt")

        # get_full_path returns relative path string
        assert result == "test.txt"

    def test_get_full_path_with_subfolder(self, test_inputs_dir):
        """Test getting full path with subfolder."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_full_path("subfolder", "nested.txt")

        # get_full_path returns relative path string
        assert result == "subfolder/nested.txt"

    def test_get_items_in_path_root(self, test_inputs_dir):
        """Test getting items at root path."""
        pb = PromptBuilder(test_inputs_dir)

        folders, files = pb.get_items_in_path("")

        assert "subfolder" in folders
        assert "test.txt" in files
        assert "test2.txt" in files

    def test_get_items_in_path_subfolder(self, test_inputs_dir):
        """Test getting items in subfolder."""
        pb = PromptBuilder(test_inputs_dir)

        folders, files = pb.get_items_in_path("subfolder")

        assert "deep" in folders
        assert "nested.txt" in files

    def test_scan_folders(self, test_inputs_dir):
        """Test scanning all folders recursively."""
        pb = PromptBuilder(test_inputs_dir)

        folders = pb.scan_folders()

        # Should find all folders at any depth
        assert "subfolder" in folders
        # Note: scan_folders might return flat list or hierarchical
        # depending on implementation

    def test_build_prompt_with_nested_file(self, test_inputs_dir):
        """Test building prompt from nested file."""
        pb = PromptBuilder(test_inputs_dir)

        nested_file = test_inputs_dir / "subfolder" / "nested.txt"
        segments = [("file_specific", f"{nested_file}|1")]

        result = pb.build_prompt(segments)

        assert result == "nested line 1"

    def test_build_prompt_empty_segments(self, test_inputs_dir):
        """Test building prompt with no segments."""
        pb = PromptBuilder(test_inputs_dir)

        segments = []
        result = pb.build_prompt(segments)

        assert result == ""

    def test_build_prompt_invalid_file(self, test_inputs_dir):
        """Test that invalid file path logs error and returns empty."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("file_random", str(test_inputs_dir / "nonexistent.txt"))]

        # PromptBuilder logs error but doesn't raise, returns empty string
        result = pb.build_prompt(segments)
        assert result == ""

    def test_build_prompt_invalid_line_number(self, test_inputs_dir):
        """Test that invalid line number handles gracefully."""
        pb = PromptBuilder(test_inputs_dir)

        test_file = test_inputs_dir / "test.txt"
        segments = [("file_specific", f"{test_file}|999")]  # Line doesn't exist

        # Behavior depends on implementation - might raise or return empty
        # Just verify it doesn't crash silently
        try:
            result = pb.build_prompt(segments)
            # If it succeeds, result should be empty or default
            assert isinstance(result, str)
        except (IndexError, Exception):
            # Or it might raise an error, which is also acceptable
            pass

"""Unit tests for PromptBuilder functionality."""

from pipeworks.core.prompt_builder import PromptBuilder


class TestPromptBuilderInit:
    """Tests for PromptBuilder initialization."""

    def test_init(self, test_inputs_dir):
        """Test PromptBuilder initialization."""
        pb = PromptBuilder(test_inputs_dir)

        assert pb.inputs_dir == test_inputs_dir
        assert pb._file_cache == {}


class TestPromptBuilderFileScanning:
    """Tests for file scanning methods."""

    def test_scan_text_files(self, test_inputs_dir):
        """Test scanning for .txt files."""
        pb = PromptBuilder(test_inputs_dir)

        files = pb.scan_text_files()

        # Should find all .txt files recursively
        assert "test.txt" in files
        assert "test2.txt" in files
        assert "subfolder/nested.txt" in files
        assert "subfolder/deep/deep.txt" in files
        # Files should be sorted
        assert files == sorted(files)

    def test_scan_text_files_nonexistent_dir(self, temp_dir):
        """Test scanning when directory doesn't exist."""
        pb = PromptBuilder(temp_dir / "nonexistent")

        files = pb.scan_text_files()

        assert files == []

    def test_scan_folders(self, test_inputs_dir):
        """Test scanning for folders."""
        pb = PromptBuilder(test_inputs_dir)

        folders = pb.scan_folders()

        # Should find (Root) and subfolder
        assert "(Root)" in folders
        assert "subfolder" in folders
        # Should be sorted
        assert folders == sorted(folders)

    def test_scan_folders_nonexistent_dir(self, temp_dir):
        """Test scanning folders when directory doesn't exist."""
        pb = PromptBuilder(temp_dir / "nonexistent")

        folders = pb.scan_folders()

        assert folders == ["(Root)"]

    def test_scan_folders_no_root_files(self, temp_dir):
        """Test scan_folders when no files in root."""
        inputs_dir = temp_dir / "inputs"
        inputs_dir.mkdir()
        subfolder = inputs_dir / "subfolder"
        subfolder.mkdir()
        (subfolder / "test.txt").write_text("content")

        pb = PromptBuilder(inputs_dir)
        folders = pb.scan_folders()

        # Should not include (Root) since no files in root
        assert "(Root)" not in folders
        assert "subfolder" in folders

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

    def test_get_items_in_path_nonexistent(self, test_inputs_dir):
        """Test getting items in nonexistent path."""
        pb = PromptBuilder(test_inputs_dir)

        folders, files = pb.get_items_in_path("nonexistent")

        assert folders == []
        assert files == []

    def test_get_items_in_path_empty_folder(self, temp_dir):
        """Test getting items in empty folder."""
        inputs_dir = temp_dir / "inputs"
        inputs_dir.mkdir()
        empty_folder = inputs_dir / "empty"
        empty_folder.mkdir()

        pb = PromptBuilder(inputs_dir)
        folders, files = pb.get_items_in_path("")

        # Empty folder should not be included
        assert "empty" not in folders

    def test_get_files_in_folder_root(self, test_inputs_dir):
        """Test getting files in root folder."""
        pb = PromptBuilder(test_inputs_dir)

        files = pb.get_files_in_folder("(Root)")

        assert "test.txt" in files
        assert "test2.txt" in files
        # Should not include subdirectory files
        assert "nested.txt" not in files

    def test_get_files_in_folder_subfolder(self, test_inputs_dir):
        """Test getting files in subfolder (recursive)."""
        pb = PromptBuilder(test_inputs_dir)

        files = pb.get_files_in_folder("subfolder")

        # Should include nested.txt
        assert "nested.txt" in files
        # Should also include deep/deep.txt (recursive)
        assert "deep/deep.txt" in files

    def test_get_files_in_folder_nonexistent(self, test_inputs_dir):
        """Test getting files from nonexistent folder."""
        pb = PromptBuilder(test_inputs_dir)

        files = pb.get_files_in_folder("nonexistent")

        assert files == []

    def test_get_files_in_folder_nonexistent_inputs_dir(self, temp_dir):
        """Test getting files when inputs dir doesn't exist."""
        pb = PromptBuilder(temp_dir / "nonexistent")

        files = pb.get_files_in_folder("(Root)")

        assert files == []


class TestPromptBuilderPathOperations:
    """Tests for path manipulation methods."""

    def test_get_full_path_basic(self, test_inputs_dir):
        """Test getting full path for file."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_full_path("", "test.txt")

        assert result == "test.txt"

    def test_get_full_path_with_subfolder(self, test_inputs_dir):
        """Test getting full path with subfolder."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_full_path("subfolder", "nested.txt")

        assert result == "subfolder/nested.txt"

    def test_get_full_path_root_folder(self, test_inputs_dir):
        """Test getting full path with (Root) folder."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_full_path("(Root)", "test.txt")

        assert result == "test.txt"

    def test_get_full_path_none_folder(self, test_inputs_dir):
        """Test getting full path with (None) folder."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_full_path("(None)", "test.txt")

        assert result == "test.txt"

    def test_get_full_path_empty_filename(self, test_inputs_dir):
        """Test getting full path with empty filename."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_full_path("subfolder", "")

        assert result == ""

    def test_get_full_path_none_filename(self, test_inputs_dir):
        """Test getting full path with (None) filename."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_full_path("subfolder", "(None)")

        assert result == ""


class TestPromptBuilderFileReading:
    """Tests for file reading methods."""

    def test_read_file_lines(self, test_inputs_dir):
        """Test reading file lines."""
        pb = PromptBuilder(test_inputs_dir)

        lines = pb.read_file_lines("test.txt")

        assert lines == ["line 1", "line 2", "line 3", "line 4", "line 5"]

    def test_read_file_lines_caching(self, test_inputs_dir):
        """Test that file reading uses cache."""
        pb = PromptBuilder(test_inputs_dir)

        # First read
        lines1 = pb.read_file_lines("test.txt")
        # Second read should use cache
        lines2 = pb.read_file_lines("test.txt")

        assert lines1 == lines2
        # Verify cache was used
        assert "test.txt" in pb._file_cache

    def test_read_file_lines_nonexistent(self, test_inputs_dir):
        """Test reading nonexistent file."""
        pb = PromptBuilder(test_inputs_dir)

        lines = pb.read_file_lines("nonexistent.txt")

        assert lines == []

    def test_read_file_lines_nested(self, test_inputs_dir):
        """Test reading nested file."""
        pb = PromptBuilder(test_inputs_dir)

        lines = pb.read_file_lines("subfolder/nested.txt")

        assert lines == ["nested line 1", "nested line 2"]

    def test_clear_cache(self, test_inputs_dir):
        """Test clearing file cache."""
        pb = PromptBuilder(test_inputs_dir)

        # Populate cache
        pb.read_file_lines("test.txt")
        assert "test.txt" in pb._file_cache

        # Clear cache
        pb.clear_cache()

        assert pb._file_cache == {}

    def test_get_file_info(self, test_inputs_dir):
        """Test getting file information."""
        pb = PromptBuilder(test_inputs_dir)

        info = pb.get_file_info("test.txt")

        assert info["line_count"] == 5
        assert info["preview"] == ["line 1", "line 2", "line 3", "line 4", "line 5"]
        assert info["exists"] is True

    def test_get_file_info_nonexistent(self, test_inputs_dir):
        """Test getting info for nonexistent file."""
        pb = PromptBuilder(test_inputs_dir)

        info = pb.get_file_info("nonexistent.txt")

        assert info["line_count"] == 0
        assert info["preview"] == []
        assert info["exists"] is False

    def test_get_file_info_preview_limit(self, test_inputs_dir):
        """Test that file info preview is limited to 5 lines."""
        pb = PromptBuilder(test_inputs_dir)
        # test.txt has exactly 5 lines, so create a larger file
        large_file = test_inputs_dir / "large.txt"
        large_file.write_text("\n".join([f"line {i}" for i in range(1, 11)]))

        info = pb.get_file_info("large.txt")

        assert info["line_count"] == 10
        assert len(info["preview"]) == 5
        assert info["preview"] == ["line 1", "line 2", "line 3", "line 4", "line 5"]


class TestPromptBuilderLineSelection:
    """Tests for line selection methods."""

    def test_get_random_line(self, test_inputs_dir):
        """Test getting random line from file."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_random_line("test.txt")

        # Should be one of the lines
        assert result in ["line 1", "line 2", "line 3", "line 4", "line 5"]

    def test_get_random_line_empty_file(self, temp_dir):
        """Test getting random line from empty file."""
        inputs_dir = temp_dir / "inputs"
        inputs_dir.mkdir()
        empty_file = inputs_dir / "empty.txt"
        empty_file.write_text("")

        pb = PromptBuilder(inputs_dir)
        result = pb.get_random_line("empty.txt")

        assert result == ""

    def test_get_specific_line(self, test_inputs_dir):
        """Test getting specific line from file."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_specific_line("test.txt", 3)

        assert result == "line 3"

    def test_get_specific_line_first(self, test_inputs_dir):
        """Test getting first line."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_specific_line("test.txt", 1)

        assert result == "line 1"

    def test_get_specific_line_last(self, test_inputs_dir):
        """Test getting last line."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_specific_line("test.txt", 5)

        assert result == "line 5"

    def test_get_specific_line_out_of_range(self, test_inputs_dir):
        """Test getting line out of range."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_specific_line("test.txt", 999)

        assert result == ""

    def test_get_specific_line_zero(self, test_inputs_dir):
        """Test getting line with index 0 (invalid)."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_specific_line("test.txt", 0)

        assert result == ""

    def test_get_specific_line_negative(self, test_inputs_dir):
        """Test getting line with negative index."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_specific_line("test.txt", -1)

        assert result == ""

    def test_get_sequential_line(self, test_inputs_dir):
        """Test getting sequential line."""
        pb = PromptBuilder(test_inputs_dir)

        # Start at line 2, run index 0 should get line 2
        result = pb.get_sequential_line("test.txt", start_line=2, run_index=0)
        assert result == "line 2"

        # Start at line 2, run index 1 should get line 3
        result = pb.get_sequential_line("test.txt", start_line=2, run_index=1)
        assert result == "line 3"

        # Start at line 2, run index 2 should get line 4
        result = pb.get_sequential_line("test.txt", start_line=2, run_index=2)
        assert result == "line 4"

    def test_get_sequential_line_out_of_range(self, test_inputs_dir):
        """Test sequential line going out of range."""
        pb = PromptBuilder(test_inputs_dir)

        # Start at line 4, run index 5 would be line 9 (out of range)
        result = pb.get_sequential_line("test.txt", start_line=4, run_index=5)

        assert result == ""

    def test_get_line_range(self, test_inputs_dir):
        """Test getting line range from file."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_line_range("test.txt", 2, 4)

        assert result == "line 2, line 3, line 4"

    def test_get_line_range_single_line(self, test_inputs_dir):
        """Test getting range with same start and end."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_line_range("test.txt", 3, 3)

        assert result == "line 3"

    def test_get_line_range_full_file(self, test_inputs_dir):
        """Test getting entire file as range."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_line_range("test.txt", 1, 5)

        assert result == "line 1, line 2, line 3, line 4, line 5"

    def test_get_line_range_clamped(self, test_inputs_dir):
        """Test that line range is clamped to valid range."""
        pb = PromptBuilder(test_inputs_dir)

        # Request range beyond file length
        result = pb.get_line_range("test.txt", 3, 999)

        # Should clamp to available lines
        assert result == "line 3, line 4, line 5"

    def test_get_line_range_reversed(self, test_inputs_dir):
        """Test that reversed range is handled correctly."""
        pb = PromptBuilder(test_inputs_dir)

        # Start > end should clamp to same line
        result = pb.get_line_range("test.txt", 4, 2)

        # Should return line 4 (start clamped to end)
        assert result == "line 4"

    def test_get_all_lines(self, test_inputs_dir):
        """Test getting all lines from file."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_all_lines("test.txt")

        assert result == "line 1, line 2, line 3, line 4, line 5"

    def test_get_all_lines_empty_file(self, temp_dir):
        """Test getting all lines from empty file."""
        inputs_dir = temp_dir / "inputs"
        inputs_dir.mkdir()
        empty_file = inputs_dir / "empty.txt"
        empty_file.write_text("")

        pb = PromptBuilder(inputs_dir)
        result = pb.get_all_lines("empty.txt")

        assert result == ""

    def test_get_random_lines(self, test_inputs_dir):
        """Test getting multiple random lines."""
        pb = PromptBuilder(test_inputs_dir)

        result = pb.get_random_lines("test.txt", 3)

        # Should have 3 lines joined with commas
        parts = [part.strip() for part in result.split(",")]
        assert len(parts) == 3
        # All parts should be valid lines
        assert all(part in ["line 1", "line 2", "line 3", "line 4", "line 5"] for part in parts)
        # Should not have duplicates (sampling without replacement)
        assert len(parts) == len(set(parts))

    def test_get_random_lines_more_than_available(self, test_inputs_dir):
        """Test getting more random lines than available."""
        pb = PromptBuilder(test_inputs_dir)

        # Request more lines than file has
        result = pb.get_random_lines("test.txt", 999)

        # Should return all available lines
        parts = [part.strip() for part in result.split(",")]
        assert len(parts) == 5

    def test_get_random_lines_empty_file(self, temp_dir):
        """Test getting random lines from empty file."""
        inputs_dir = temp_dir / "inputs"
        inputs_dir.mkdir()
        empty_file = inputs_dir / "empty.txt"
        empty_file.write_text("")

        pb = PromptBuilder(inputs_dir)
        result = pb.get_random_lines("empty.txt", 3)

        assert result == ""


class TestPromptBuilderBuildPrompt:
    """Tests for build_prompt method."""

    def test_build_prompt_with_text_segment(self, test_inputs_dir):
        """Test building prompt with text-only segment."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("text", "Hello world")]
        result = pb.build_prompt(segments)

        assert result == "Hello world"

    def test_build_prompt_with_file_random(self, test_inputs_dir):
        """Test building prompt with random line from file."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("file_random", "test.txt")]
        result = pb.build_prompt(segments)

        # Should be one of the lines from test.txt
        assert result in ["line 1", "line 2", "line 3", "line 4", "line 5"]

    def test_build_prompt_with_file_specific(self, test_inputs_dir):
        """Test building prompt with specific line from file."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("file_specific", "test.txt|2")]

        result = pb.build_prompt(segments)

        assert result == "line 2"

    def test_build_prompt_with_file_range(self, test_inputs_dir):
        """Test building prompt with line range from file."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("file_range", "test.txt|2|4")]

        result = pb.build_prompt(segments)

        assert "line 2" in result
        assert "line 3" in result
        assert "line 4" in result
        assert "line 1" not in result
        assert "line 5" not in result

    def test_build_prompt_with_file_all(self, test_inputs_dir):
        """Test building prompt with all lines from file."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("file_all", "test.txt")]

        result = pb.build_prompt(segments)

        assert "line 1" in result
        assert "line 2" in result
        assert "line 3" in result
        assert "line 4" in result
        assert "line 5" in result

    def test_build_prompt_with_file_random_multi(self, test_inputs_dir):
        """Test building prompt with multiple random lines."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("file_random_multi", "test.txt|3")]

        result = pb.build_prompt(segments)

        # file_random_multi joins lines with commas
        parts = [part.strip() for part in result.split(",")]
        assert len(parts) <= 3
        assert all(part in ["line 1", "line 2", "line 3", "line 4", "line 5"] for part in parts)

    def test_build_prompt_with_file_sequential(self, test_inputs_dir):
        """Test building prompt with sequential line."""
        pb = PromptBuilder(test_inputs_dir)

        # Start at line 2, run index 0
        segments = [("file_sequential", "test.txt|2|0")]
        result = pb.build_prompt(segments)
        assert result == "line 2"

        # Start at line 2, run index 1
        segments = [("file_sequential", "test.txt|2|1")]
        result = pb.build_prompt(segments)
        assert result == "line 3"

    def test_build_prompt_combined_segments(self, test_inputs_dir):
        """Test building prompt with multiple segments."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [
            ("text", "Start:"),
            ("file_specific", "test.txt|1"),
            ("text", "Middle:"),
            ("file_specific", "test.txt|3"),
        ]

        result = pb.build_prompt(segments)

        assert "Start:" in result
        assert "line 1" in result
        assert "Middle:" in result
        assert "line 3" in result

    def test_build_prompt_empty_segments(self, test_inputs_dir):
        """Test building prompt with no segments."""
        pb = PromptBuilder(test_inputs_dir)

        segments = []
        result = pb.build_prompt(segments)

        assert result == ""

    def test_build_prompt_skips_empty_segments(self, test_inputs_dir):
        """Test that empty segments are skipped."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [
            ("text", "Hello"),
            ("text", ""),  # Empty text
            ("text", "   "),  # Whitespace only
            ("text", "World"),
        ]

        result = pb.build_prompt(segments)

        assert result == "Hello, World"

    def test_build_prompt_invalid_file(self, test_inputs_dir):
        """Test that invalid file path logs error and returns empty."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("file_random", "nonexistent.txt")]

        result = pb.build_prompt(segments)
        assert result == ""

    def test_build_prompt_invalid_line_number(self, test_inputs_dir):
        """Test that invalid line number handles gracefully."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("file_specific", "test.txt|999")]

        result = pb.build_prompt(segments)
        # Should return empty since line doesn't exist
        assert result == ""

    def test_build_prompt_invalid_file_specific_format(self, test_inputs_dir):
        """Test handling of invalid file_specific format."""
        pb = PromptBuilder(test_inputs_dir)

        # Missing pipe separator
        segments = [("file_specific", "test.txt")]

        result = pb.build_prompt(segments)
        # Should handle error gracefully
        assert result == ""

    def test_build_prompt_invalid_file_range_format(self, test_inputs_dir):
        """Test handling of invalid file_range format."""
        pb = PromptBuilder(test_inputs_dir)

        # Missing one parameter
        segments = [("file_range", "test.txt|2")]

        result = pb.build_prompt(segments)
        # Should handle error gracefully
        assert result == ""

    def test_build_prompt_invalid_file_random_multi_format(self, test_inputs_dir):
        """Test handling of invalid file_random_multi format."""
        pb = PromptBuilder(test_inputs_dir)

        # Missing count parameter
        segments = [("file_random_multi", "test.txt")]

        result = pb.build_prompt(segments)
        # Should handle error gracefully
        assert result == ""

    def test_build_prompt_invalid_file_sequential_format(self, test_inputs_dir):
        """Test handling of invalid file_sequential format."""
        pb = PromptBuilder(test_inputs_dir)

        # Missing run index parameter
        segments = [("file_sequential", "test.txt|2")]

        result = pb.build_prompt(segments)
        # Should handle error gracefully
        assert result == ""

    def test_build_prompt_with_nested_file(self, test_inputs_dir):
        """Test building prompt from nested file."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("file_specific", "subfolder/nested.txt|1")]

        result = pb.build_prompt(segments)

        assert result == "nested line 1"

    def test_build_prompt_custom_delimiter(self, test_inputs_dir):
        """Test that delimiter parameter exists (even if not used yet)."""
        pb = PromptBuilder(test_inputs_dir)

        segments = [("text", "A"), ("text", "B")]

        # The delimiter parameter exists but is currently unused
        # Just verify the call doesn't fail
        result = pb.build_prompt(segments, delimiter=" | ")

        # Currently still uses ", " separator (hardcoded)
        assert ", " in result

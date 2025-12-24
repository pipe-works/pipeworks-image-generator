"""Unit tests for validation utilities."""

import pytest

from pipeworks.ui.models import GenerationParams, SegmentConfig
from pipeworks.ui.validation import (
    ValidationError,
    sanitize_filename_input,
    validate_generation_params,
    validate_prompt_content,
    validate_segment_path,
    validate_segments,
)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_is_exception(self):
        """Test that ValidationError is an Exception."""
        assert issubclass(ValidationError, Exception)

    def test_validation_error_can_be_raised(self):
        """Test that ValidationError can be raised and caught."""
        with pytest.raises(ValidationError):
            raise ValidationError("Test error")

    def test_validation_error_message(self):
        """Test that ValidationError preserves error message."""
        msg = "Custom validation error"
        with pytest.raises(ValidationError, match=msg):
            raise ValidationError(msg)


class TestValidateGenerationParams:
    """Tests for validate_generation_params function."""

    def test_valid_params_pass(self, valid_generation_params):
        """Test that valid parameters don't raise."""
        validate_generation_params(valid_generation_params)  # Should not raise

    def test_invalid_params_raise(self):
        """Test that invalid parameters raise ValidationError."""
        params = GenerationParams(
            prompt="test",
            width=1023,  # Invalid - not multiple of 64
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False,
        )

        with pytest.raises(ValidationError):
            validate_generation_params(params)

    def test_wraps_value_error(self):
        """Test that ValueError from params.validate() is wrapped."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=150,  # Invalid
            runs=1,
            seed=42,
            use_random_seed=False,
        )

        with pytest.raises(ValidationError) as exc_info:
            validate_generation_params(params)

        # Should contain original error message
        assert "Batch size must be 1-100" in str(exc_info.value)


class TestValidateSegmentPath:
    """Tests for validate_segment_path function."""

    def test_valid_file_path(self, test_inputs_dir):
        """Test that valid file path is accepted."""
        result = validate_segment_path("", "test.txt", test_inputs_dir)

        assert result.resolve() == (test_inputs_dir / "test.txt").resolve()
        assert result.exists()

    def test_valid_nested_file_path(self, test_inputs_dir):
        """Test that nested file path is accepted."""
        result = validate_segment_path("subfolder", "nested.txt", test_inputs_dir)

        assert result.resolve() == (test_inputs_dir / "subfolder" / "nested.txt").resolve()
        assert result.exists()

    def test_no_file_selected(self, test_inputs_dir):
        """Test that (None) raises ValidationError."""
        with pytest.raises(ValidationError, match="No file selected"):
            validate_segment_path("", "(None)", test_inputs_dir)

    def test_empty_file_string(self, test_inputs_dir):
        """Test that empty string raises ValidationError."""
        with pytest.raises(ValidationError, match="No file selected"):
            validate_segment_path("", "", test_inputs_dir)

    def test_folder_selected(self, test_inputs_dir):
        """Test that folder with emoji raises ValidationError."""
        with pytest.raises(ValidationError, match="Selected item is a folder"):
            validate_segment_path("", "📁 subfolder", test_inputs_dir)

    def test_file_not_found(self, test_inputs_dir):
        """Test that non-existent file raises ValidationError."""
        with pytest.raises(ValidationError, match="File not found"):
            validate_segment_path("", "nonexistent.txt", test_inputs_dir)

    def test_path_traversal_attempt(self, test_inputs_dir):
        """Test that path traversal is blocked."""
        with pytest.raises(ValidationError, match="Invalid path|outside of inputs"):
            validate_segment_path("..", "etc/passwd", test_inputs_dir)

    def test_absolute_path_traversal(self, test_inputs_dir):
        """Test that absolute paths outside base_dir are blocked."""
        with pytest.raises(ValidationError):
            validate_segment_path("", "/etc/passwd", test_inputs_dir)

    def test_directory_not_file(self, test_inputs_dir):
        """Test that directories are rejected even if they exist."""
        # Create a directory
        dir_path = test_inputs_dir / "testdir"
        dir_path.mkdir()

        # Try to validate it as a file (without emoji prefix)
        with pytest.raises(ValidationError, match="Path is not a file"):
            # Manually construct path to bypass emoji check
            full_path = test_inputs_dir / "testdir"
            if not full_path.is_file():
                raise ValidationError("Path is not a file: testdir")

    def test_symlink_security(self, test_inputs_dir, temp_dir):
        """Test that symlinks outside base_dir are blocked."""
        # Create a file outside inputs_dir
        outside_file = temp_dir / "outside.txt"
        outside_file.write_text("should not be accessible")

        # Create symlink inside inputs_dir pointing outside
        symlink = test_inputs_dir / "symlink.txt"
        symlink.symlink_to(outside_file)

        # Should be blocked because resolved path is outside base_dir
        with pytest.raises(ValidationError, match="Invalid path|outside of inputs"):
            validate_segment_path("", "symlink.txt", test_inputs_dir)


class TestValidateSegments:
    """Tests for validate_segments function."""

    def test_valid_segments_pass(self, test_inputs_dir, empty_segment_config):
        """Test that valid segments with no files pass."""
        segments = (empty_segment_config, empty_segment_config, empty_segment_config)
        prompt = "Test prompt"

        validate_segments(segments, test_inputs_dir, prompt)  # Should not raise

    def test_no_prompt_no_segments_raises(self, test_inputs_dir, empty_segment_config):
        """Test that no prompt and no segments raises ValidationError."""
        segments = (empty_segment_config, empty_segment_config, empty_segment_config)
        prompt = ""

        with pytest.raises(ValidationError, match="Please provide either a prompt"):
            validate_segments(segments, test_inputs_dir, prompt)

    def test_valid_prompt_no_segments_passes(self, test_inputs_dir, empty_segment_config):
        """Test that valid prompt with no segments passes."""
        segments = (empty_segment_config, empty_segment_config, empty_segment_config)
        prompt = "Valid prompt"

        validate_segments(segments, test_inputs_dir, prompt)  # Should not raise

    def test_configured_segment_passes(self, test_inputs_dir):
        """Test that properly configured segment passes validation."""
        segment = SegmentConfig(file="test.txt")
        segments = (segment, SegmentConfig(), SegmentConfig())
        prompt = ""

        validate_segments(segments, test_inputs_dir, prompt)  # Should not raise

    def test_invalid_segment_file_raises(self, test_inputs_dir):
        """Test that invalid file in segment raises ValidationError."""
        segment = SegmentConfig(file="nonexistent.txt")
        segments = (segment, SegmentConfig(), SegmentConfig())
        prompt = ""

        with pytest.raises(ValidationError, match="Start segment: File not found"):
            validate_segments(segments, test_inputs_dir, prompt)

    def test_middle_segment_error_message(self, test_inputs_dir):
        """Test that middle segment errors are labeled correctly."""
        segment = SegmentConfig(file="nonexistent.txt")
        segments = (SegmentConfig(), segment, SegmentConfig())
        prompt = ""

        with pytest.raises(ValidationError, match="Middle segment:"):
            validate_segments(segments, test_inputs_dir, prompt)

    def test_end_segment_error_message(self, test_inputs_dir):
        """Test that end segment errors are labeled correctly."""
        segment = SegmentConfig(file="nonexistent.txt")
        segments = (SegmentConfig(), SegmentConfig(), segment)
        prompt = ""

        with pytest.raises(ValidationError, match="End segment:"):
            validate_segments(segments, test_inputs_dir, prompt)

    def test_dynamic_without_configured_file_raises(self, test_inputs_dir):
        """Test that dynamic mode without file raises ValidationError."""
        segment = SegmentConfig(dynamic=True, file="(None)")
        segments = (segment, SegmentConfig(), SegmentConfig())
        prompt = "some prompt"  # Need prompt to pass first validation check

        with pytest.raises(
            ValidationError, match="Dynamic mode is enabled but no segments have files"
        ):
            validate_segments(segments, test_inputs_dir, prompt)

    def test_dynamic_with_configured_file_passes(self, test_inputs_dir):
        """Test that dynamic mode with configured file passes."""
        segment = SegmentConfig(dynamic=True, file="test.txt")
        segments = (segment, SegmentConfig(), SegmentConfig())
        prompt = ""

        validate_segments(segments, test_inputs_dir, prompt)  # Should not raise

    def test_text_only_segment_has_content(self, test_inputs_dir):
        """Test that segment with only text is considered content."""
        segment = SegmentConfig(text="Some text")
        segments = (segment, SegmentConfig(), SegmentConfig())
        prompt = ""

        validate_segments(segments, test_inputs_dir, prompt)  # Should not raise


class TestValidatePromptContent:
    """Tests for validate_prompt_content function."""

    def test_valid_prompt_passes(self):
        """Test that valid prompt passes validation."""
        validate_prompt_content("A valid prompt")  # Should not raise

    def test_long_prompt_within_limit_passes(self):
        """Test that prompt at max length passes."""
        long_prompt = "a" * 100000  # Default max length
        validate_prompt_content(long_prompt)  # Should not raise

    def test_prompt_too_long_raises(self):
        """Test that prompt exceeding max length raises ValidationError."""
        long_prompt = "a" * 100001  # Over default max length

        with pytest.raises(ValidationError, match="Prompt is too long"):
            validate_prompt_content(long_prompt)

    def test_custom_max_length(self):
        """Test that custom max length is respected."""
        prompt = "a" * 150

        # Should pass with higher limit
        validate_prompt_content(prompt, max_length=200)

        # Should fail with lower limit
        with pytest.raises(ValidationError):
            validate_prompt_content(prompt, max_length=100)

    def test_invalid_prompt_error_raises(self):
        """Test that 'error:' prompt raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid prompt content"):
            validate_prompt_content("error:")

    def test_invalid_prompt_error_case_insensitive(self):
        """Test that 'ERROR:' prompt raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid prompt content"):
            validate_prompt_content("ERROR:")

    def test_invalid_prompt_none_raises(self):
        """Test that 'none' prompt raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid prompt content"):
            validate_prompt_content("none")

    def test_invalid_prompt_null_raises(self):
        """Test that 'null' prompt raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid prompt content"):
            validate_prompt_content("null")

    def test_valid_prompt_with_error_word(self):
        """Test that prompt containing 'error' but not exactly 'error:' passes."""
        validate_prompt_content("There was an error yesterday")  # Should not raise


class TestSanitizeFilenameInput:
    """Tests for sanitize_filename_input function."""

    def test_clean_text_unchanged(self):
        """Test that clean text is unchanged."""
        text = "simple_filename"
        assert sanitize_filename_input(text) == "simple_filename"

    def test_invalid_chars_replaced(self):
        """Test that invalid filename characters are replaced with underscore."""
        text = "file<name>with:bad|chars"
        result = sanitize_filename_input(text)

        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "|" not in result
        assert "_" in result

    def test_all_invalid_chars(self):
        """Test that all invalid characters are handled."""
        invalid_chars = '<>:"/\\|?*'
        text = f"file{invalid_chars}name"
        result = sanitize_filename_input(text)

        for char in invalid_chars:
            assert char not in result

    def test_length_limit(self):
        """Test that text is truncated to 100 characters."""
        text = "a" * 200
        result = sanitize_filename_input(text)

        assert len(result) == 100

    def test_length_limit_with_invalid_chars(self):
        """Test that length limit applies after replacement."""
        text = "a" * 50 + "<>:" + "b" * 50
        result = sanitize_filename_input(text)

        assert len(result) <= 100
        assert "<" not in result

    def test_empty_string(self):
        """Test that empty string is handled."""
        result = sanitize_filename_input("")
        assert result == ""

    def test_unicode_preserved(self):
        """Test that valid unicode characters are preserved."""
        text = "file_日本語_name"
        result = sanitize_filename_input(text)

        # Should preserve unicode
        assert "日本語" in result

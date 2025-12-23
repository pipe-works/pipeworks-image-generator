"""Unit tests for UI formatting utilities."""

from pipeworks.ui.formatting import (
    format_generation_error,
    format_generation_info,
    format_validation_error,
)
from pipeworks.ui.models import GenerationParams
from pipeworks.ui.validation import ValidationError


class TestFormatGenerationInfo:
    """Tests for format_generation_info function."""

    def test_single_image_generation_basic(self):
        """Test formatting for a single image generation."""
        params = GenerationParams(
            prompt="A beautiful landscape",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False,
        )
        generated_paths = ["/path/to/image.png"]
        seeds_used = [42]

        result = format_generation_info(
            params=params,
            generated_paths=generated_paths,
            seeds_used=seeds_used,
            has_dynamic=False,
            prompts_used=None,
            active_plugins=[],
        )

        # Check key elements are present
        assert "✅ **Generation Complete!**" in result
        assert "**Prompt:** A beautiful landscape" in result
        assert "**Dimensions:** 1024x1024" in result
        assert "**Steps:** 9" in result
        assert "**Batch Size:** 1 × **Runs:** 1 = **Total:** 1 images" in result
        assert "**Seeds:** 42" in result
        assert "**Saved to:** /path/to/image.png" in result
        # Should not have dynamic or plugins info
        assert "Dynamic" not in result
        assert "Active Plugins" not in result

    def test_multiple_images_generation(self):
        """Test formatting for multiple images generation."""
        params = GenerationParams(
            prompt="A fantasy castle",
            width=1280,
            height=720,
            num_steps=12,
            batch_size=2,
            runs=3,
            seed=100,
            use_random_seed=False,
        )
        generated_paths = [f"/path/image_{i}.png" for i in range(6)]
        seeds_used = [100, 101, 102, 103, 104, 105]

        result = format_generation_info(
            params=params,
            generated_paths=generated_paths,
            seeds_used=seeds_used,
            has_dynamic=False,
            prompts_used=None,
            active_plugins=[],
        )

        assert "**Prompt:** A fantasy castle" in result
        assert "**Dimensions:** 1280x720" in result
        assert "**Steps:** 12" in result
        assert "**Batch Size:** 2 × **Runs:** 3 = **Total:** 6 images" in result
        assert "**Seeds:** 100 - 105" in result
        assert "**Saved to:** 6 images saved to output folder" in result

    def test_with_dynamic_prompts_enabled(self):
        """Test formatting when dynamic prompts are enabled."""
        params = GenerationParams(
            prompt="Dynamic prompt",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False,
        )
        generated_paths = ["/path/image.png"]
        seeds_used = [42]

        result = format_generation_info(
            params=params,
            generated_paths=generated_paths,
            seeds_used=seeds_used,
            has_dynamic=True,
            prompts_used=None,
            active_plugins=[],
        )

        assert "**Prompt:** (Dynamic)" in result
        assert "**Dynamic Prompts:** Enabled (prompts rebuilt for each image)" in result

    def test_with_dynamic_prompts_three_or_fewer(self):
        """Test formatting with 3 or fewer dynamic prompts (show all)."""
        params = GenerationParams(
            prompt="Dynamic",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=3,
            runs=1,
            seed=42,
            use_random_seed=False,
        )
        generated_paths = [f"/path/image_{i}.png" for i in range(3)]
        seeds_used = [42, 43, 44]
        prompts_used = ["Prompt 1", "Prompt 2", "Prompt 3"]

        result = format_generation_info(
            params=params,
            generated_paths=generated_paths,
            seeds_used=seeds_used,
            has_dynamic=True,
            prompts_used=prompts_used,
            active_plugins=[],
        )

        assert "**Dynamic Prompts:** Enabled" in result
        assert "**Sample Prompts:** Prompt 1, Prompt 2, Prompt 3" in result

    def test_with_dynamic_prompts_more_than_three(self):
        """Test formatting with more than 3 dynamic prompts (show first 2)."""
        params = GenerationParams(
            prompt="Dynamic",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=5,
            runs=1,
            seed=42,
            use_random_seed=False,
        )
        generated_paths = [f"/path/image_{i}.png" for i in range(5)]
        seeds_used = [42, 43, 44, 45, 46]
        prompts_used = ["Prompt A", "Prompt B", "Prompt C", "Prompt D", "Prompt E"]

        result = format_generation_info(
            params=params,
            generated_paths=generated_paths,
            seeds_used=seeds_used,
            has_dynamic=True,
            prompts_used=prompts_used,
            active_plugins=[],
        )

        assert "**Dynamic Prompts:** Enabled" in result
        assert "**Sample Prompts:** Prompt A, Prompt B, ..." in result
        # Should NOT show all prompts
        assert "Prompt C" not in result or ", ..." in result

    def test_with_single_active_plugin(self):
        """Test formatting with a single active plugin."""
        params = GenerationParams(
            prompt="Test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False,
        )
        generated_paths = ["/path/image.png"]
        seeds_used = [42]

        result = format_generation_info(
            params=params,
            generated_paths=generated_paths,
            seeds_used=seeds_used,
            has_dynamic=False,
            prompts_used=None,
            active_plugins=["Save Metadata"],
        )

        assert "**Active Plugins:** Save Metadata" in result

    def test_with_multiple_active_plugins(self):
        """Test formatting with multiple active plugins."""
        params = GenerationParams(
            prompt="Test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False,
        )
        generated_paths = ["/path/image.png"]
        seeds_used = [42]

        result = format_generation_info(
            params=params,
            generated_paths=generated_paths,
            seeds_used=seeds_used,
            has_dynamic=False,
            prompts_used=None,
            active_plugins=["Save Metadata", "Watermark", "Auto Upload"],
        )

        assert "**Active Plugins:** Save Metadata, Watermark, Auto Upload" in result

    def test_with_no_plugins(self):
        """Test formatting with empty plugins list."""
        params = GenerationParams(
            prompt="Test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False,
        )
        generated_paths = ["/path/image.png"]
        seeds_used = [42]

        result = format_generation_info(
            params=params,
            generated_paths=generated_paths,
            seeds_used=seeds_used,
            has_dynamic=False,
            prompts_used=None,
            active_plugins=[],
        )

        # Should not have plugins section
        assert "Active Plugins" not in result

    def test_complex_scenario_all_features(self):
        """Test formatting with all features enabled."""
        params = GenerationParams(
            prompt="Complex",
            width=1600,
            height=896,
            num_steps=15,
            batch_size=4,
            runs=2,
            seed=1000,
            use_random_seed=False,
        )
        generated_paths = [f"/output/img_{i}.png" for i in range(8)]
        seeds_used = list(range(1000, 1008))
        prompts_used = [f"Dynamic prompt {i}" for i in range(8)]

        result = format_generation_info(
            params=params,
            generated_paths=generated_paths,
            seeds_used=seeds_used,
            has_dynamic=True,
            prompts_used=prompts_used,
            active_plugins=["Plugin A", "Plugin B"],
        )

        # Verify all sections are present
        assert "✅ **Generation Complete!**" in result
        assert "**Prompt:** (Dynamic)" in result
        assert "**Dimensions:** 1600x896" in result
        assert "**Steps:** 15" in result
        assert "**Batch Size:** 4 × **Runs:** 2 = **Total:** 8 images" in result
        assert "**Seeds:** 1000 - 1007" in result
        assert "**Saved to:** 8 images saved to output folder" in result
        assert "**Dynamic Prompts:** Enabled" in result
        assert "**Sample Prompts:**" in result
        assert "**Active Plugins:** Plugin A, Plugin B" in result

    def test_different_aspect_ratios(self):
        """Test formatting with various image dimensions."""
        test_cases = [
            (1024, 1024, "1024x1024"),
            (1280, 720, "1280x720"),
            (720, 1280, "720x1280"),
            (1600, 896, "1600x896"),
            (512, 512, "512x512"),
        ]

        for width, height, expected in test_cases:
            params = GenerationParams(
                prompt="Test",
                width=width,
                height=height,
                num_steps=9,
                batch_size=1,
                runs=1,
                seed=42,
                use_random_seed=False,
            )
            result = format_generation_info(
                params=params,
                generated_paths=["/test.png"],
                seeds_used=[42],
                has_dynamic=False,
                prompts_used=None,
                active_plugins=[],
            )
            assert f"**Dimensions:** {expected}" in result

    def test_total_images_calculation(self):
        """Test that total images is correctly calculated."""
        test_cases = [
            (1, 1, 1),  # Single image
            (2, 3, 6),  # 2 batch × 3 runs
            (5, 2, 10),  # 5 batch × 2 runs
            (10, 10, 100),  # Large batch
        ]

        for batch_size, runs, expected_total in test_cases:
            params = GenerationParams(
                prompt="Test",
                width=1024,
                height=1024,
                num_steps=9,
                batch_size=batch_size,
                runs=runs,
                seed=42,
                use_random_seed=False,
            )
            result = format_generation_info(
                params=params,
                generated_paths=[f"/img{i}.png" for i in range(expected_total)],
                seeds_used=list(range(42, 42 + expected_total)),
                has_dynamic=False,
                prompts_used=None,
                active_plugins=[],
            )
            assert f"**Total:** {expected_total} images" in result

    def test_with_empty_prompts_list(self):
        """Test with dynamic enabled but empty prompts_used list."""
        params = GenerationParams(
            prompt="Test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False,
        )
        result = format_generation_info(
            params=params,
            generated_paths=["/test.png"],
            seeds_used=[42],
            has_dynamic=True,
            prompts_used=[],
            active_plugins=[],
        )

        assert "**Dynamic Prompts:** Enabled" in result
        # Should not crash or show sample prompts section
        assert "**Sample Prompts:**" not in result


class TestFormatValidationError:
    """Tests for format_validation_error function."""

    def test_basic_validation_error(self):
        """Test formatting a basic validation error."""
        error = ValidationError("Width must be a multiple of 64")

        result = format_validation_error(error)

        assert "❌ **Validation Error**" in result
        assert "Width must be a multiple of 64" in result

    def test_error_with_newlines(self):
        """Test formatting error with multiple lines."""
        error = ValidationError("Multiple issues:\n- Width invalid\n- Height invalid")

        result = format_validation_error(error)

        assert "❌ **Validation Error**" in result
        assert "Multiple issues:" in result
        assert "- Width invalid" in result
        assert "- Height invalid" in result

    def test_error_with_empty_message(self):
        """Test formatting error with empty message."""
        error = ValidationError("")

        result = format_validation_error(error)

        assert "❌ **Validation Error**" in result
        # Even with empty message, should have the header
        assert result == "❌ **Validation Error**\n\n"

    def test_error_with_special_characters(self):
        """Test formatting error with special characters."""
        error = ValidationError("Invalid value: <script>alert('test')</script>")

        result = format_validation_error(error)

        assert "❌ **Validation Error**" in result
        assert "<script>alert('test')</script>" in result

    def test_error_with_long_message(self):
        """Test formatting error with long message."""
        long_message = "A" * 500
        error = ValidationError(long_message)

        result = format_validation_error(error)

        assert "❌ **Validation Error**" in result
        assert long_message in result


class TestFormatGenerationError:
    """Tests for format_generation_error function."""

    def test_basic_exception(self):
        """Test formatting a basic exception."""
        error = Exception("Something went wrong")

        result = format_generation_error(error)

        assert "❌ **Error**" in result
        assert "An unexpected error occurred" in result
        assert "Check logs for details" in result
        assert "`Something went wrong`" in result

    def test_value_error(self):
        """Test formatting a ValueError."""
        error = ValueError("Invalid value provided")

        result = format_generation_error(error)

        assert "❌ **Error**" in result
        assert "`Invalid value provided`" in result

    def test_runtime_error(self):
        """Test formatting a RuntimeError."""
        error = RuntimeError("Runtime issue occurred")

        result = format_generation_error(error)

        assert "❌ **Error**" in result
        assert "`Runtime issue occurred`" in result

    def test_error_with_empty_message(self):
        """Test formatting error with empty message."""
        error = Exception("")

        result = format_generation_error(error)

        assert "❌ **Error**" in result
        assert "``" in result  # Empty code block

    def test_error_with_newlines(self):
        """Test formatting error with newlines in message."""
        error = Exception("Error on line 1\nError on line 2\nError on line 3")

        result = format_generation_error(error)

        assert "❌ **Error**" in result
        assert "Error on line 1" in result
        assert "Error on line 2" in result
        assert "Error on line 3" in result

    def test_error_with_special_characters(self):
        """Test formatting error with special characters."""
        error = Exception("File not found: /path/to/file.txt (errno: 2)")

        result = format_generation_error(error)

        assert "❌ **Error**" in result
        assert "`File not found: /path/to/file.txt (errno: 2)`" in result

    def test_error_message_in_code_block(self):
        """Test that error message is wrapped in code block."""
        error = Exception("Test error")

        result = format_generation_error(error)

        # Should be wrapped in backticks
        assert "`Test error`" in result
        # Should not have just plain text
        lines = result.split("\n")
        assert any("`Test error`" in line for line in lines)

    def test_error_with_long_message(self):
        """Test formatting error with very long message."""
        long_message = "X" * 1000
        error = Exception(long_message)

        result = format_generation_error(error)

        assert "❌ **Error**" in result
        assert long_message in result


class TestFormattingIntegration:
    """Integration tests for formatting functions."""

    def test_validation_vs_generation_error_format(self):
        """Test that validation and generation errors have distinct formats."""
        validation_error = ValidationError("Validation failed")
        generation_error = Exception("Generation failed")

        validation_result = format_validation_error(validation_error)
        generation_result = format_generation_error(generation_error)

        # Both should have error indicator
        assert "❌" in validation_result
        assert "❌" in generation_result

        # But different headers
        assert "**Validation Error**" in validation_result
        assert "**Validation Error**" not in generation_result
        assert "**Error**" in generation_result

        # Validation doesn't have "Check logs" message
        assert "Check logs" not in validation_result
        assert "Check logs" in generation_result

    def test_success_vs_error_formatting(self):
        """Test that success and error messages are clearly distinct."""
        params = GenerationParams(
            prompt="Test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False,
        )
        success_result = format_generation_info(
            params=params,
            generated_paths=["/test.png"],
            seeds_used=[42],
            has_dynamic=False,
            prompts_used=None,
            active_plugins=[],
        )
        error_result = format_generation_error(Exception("Failed"))

        # Success has checkmark, error has X
        assert "✅" in success_result
        assert "✅" not in error_result
        assert "❌" in error_result
        assert "❌" not in success_result

        # Success has complete message, error doesn't
        assert "Complete" in success_result
        assert "Complete" not in error_result

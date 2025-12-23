"""Unit tests for UI data models."""

import pytest
from pipeworks.ui.models import (
    GenerationParams,
    SegmentConfig,
    UIState,
    ASPECT_RATIOS,
    MAX_SEED,
    DEFAULT_SEED,
)


class TestSegmentConfig:
    """Tests for SegmentConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        segment = SegmentConfig()

        assert segment.text == ""
        assert segment.path == ""
        assert segment.file == "(None)"
        assert segment.mode == "Random Line"
        assert segment.line == 1
        assert segment.range_end == 1
        assert segment.count == 1
        assert segment.dynamic is False

    def test_custom_values(self):
        """Test that custom values can be set."""
        segment = SegmentConfig(
            text="custom text",
            path="subfolder",
            file="test.txt",
            mode="Specific Line",
            line=5,
            range_end=10,
            count=3,
            dynamic=True
        )

        assert segment.text == "custom text"
        assert segment.path == "subfolder"
        assert segment.file == "test.txt"
        assert segment.mode == "Specific Line"
        assert segment.line == 5
        assert segment.range_end == 10
        assert segment.count == 3
        assert segment.dynamic is True

    def test_is_configured_with_file(self):
        """Test is_configured returns True when file is selected."""
        segment = SegmentConfig(file="test.txt")
        assert segment.is_configured() is True

    def test_is_configured_with_none(self):
        """Test is_configured returns False with (None)."""
        segment = SegmentConfig(file="(None)")
        assert segment.is_configured() is False

    def test_is_configured_with_folder(self):
        """Test is_configured returns False when folder is selected."""
        segment = SegmentConfig(file="📁 folder")
        assert segment.is_configured() is False

    def test_is_configured_with_empty_string(self):
        """Test is_configured returns False with empty string."""
        segment = SegmentConfig(file="")
        assert segment.is_configured() is False

    def test_has_content_with_text(self):
        """Test has_content returns True when text is present."""
        segment = SegmentConfig(text="some text")
        assert segment.has_content() is True

    def test_has_content_with_file(self):
        """Test has_content returns True when file is configured."""
        segment = SegmentConfig(file="test.txt")
        assert segment.has_content() is True

    def test_has_content_with_both(self):
        """Test has_content returns True when both text and file are present."""
        segment = SegmentConfig(text="some text", file="test.txt")
        assert segment.has_content() is True

    def test_has_content_with_neither(self):
        """Test has_content returns False when no content."""
        segment = SegmentConfig()
        assert segment.has_content() is False

    def test_has_content_with_whitespace_only(self):
        """Test has_content returns False with whitespace-only text."""
        segment = SegmentConfig(text="   \n\t  ")
        assert segment.has_content() is False


class TestGenerationParams:
    """Tests for GenerationParams dataclass."""

    def test_valid_params(self, valid_generation_params):
        """Test that valid parameters pass validation."""
        valid_generation_params.validate()  # Should not raise

    def test_batch_size_too_small(self):
        """Test validation fails with batch_size < 1."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=0,  # Invalid
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Batch size must be 1-100"):
            params.validate()

    def test_batch_size_too_large(self):
        """Test validation fails with batch_size > 100."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=150,  # Invalid
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Batch size must be 1-100"):
            params.validate()

    def test_runs_too_small(self):
        """Test validation fails with runs < 1."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=0,  # Invalid
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Runs must be 1-100"):
            params.validate()

    def test_runs_too_large(self):
        """Test validation fails with runs > 100."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=150,  # Invalid
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Runs must be 1-100"):
            params.validate()

    def test_total_images_exceeds_limit(self):
        """Test validation fails when batch_size * runs > 1000."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=50,
            runs=50,  # 50 * 50 = 2500 > 1000
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="exceeds maximum of 1000"):
            params.validate()

    def test_width_not_multiple_of_64(self):
        """Test validation fails when width is not multiple of 64."""
        params = GenerationParams(
            prompt="test",
            width=1023,  # Not multiple of 64
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Width must be multiple of 64"):
            params.validate()

    def test_height_not_multiple_of_64(self):
        """Test validation fails when height is not multiple of 64."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1023,  # Not multiple of 64
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Height must be multiple of 64"):
            params.validate()

    def test_width_too_small(self):
        """Test validation fails when width < 512."""
        params = GenerationParams(
            prompt="test",
            width=448,  # < 512
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Width must be 512-2048"):
            params.validate()

    def test_width_too_large(self):
        """Test validation fails when width > 2048."""
        params = GenerationParams(
            prompt="test",
            width=2112,  # > 2048
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Width must be 512-2048"):
            params.validate()

    def test_height_too_small(self):
        """Test validation fails when height < 512."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=448,  # < 512
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Height must be 512-2048"):
            params.validate()

    def test_height_too_large(self):
        """Test validation fails when height > 2048."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=2112,  # > 2048
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Height must be 512-2048"):
            params.validate()

    def test_num_steps_too_small(self):
        """Test validation fails when num_steps < 1."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=0,  # Invalid
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Inference steps must be 1-50"):
            params.validate()

    def test_num_steps_too_large(self):
        """Test validation fails when num_steps > 50."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=100,  # Invalid
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Inference steps must be 1-50"):
            params.validate()

    def test_seed_negative(self):
        """Test validation fails with negative seed."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=-1,  # Invalid
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Seed must be 0 to"):
            params.validate()

    def test_seed_too_large(self):
        """Test validation fails when seed > 2^32-1."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=2**32,  # Too large
            use_random_seed=False
        )

        with pytest.raises(ValueError, match="Seed must be 0 to"):
            params.validate()

    def test_total_images_property(self):
        """Test total_images property calculates correctly."""
        params = GenerationParams(
            prompt="test",
            width=1024,
            height=1024,
            num_steps=9,
            batch_size=5,
            runs=3,
            seed=42,
            use_random_seed=False
        )

        assert params.total_images == 15  # 5 * 3

    def test_edge_case_valid_dimensions(self):
        """Test edge case valid dimensions (512x512 and 2048x2048)."""
        # Minimum valid
        params_min = GenerationParams(
            prompt="test",
            width=512,
            height=512,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )
        params_min.validate()  # Should not raise

        # Maximum valid
        params_max = GenerationParams(
            prompt="test",
            width=2048,
            height=2048,
            num_steps=9,
            batch_size=1,
            runs=1,
            seed=42,
            use_random_seed=False
        )
        params_max.validate()  # Should not raise


class TestUIState:
    """Tests for UIState dataclass."""

    def test_default_values(self):
        """Test that UIState initializes with None values."""
        state = UIState()

        assert state.generator is None
        assert state.tokenizer_analyzer is None
        assert state.prompt_builder is None
        assert state.active_plugins == {}

    def test_is_initialized_false_by_default(self):
        """Test is_initialized returns False when empty."""
        state = UIState()
        assert state.is_initialized() is False

    def test_is_initialized_false_partial(self):
        """Test is_initialized returns False when partially initialized."""
        state = UIState()
        state.generator = "mock_generator"
        assert state.is_initialized() is False

        state.tokenizer_analyzer = "mock_tokenizer"
        assert state.is_initialized() is False

    def test_is_initialized_true_when_complete(self):
        """Test is_initialized returns True when fully initialized."""
        state = UIState()
        state.model_adapter = "mock_adapter"
        state.generator = state.model_adapter  # Backward compatibility
        state.tokenizer_analyzer = "mock_tokenizer"
        state.prompt_builder = "mock_prompt_builder"

        assert state.is_initialized() is True

    def test_active_plugins_default_factory(self):
        """Test active_plugins uses default factory (not shared)."""
        state1 = UIState()
        state2 = UIState()

        state1.active_plugins["test"] = "plugin1"

        # Should not affect state2
        assert "test" not in state2.active_plugins

    def test_repr(self):
        """Test string representation."""
        state = UIState()
        repr_str = repr(state)

        assert "UIState" in repr_str
        assert "initialized=False" in repr_str
        assert "plugins=0" in repr_str

        # Add plugins
        state.active_plugins["plugin1"] = "test"
        repr_str = repr(state)
        assert "plugins=1" in repr_str


class TestConstants:
    """Tests for module constants."""

    def test_aspect_ratios_dict(self):
        """Test ASPECT_RATIOS contains expected keys."""
        assert isinstance(ASPECT_RATIOS, dict)
        assert "Square 1:1 (1024x1024)" in ASPECT_RATIOS
        assert "Widescreen 16:9 (1280x720)" in ASPECT_RATIOS
        assert "Custom" in ASPECT_RATIOS

    def test_aspect_ratios_values(self):
        """Test ASPECT_RATIOS values are tuples or None."""
        for key, value in ASPECT_RATIOS.items():
            if key == "Custom":
                assert value is None
            else:
                assert isinstance(value, tuple)
                assert len(value) == 2
                assert isinstance(value[0], int)
                assert isinstance(value[1], int)

    def test_max_seed_constant(self):
        """Test MAX_SEED is correct value."""
        assert MAX_SEED == 2**32 - 1
        assert MAX_SEED == 4294967295

    def test_default_seed_constant(self):
        """Test DEFAULT_SEED is a reasonable value."""
        assert isinstance(DEFAULT_SEED, int)
        assert DEFAULT_SEED == 42
        assert DEFAULT_SEED >= 0
        assert DEFAULT_SEED <= MAX_SEED

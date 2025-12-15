"""Unit tests for UI state management."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from pipeworks.ui.state import (
    initialize_ui_state,
    update_generator_plugins,
    toggle_plugin,
    cleanup_ui_state,
)
from pipeworks.ui.models import UIState


class TestInitializeUIState:
    """Tests for initialize_ui_state function."""

    def test_initialize_none_creates_new_state(self):
        """Test that passing None creates a new UIState."""
        with patch('pipeworks.ui.state.ImageGenerator'), \
             patch('pipeworks.ui.state.TokenizerAnalyzer'), \
             patch('pipeworks.ui.state.PromptBuilder'):

            result = initialize_ui_state(None)

            assert isinstance(result, UIState)

    def test_initialize_returns_if_already_initialized(self):
        """Test that already initialized state is returned as-is."""
        state = UIState()
        state.generator = Mock()
        state.tokenizer_analyzer = Mock()
        state.prompt_builder = Mock()

        result = initialize_ui_state(state)

        assert result is state
        assert result.is_initialized()

    def test_initialize_creates_generator(self, test_config):
        """Test that generator is created if None."""
        state = UIState()

        with patch('pipeworks.ui.state.ImageGenerator') as MockGen, \
             patch('pipeworks.ui.state.TokenizerAnalyzer') as MockTok, \
             patch('pipeworks.ui.state.PromptBuilder') as MockPB, \
             patch('pipeworks.ui.state.config', test_config):

            mock_gen = Mock()
            MockGen.return_value = mock_gen

            result = initialize_ui_state(state)

            assert result.generator is mock_gen
            MockGen.assert_called_once()

    def test_initialize_creates_tokenizer(self, test_config):
        """Test that tokenizer is created if None."""
        state = UIState()

        with patch('pipeworks.ui.state.ImageGenerator'), \
             patch('pipeworks.ui.state.TokenizerAnalyzer') as MockTok, \
             patch('pipeworks.ui.state.PromptBuilder'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_tok = Mock()
            MockTok.return_value = mock_tok

            result = initialize_ui_state(state)

            assert result.tokenizer_analyzer is mock_tok
            MockTok.assert_called_once()

    def test_initialize_creates_prompt_builder(self, test_config):
        """Test that prompt builder is created if None."""
        state = UIState()

        with patch('pipeworks.ui.state.ImageGenerator'), \
             patch('pipeworks.ui.state.TokenizerAnalyzer'), \
             patch('pipeworks.ui.state.PromptBuilder') as MockPB, \
             patch('pipeworks.ui.state.config', test_config):

            mock_pb = Mock()
            MockPB.return_value = mock_pb

            result = initialize_ui_state(state)

            assert result.prompt_builder is mock_pb
            MockPB.assert_called_once()

    def test_initialize_loads_model(self, test_config):
        """Test that model loading is attempted."""
        state = UIState()

        with patch('pipeworks.ui.state.ImageGenerator') as MockGen, \
             patch('pipeworks.ui.state.TokenizerAnalyzer'), \
             patch('pipeworks.ui.state.PromptBuilder'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_gen = Mock()
            MockGen.return_value = mock_gen

            initialize_ui_state(state)

            mock_gen.load_model.assert_called_once()

    def test_initialize_handles_model_load_failure(self, test_config):
        """Test that model load failure is handled gracefully."""
        state = UIState()

        with patch('pipeworks.ui.state.ImageGenerator') as MockGen, \
             patch('pipeworks.ui.state.TokenizerAnalyzer'), \
             patch('pipeworks.ui.state.PromptBuilder'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_gen = Mock()
            mock_gen.load_model.side_effect = Exception("Model load failed")
            MockGen.return_value = mock_gen

            # Should not raise, should log error
            result = initialize_ui_state(state)

            assert result.generator is mock_gen

    def test_initialize_loads_tokenizer(self, test_config):
        """Test that tokenizer loading is attempted."""
        state = UIState()

        with patch('pipeworks.ui.state.ImageGenerator'), \
             patch('pipeworks.ui.state.TokenizerAnalyzer') as MockTok, \
             patch('pipeworks.ui.state.PromptBuilder'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_tok = Mock()
            MockTok.return_value = mock_tok

            initialize_ui_state(state)

            mock_tok.load.assert_called_once()

    def test_initialize_partial_state(self, test_config):
        """Test that partial state is completed."""
        state = UIState()
        state.generator = Mock()  # Already set

        with patch('pipeworks.ui.state.ImageGenerator'), \
             patch('pipeworks.ui.state.TokenizerAnalyzer') as MockTok, \
             patch('pipeworks.ui.state.PromptBuilder') as MockPB, \
             patch('pipeworks.ui.state.config', test_config):

            mock_tok = Mock()
            mock_pb = Mock()
            MockTok.return_value = mock_tok
            MockPB.return_value = mock_pb

            result = initialize_ui_state(state)

            # Generator should not be replaced
            assert result.generator is state.generator
            # But tokenizer and prompt_builder should be created
            assert result.tokenizer_analyzer is mock_tok
            assert result.prompt_builder is mock_pb


class TestUpdateGeneratorPlugins:
    """Tests for update_generator_plugins function."""

    def test_update_with_no_generator(self):
        """Test that update handles missing generator gracefully."""
        state = UIState()

        result = update_generator_plugins(state)

        assert result is state
        assert state.generator is None

    def test_update_with_enabled_plugins(self):
        """Test that enabled plugins are added to generator."""
        state = UIState()
        state.generator = Mock()

        plugin1 = Mock(enabled=True)
        plugin2 = Mock(enabled=True)
        plugin3 = Mock(enabled=False)

        state.active_plugins = {
            "plugin1": plugin1,
            "plugin2": plugin2,
            "plugin3": plugin3,
        }

        result = update_generator_plugins(state)

        assert len(state.generator.plugins) == 2
        assert plugin1 in state.generator.plugins
        assert plugin2 in state.generator.plugins
        assert plugin3 not in state.generator.plugins

    def test_update_with_no_enabled_plugins(self):
        """Test that generator gets empty list when no enabled plugins."""
        state = UIState()
        state.generator = Mock()

        plugin1 = Mock(enabled=False)
        state.active_plugins = {"plugin1": plugin1}

        result = update_generator_plugins(state)

        assert state.generator.plugins == []

    def test_update_with_empty_plugins(self):
        """Test that generator gets empty list when no plugins."""
        state = UIState()
        state.generator = Mock()
        state.active_plugins = {}

        result = update_generator_plugins(state)

        assert state.generator.plugins == []


class TestTogglePlugin:
    """Tests for toggle_plugin function."""

    def test_enable_plugin_creates_instance(self):
        """Test that enabling plugin creates new instance."""
        state = UIState()
        state.generator = Mock()

        with patch('pipeworks.plugins.base.plugin_registry') as mock_registry:
            mock_plugin = Mock(enabled=True)
            mock_registry.instantiate.return_value = mock_plugin

            result = toggle_plugin(state, "TestPlugin", True, test_config="value")

            mock_registry.instantiate.assert_called_once_with(
                "TestPlugin",
                test_config="value"
            )
            assert "TestPlugin" in state.active_plugins
            assert state.active_plugins["TestPlugin"] is mock_plugin

    def test_disable_plugin_sets_enabled_false(self):
        """Test that disabling plugin sets enabled to False."""
        state = UIState()
        state.generator = Mock()

        plugin = Mock(enabled=True)
        state.active_plugins = {"TestPlugin": plugin}

        result = toggle_plugin(state, "TestPlugin", False)

        assert plugin.enabled is False

    def test_disable_nonexistent_plugin(self):
        """Test that disabling nonexistent plugin doesn't crash."""
        state = UIState()
        state.generator = Mock()

        result = toggle_plugin(state, "NonExistent", False)

        # Should not raise, just handle gracefully
        assert result is state

    def test_enable_updates_generator_plugins(self):
        """Test that enabling plugin updates generator."""
        state = UIState()
        state.generator = Mock()

        with patch('pipeworks.plugins.base.plugin_registry') as mock_registry:
            mock_plugin = Mock(enabled=True)
            mock_registry.instantiate.return_value = mock_plugin

            result = toggle_plugin(state, "TestPlugin", True)

            # Generator.plugins should be updated
            assert len(state.generator.plugins) > 0

    def test_enable_with_config_params(self):
        """Test that plugin config params are passed through."""
        state = UIState()
        state.generator = Mock()

        with patch('pipeworks.plugins.base.plugin_registry') as mock_registry:
            mock_plugin = Mock(enabled=True)
            mock_registry.instantiate.return_value = mock_plugin

            result = toggle_plugin(
                state,
                "SaveMetadata",
                True,
                folder_name="metadata",
                filename_prefix="test_"
            )

            mock_registry.instantiate.assert_called_once_with(
                "SaveMetadata",
                folder_name="metadata",
                filename_prefix="test_"
            )


class TestCleanupUIState:
    """Tests for cleanup_ui_state function."""

    def test_cleanup_unloads_model(self):
        """Test that cleanup unloads the model."""
        state = UIState()
        mock_generator = Mock()
        state.generator = mock_generator

        cleanup_ui_state(state)

        mock_generator.unload_model.assert_called_once()

    def test_cleanup_clears_generator(self):
        """Test that cleanup sets generator to None."""
        state = UIState()
        state.generator = Mock()

        cleanup_ui_state(state)

        assert state.generator is None

    def test_cleanup_clears_tokenizer(self):
        """Test that cleanup sets tokenizer to None."""
        state = UIState()
        state.tokenizer_analyzer = Mock()

        cleanup_ui_state(state)

        assert state.tokenizer_analyzer is None

    def test_cleanup_clears_prompt_builder(self):
        """Test that cleanup sets prompt_builder to None."""
        state = UIState()
        state.prompt_builder = Mock()

        cleanup_ui_state(state)

        assert state.prompt_builder is None

    def test_cleanup_clears_plugins(self):
        """Test that cleanup clears active_plugins dict."""
        state = UIState()
        state.active_plugins = {"plugin1": Mock(), "plugin2": Mock()}

        cleanup_ui_state(state)

        assert len(state.active_plugins) == 0

    def test_cleanup_handles_unload_failure(self):
        """Test that cleanup handles model unload failure gracefully."""
        state = UIState()
        state.generator = Mock()
        state.generator.unload_model.side_effect = Exception("Unload failed")

        # Should not raise
        cleanup_ui_state(state)

        assert state.generator is None

    def test_cleanup_with_partial_state(self):
        """Test that cleanup works with partial state."""
        state = UIState()
        state.generator = None  # Already None
        state.tokenizer_analyzer = Mock()

        # Should not crash
        cleanup_ui_state(state)

        assert state.tokenizer_analyzer is None

    def test_cleanup_empty_state(self):
        """Test that cleanup works with empty state."""
        state = UIState()

        # Should not crash
        cleanup_ui_state(state)

        assert state.generator is None
        assert state.tokenizer_analyzer is None
        assert state.prompt_builder is None
        assert len(state.active_plugins) == 0

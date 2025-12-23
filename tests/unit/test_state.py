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
        with patch('pipeworks.ui.state.model_registry') as mock_registry, \
             patch('pipeworks.ui.state.TokenizerAnalyzer'), \
             patch('pipeworks.ui.state.PromptBuilder'), \
             patch('pipeworks.ui.state.GalleryBrowser'), \
             patch('pipeworks.ui.state.FavoritesDB'), \
             patch('pipeworks.ui.state.CatalogManager'):

            mock_registry.instantiate.return_value = Mock()

            result = initialize_ui_state(None)

            assert isinstance(result, UIState)

    def test_initialize_returns_if_already_initialized(self):
        """Test that already initialized state is returned as-is."""
        state = UIState()
        state.model_adapter = Mock()
        state.generator = state.model_adapter  # Backward compatibility
        state.tokenizer_analyzer = Mock()
        state.prompt_builder = Mock()

        result = initialize_ui_state(state)

        assert result is state
        assert result.is_initialized()

    def test_initialize_creates_generator(self, test_config):
        """Test that model adapter is created if None."""
        state = UIState()

        with patch('pipeworks.ui.state.model_registry') as mock_registry, \
             patch('pipeworks.ui.state.TokenizerAnalyzer') as MockTok, \
             patch('pipeworks.ui.state.PromptBuilder') as MockPB, \
             patch('pipeworks.ui.state.GalleryBrowser'), \
             patch('pipeworks.ui.state.FavoritesDB'), \
             patch('pipeworks.ui.state.CatalogManager'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_adapter = Mock()
            mock_adapter.load_model = Mock()
            mock_registry.instantiate.return_value = mock_adapter

            result = initialize_ui_state(state)

            assert result.model_adapter is mock_adapter
            assert result.generator is mock_adapter  # Backward compatibility alias
            mock_registry.instantiate.assert_called_once()

    def test_initialize_creates_tokenizer(self, test_config):
        """Test that tokenizer is created if None."""
        state = UIState()

        with patch('pipeworks.ui.state.model_registry') as mock_registry, \
             patch('pipeworks.ui.state.TokenizerAnalyzer') as MockTok, \
             patch('pipeworks.ui.state.PromptBuilder'), \
             patch('pipeworks.ui.state.GalleryBrowser'), \
             patch('pipeworks.ui.state.FavoritesDB'), \
             patch('pipeworks.ui.state.CatalogManager'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_registry.instantiate.return_value = Mock()
            mock_tok = Mock()
            MockTok.return_value = mock_tok

            result = initialize_ui_state(state)

            assert result.tokenizer_analyzer is mock_tok
            MockTok.assert_called_once()

    def test_initialize_creates_prompt_builder(self, test_config):
        """Test that prompt builder is created if None."""
        state = UIState()

        with patch('pipeworks.ui.state.model_registry') as mock_registry, \
             patch('pipeworks.ui.state.TokenizerAnalyzer'), \
             patch('pipeworks.ui.state.PromptBuilder') as MockPB, \
             patch('pipeworks.ui.state.GalleryBrowser'), \
             patch('pipeworks.ui.state.FavoritesDB'), \
             patch('pipeworks.ui.state.CatalogManager'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_registry.instantiate.return_value = Mock()
            mock_pb = Mock()
            MockPB.return_value = mock_pb

            result = initialize_ui_state(state)

            assert result.prompt_builder is mock_pb
            MockPB.assert_called_once()

    def test_initialize_loads_model(self, test_config):
        """Test that model loading is attempted."""
        state = UIState()

        with patch('pipeworks.ui.state.model_registry') as mock_registry, \
             patch('pipeworks.ui.state.TokenizerAnalyzer'), \
             patch('pipeworks.ui.state.PromptBuilder'), \
             patch('pipeworks.ui.state.GalleryBrowser'), \
             patch('pipeworks.ui.state.FavoritesDB'), \
             patch('pipeworks.ui.state.CatalogManager'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_adapter = Mock()
            mock_registry.instantiate.return_value = mock_adapter

            initialize_ui_state(state)

            mock_adapter.load_model.assert_called_once()

    def test_initialize_handles_model_load_failure(self, test_config):
        """Test that model load failure is handled gracefully."""
        state = UIState()

        with patch('pipeworks.ui.state.model_registry') as mock_registry, \
             patch('pipeworks.ui.state.TokenizerAnalyzer'), \
             patch('pipeworks.ui.state.PromptBuilder'), \
             patch('pipeworks.ui.state.GalleryBrowser'), \
             patch('pipeworks.ui.state.FavoritesDB'), \
             patch('pipeworks.ui.state.CatalogManager'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_adapter = Mock()
            mock_adapter.load_model.side_effect = Exception("Model load failed")
            mock_registry.instantiate.return_value = mock_adapter

            # Should not raise, should log error
            result = initialize_ui_state(state)

            assert result.model_adapter is mock_adapter
            assert result.generator is mock_adapter

    def test_initialize_loads_tokenizer(self, test_config):
        """Test that tokenizer loading is attempted."""
        state = UIState()

        with patch('pipeworks.ui.state.model_registry') as mock_registry, \
             patch('pipeworks.ui.state.TokenizerAnalyzer') as MockTok, \
             patch('pipeworks.ui.state.PromptBuilder'), \
             patch('pipeworks.ui.state.GalleryBrowser'), \
             patch('pipeworks.ui.state.FavoritesDB'), \
             patch('pipeworks.ui.state.CatalogManager'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_registry.instantiate.return_value = Mock()
            mock_tok = Mock()
            MockTok.return_value = mock_tok

            initialize_ui_state(state)

            mock_tok.load.assert_called_once()

    def test_initialize_partial_state(self, test_config):
        """Test that partial state is completed."""
        state = UIState()
        existing_adapter = Mock()
        state.model_adapter = existing_adapter
        state.generator = existing_adapter  # Backward compatibility

        with patch('pipeworks.ui.state.model_registry'), \
             patch('pipeworks.ui.state.TokenizerAnalyzer') as MockTok, \
             patch('pipeworks.ui.state.PromptBuilder') as MockPB, \
             patch('pipeworks.ui.state.GalleryBrowser'), \
             patch('pipeworks.ui.state.FavoritesDB'), \
             patch('pipeworks.ui.state.CatalogManager'), \
             patch('pipeworks.ui.state.config', test_config):

            mock_tok = Mock()
            mock_pb = Mock()
            MockTok.return_value = mock_tok
            MockPB.return_value = mock_pb

            result = initialize_ui_state(state)

            # Model adapter should not be replaced
            assert result.model_adapter is existing_adapter
            assert result.generator is existing_adapter
            # But tokenizer and prompt_builder should be created
            assert result.tokenizer_analyzer is mock_tok
            assert result.prompt_builder is mock_pb


class TestUpdateGeneratorPlugins:
    """Tests for update_generator_plugins function."""

    def test_update_with_no_generator(self):
        """Test that update handles missing model adapter gracefully."""
        state = UIState()

        result = update_generator_plugins(state)

        assert result is state
        assert state.model_adapter is None

    def test_update_with_enabled_plugins(self):
        """Test that enabled plugins are added to model adapter."""
        state = UIState()
        state.model_adapter = Mock()
        state.generator = state.model_adapter  # Backward compatibility

        plugin1 = Mock(enabled=True)
        plugin2 = Mock(enabled=True)
        plugin3 = Mock(enabled=False)

        state.active_plugins = {
            "plugin1": plugin1,
            "plugin2": plugin2,
            "plugin3": plugin3,
        }

        result = update_generator_plugins(state)

        assert len(state.model_adapter.plugins) == 2
        assert plugin1 in state.model_adapter.plugins
        assert plugin2 in state.model_adapter.plugins
        assert plugin3 not in state.model_adapter.plugins

    def test_update_with_no_enabled_plugins(self):
        """Test that model adapter gets empty list when no enabled plugins."""
        state = UIState()
        state.model_adapter = Mock()
        state.generator = state.model_adapter

        plugin1 = Mock(enabled=False)
        state.active_plugins = {"plugin1": plugin1}

        result = update_generator_plugins(state)

        assert state.model_adapter.plugins == []

    def test_update_with_empty_plugins(self):
        """Test that model adapter gets empty list when no plugins."""
        state = UIState()
        state.model_adapter = Mock()
        state.generator = state.model_adapter
        state.active_plugins = {}

        result = update_generator_plugins(state)

        assert state.model_adapter.plugins == []


class TestTogglePlugin:
    """Tests for toggle_plugin function."""

    def test_enable_plugin_creates_instance(self):
        """Test that enabling plugin creates new instance."""
        state = UIState()
        state.model_adapter = Mock()
        state.generator = state.model_adapter

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
        state.model_adapter = Mock()
        state.generator = state.model_adapter

        plugin = Mock(enabled=True)
        state.active_plugins = {"TestPlugin": plugin}

        result = toggle_plugin(state, "TestPlugin", False)

        assert plugin.enabled is False

    def test_disable_nonexistent_plugin(self):
        """Test that disabling nonexistent plugin doesn't crash."""
        state = UIState()
        state.model_adapter = Mock()
        state.generator = state.model_adapter

        result = toggle_plugin(state, "NonExistent", False)

        # Should not raise, just handle gracefully
        assert result is state

    def test_enable_updates_generator_plugins(self):
        """Test that enabling plugin updates model adapter."""
        state = UIState()
        state.model_adapter = Mock()
        state.generator = state.model_adapter

        with patch('pipeworks.plugins.base.plugin_registry') as mock_registry:
            mock_plugin = Mock(enabled=True)
            mock_registry.instantiate.return_value = mock_plugin

            result = toggle_plugin(state, "TestPlugin", True)

            # Model adapter plugins should be updated
            assert len(state.model_adapter.plugins) > 0

    def test_enable_with_config_params(self):
        """Test that plugin config params are passed through."""
        state = UIState()
        state.model_adapter = Mock()
        state.generator = state.model_adapter

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
        mock_adapter = Mock()
        state.model_adapter = mock_adapter
        state.generator = mock_adapter

        cleanup_ui_state(state)

        mock_adapter.unload_model.assert_called_once()

    def test_cleanup_clears_generator(self):
        """Test that cleanup sets model adapter to None."""
        state = UIState()
        state.model_adapter = Mock()
        state.generator = state.model_adapter

        cleanup_ui_state(state)

        assert state.model_adapter is None
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
        mock_adapter = Mock()
        mock_adapter.unload_model.side_effect = Exception("Unload failed")
        state.model_adapter = mock_adapter
        state.generator = mock_adapter

        # Should not raise
        cleanup_ui_state(state)

        assert state.model_adapter is None
        assert state.generator is None

    def test_cleanup_with_partial_state(self):
        """Test that cleanup works with partial state."""
        state = UIState()
        state.model_adapter = None  # Already None
        state.tokenizer_analyzer = Mock()

        # Should not crash
        cleanup_ui_state(state)

        assert state.tokenizer_analyzer is None

    def test_cleanup_empty_state(self):
        """Test that cleanup works with empty state."""
        state = UIState()

        # Should not crash
        cleanup_ui_state(state)

        assert state.model_adapter is None
        assert state.generator is None
        assert state.tokenizer_analyzer is None
        assert state.prompt_builder is None
        assert len(state.active_plugins) == 0

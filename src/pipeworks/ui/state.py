"""State management utilities for Pipeworks UI.

This module handles the initialization and management of UI state, including
model adapters, plugins, and other session components.
"""

import logging

from pipeworks.core.catalog_manager import CatalogManager
from pipeworks.core.config import config
from pipeworks.core.favorites_db import FavoritesDB
from pipeworks.core.gallery_browser import GalleryBrowser
from pipeworks.core.model_adapters import model_registry
from pipeworks.core.prompt_builder import PromptBuilder
from pipeworks.core.tokenizer import TokenizerAnalyzer

from .models import UIState

logger = logging.getLogger(__name__)


def initialize_ui_state(state: UIState | None = None, model_name: str | None = None) -> UIState:
    """Initialize or ensure UI state is ready.

    This function handles lazy initialization of the UI state components.
    If state is None or uninitialized, it creates and loads all necessary
    components (model adapter, tokenizer, prompt builder).

    Args:
        state: Existing UIState or None
        model_name: Name of model adapter to use (default: from config)

    Returns:
        Initialized UIState instance
    """
    # Create new state if None
    if state is None:
        logger.info("Creating new UIState")
        state = UIState()

    # Set model name if provided
    if model_name:
        state.current_model_name = model_name

    # Use default from config if not set
    if not state.current_model_name:
        state.current_model_name = config.default_model_adapter

    # Check if already initialized
    if state.is_initialized():
        logger.debug("UIState already initialized")
        return state

    logger.info("Initializing UIState components...")

    try:
        # Initialize model adapter
        if state.model_adapter is None:
            logger.info(f"Initializing model adapter: {state.current_model_name}")
            state.model_adapter = model_registry.instantiate(
                state.current_model_name, config, plugins=[]
            )
            # Pre-load model
            try:
                state.model_adapter.load_model()
                logger.info("Model pre-loaded successfully")
            except Exception as e:
                logger.error(f"Failed to pre-load model: {e}")
                logger.warning("Model will be loaded on first generation attempt")


        # Initialize tokenizer analyzer
        if state.tokenizer_analyzer is None:
            logger.info("Initializing TokenizerAnalyzer")
            state.tokenizer_analyzer = TokenizerAnalyzer(
                model_id=config.model_id, cache_dir=config.models_dir
            )
            state.tokenizer_analyzer.load()
            logger.info("TokenizerAnalyzer loaded successfully")

        # Initialize prompt builder
        if state.prompt_builder is None:
            logger.info("Initializing PromptBuilder")
            state.prompt_builder = PromptBuilder(config.inputs_dir)
            logger.info("PromptBuilder initialized successfully")

        # Initialize gallery browser (lazy-loaded for gallery tab)
        if state.gallery_browser is None:
            logger.info("Initializing GalleryBrowser")
            state.gallery_browser = GalleryBrowser(config.outputs_dir, config.catalog_dir)
            logger.info("GalleryBrowser initialized successfully")

        # Initialize favorites database (lazy-loaded for gallery tab)
        if state.favorites_db is None:
            logger.info("Initializing FavoritesDB")
            db_path = config.outputs_dir / ".pipeworks_favorites.db"
            state.favorites_db = FavoritesDB(db_path)
            logger.info("FavoritesDB initialized successfully")

        # Initialize catalog manager (lazy-loaded for gallery tab)
        if state.catalog_manager is None:
            logger.info("Initializing CatalogManager")
            state.catalog_manager = CatalogManager(
                config.outputs_dir, config.catalog_dir, state.favorites_db
            )
            logger.info("CatalogManager initialized successfully")

        logger.info(f"UIState initialization complete: {state}")
        return state

    except Exception as e:
        logger.error(f"Error initializing UIState: {e}", exc_info=True)
        raise


def update_generator_plugins(state: UIState) -> UIState:
    """Update the model adapter's plugin list from active_plugins.

    Args:
        state: UI state containing model_adapter and active_plugins

    Returns:
        Updated state
    """
    if state.model_adapter is None:
        logger.warning("Cannot update plugins: model adapter not initialized")
        return state

    # Get list of enabled plugins
    enabled_plugins = [plugin for plugin in state.active_plugins.values() if plugin.enabled]

    # Update model adapter's plugin list
    state.model_adapter.plugins = enabled_plugins
    logger.info(f"Updated model adapter with {len(enabled_plugins)} active plugins")

    return state


def toggle_plugin(state: UIState, plugin_name: str, enabled: bool, **plugin_config) -> UIState:
    """Toggle a plugin on/off and update its configuration.

    Args:
        state: UI state
        plugin_name: Name of the plugin
        enabled: Whether to enable the plugin
        **plugin_config: Plugin-specific configuration

    Returns:
        Updated state
    """
    from pipeworks.plugins.base import plugin_registry

    if enabled:
        # Instantiate plugin with new config
        logger.info(f"Enabling plugin: {plugin_name} with config: {plugin_config}")
        state.active_plugins[plugin_name] = plugin_registry.instantiate(
            plugin_name, **plugin_config
        )
    else:
        # Disable the plugin
        logger.info(f"Disabling plugin: {plugin_name}")
        if plugin_name in state.active_plugins:
            state.active_plugins[plugin_name].enabled = False

    # Update generator's plugin list
    state = update_generator_plugins(state)

    return state


def switch_model(state: UIState, model_name: str) -> UIState:
    """Switch to a different model adapter.

    This function unloads the current model and loads a new one.
    Plugins are preserved and attached to the new model adapter.

    Args:
        state: UI state
        model_name: Name of the model adapter to switch to

    Returns:
        Updated state with new model adapter

    Raises:
        Exception: If model switching fails
    """
    logger.info(f"Switching model from {state.current_model_name} to {model_name}")

    try:
        # Unload current model if loaded
        if state.model_adapter is not None:
            logger.info(f"Unloading current model: {state.current_model_name}")
            state.model_adapter.unload_model()

            # Aggressive CUDA memory cleanup
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cleared CUDA cache")

        # Get list of current plugins to transfer
        current_plugins = list(state.active_plugins.values())

        # Instantiate new model adapter with current plugins
        logger.info(f"Loading new model: {model_name}")
        state.model_adapter = model_registry.instantiate(
            model_name, config, plugins=current_plugins
        )

        # Load the model
        state.model_adapter.load_model()

        # Update state
        state.current_model_name = model_name
        state.generator = state.model_adapter  # Maintain backward compatibility

        logger.info(f"Successfully switched to model: {model_name}")
        return state

    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        # Attempt to restore previous model if switch failed
        logger.warning("Attempting to restore previous model...")
        try:
            state.model_adapter = model_registry.instantiate(
                state.current_model_name, config, plugins=[]
            )
            state.model_adapter.load_model()
            logger.info("Previous model restored")
        except Exception as restore_error:
            logger.error(f"Failed to restore previous model: {restore_error}")
        raise


def cleanup_ui_state(state: UIState) -> None:
    """Clean up UI state resources.

    This should be called when a session ends to free resources.

    Args:
        state: UI state to clean up
    """
    logger.info("Cleaning up UIState resources")

    try:
        # Unload model if model adapter exists
        if state.model_adapter is not None:
            try:
                state.model_adapter.unload_model()
                logger.info("Model unloaded successfully")
            except Exception as e:
                logger.error(f"Error unloading model: {e}")

        # Clear references
        state.model_adapter = None
        state.generator = None  # Also clear legacy reference
        state.tokenizer_analyzer = None
        state.prompt_builder = None
        state.gallery_browser = None
        state.favorites_db = None
        state.catalog_manager = None
        state.active_plugins.clear()

        logger.info("UIState cleanup complete")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)

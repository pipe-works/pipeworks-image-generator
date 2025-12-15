"""State management utilities for Pipeworks UI."""

import logging
from typing import Optional

from pipeworks.core.config import config
from pipeworks.core.pipeline import ImageGenerator
from pipeworks.core.tokenizer import TokenizerAnalyzer
from pipeworks.core.prompt_builder import PromptBuilder
from pipeworks.plugins.base import PluginBase

from .models import UIState

logger = logging.getLogger(__name__)


def initialize_ui_state(state: Optional[UIState] = None) -> UIState:
    """Initialize or ensure UI state is ready.

    This function handles lazy initialization of the UI state components.
    If state is None or uninitialized, it creates and loads all necessary
    components (generator, tokenizer, prompt builder).

    Args:
        state: Existing UIState or None

    Returns:
        Initialized UIState instance
    """
    # Create new state if None
    if state is None:
        logger.info("Creating new UIState")
        state = UIState()

    # Check if already initialized
    if state.is_initialized():
        logger.debug("UIState already initialized")
        return state

    logger.info("Initializing UIState components...")

    try:
        # Initialize generator
        if state.generator is None:
            logger.info("Initializing ImageGenerator")
            state.generator = ImageGenerator(config, plugins=[])
            # Pre-load model
            try:
                state.generator.load_model()
                logger.info("Model pre-loaded successfully")
            except Exception as e:
                logger.error(f"Failed to pre-load model: {e}")
                logger.warning("Model will be loaded on first generation attempt")

        # Initialize tokenizer analyzer
        if state.tokenizer_analyzer is None:
            logger.info("Initializing TokenizerAnalyzer")
            state.tokenizer_analyzer = TokenizerAnalyzer(
                model_id=config.model_id,
                cache_dir=config.models_dir
            )
            state.tokenizer_analyzer.load()
            logger.info("TokenizerAnalyzer loaded successfully")

        # Initialize prompt builder
        if state.prompt_builder is None:
            logger.info("Initializing PromptBuilder")
            state.prompt_builder = PromptBuilder(config.inputs_dir)
            logger.info("PromptBuilder initialized successfully")

        logger.info(f"UIState initialization complete: {state}")
        return state

    except Exception as e:
        logger.error(f"Error initializing UIState: {e}", exc_info=True)
        raise


def update_generator_plugins(state: UIState) -> UIState:
    """Update the generator's plugin list from active_plugins.

    Args:
        state: UI state containing generator and active_plugins

    Returns:
        Updated state
    """
    if state.generator is None:
        logger.warning("Cannot update plugins: generator not initialized")
        return state

    # Get list of enabled plugins
    enabled_plugins = [
        plugin for plugin in state.active_plugins.values()
        if plugin.enabled
    ]

    # Update generator's plugin list
    state.generator.plugins = enabled_plugins
    logger.info(f"Updated generator with {len(enabled_plugins)} active plugins")

    return state


def toggle_plugin(
    state: UIState,
    plugin_name: str,
    enabled: bool,
    **plugin_config
) -> UIState:
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
            plugin_name,
            **plugin_config
        )
    else:
        # Disable the plugin
        logger.info(f"Disabling plugin: {plugin_name}")
        if plugin_name in state.active_plugins:
            state.active_plugins[plugin_name].enabled = False

    # Update generator's plugin list
    state = update_generator_plugins(state)

    return state


def cleanup_ui_state(state: UIState) -> None:
    """Clean up UI state resources.

    This should be called when a session ends to free resources.

    Args:
        state: UI state to clean up
    """
    logger.info("Cleaning up UIState resources")

    try:
        # Unload model if generator exists
        if state.generator is not None:
            try:
                state.generator.unload_model()
                logger.info("Model unloaded successfully")
            except Exception as e:
                logger.error(f"Error unloading model: {e}")

        # Clear references
        state.generator = None
        state.tokenizer_analyzer = None
        state.prompt_builder = None
        state.active_plugins.clear()

        logger.info("UIState cleanup complete")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)

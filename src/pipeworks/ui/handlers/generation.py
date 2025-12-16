"""Image generation and plugin management handlers."""

import logging
import random

import gradio as gr

from pipeworks.core.config import config

from ..models import ASPECT_RATIOS, MAX_SEED, GenerationParams, SegmentConfig, UIState
from ..state import initialize_ui_state
from ..state import toggle_plugin as toggle_plugin_state
from ..validation import (
    ValidationError,
    validate_generation_params,
    validate_prompt_content,
    validate_segments,
)

logger = logging.getLogger(__name__)


def set_aspect_ratio(ratio_name: str) -> tuple[gr.Number, gr.Number]:
    """Set width and height based on aspect ratio preset.

    Args:
        ratio_name: Name of the aspect ratio preset

    Returns:
        Tuple of (width_update, height_update)
    """
    dimensions = ASPECT_RATIOS.get(ratio_name)

    if dimensions is None:  # Custom
        width, height = config.default_width, config.default_height
    else:
        width, height = dimensions

    return gr.update(value=width), gr.update(value=height)


def generate_image(
    prompt: str,
    width: int,
    height: int,
    num_steps: int,
    batch_size: int,
    runs: int,
    seed: int,
    use_random_seed: bool,
    start: SegmentConfig,
    middle: SegmentConfig,
    end: SegmentConfig,
    state: UIState,
) -> tuple[list[str], str, str, UIState]:
    """Generate image(s) from the UI inputs.

    Args:
        prompt: Text prompt (used if no dynamic segments)
        width: Image width
        height: Image height
        num_steps: Number of inference steps
        batch_size: Number of images per run
        runs: Number of runs to execute
        seed: Random seed
        use_random_seed: Whether to use a random seed
        start: Start segment configuration
        middle: Middle segment configuration
        end: End segment configuration
        state: UI state

    Returns:
        Tuple of (image_paths, info_text, seed_used, updated_state)
    """
    # Import here to avoid circular dependency
    from .prompt import build_combined_prompt

    try:
        # Initialize state
        state = initialize_ui_state(state)

        # Create GenerationParams and validate
        params = GenerationParams(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            batch_size=batch_size,
            runs=runs,
            seed=seed,
            use_random_seed=use_random_seed,
        )

        # Validate generation parameters
        validate_generation_params(params)

        # Check if any segment is dynamic
        segments = (start, middle, end)
        has_dynamic = any(seg.dynamic for seg in segments)

        # Validate segments
        validate_segments(segments, config.inputs_dir, prompt)

        generated_paths = []
        seeds_used = []
        prompts_used = []  # Track prompts if dynamic
        current_seed = seed

        # Loop through runs
        for run in range(runs):
            logger.info(f"Starting run {run+1}/{runs}")

            # Generate batch_size images for this run
            for i in range(batch_size):
                # Build prompt dynamically if any segment is dynamic
                if has_dynamic:
                    try:
                        current_prompt = build_combined_prompt(start, middle, end, state)
                        if current_prompt:
                            prompts_used.append(current_prompt)
                        else:
                            current_prompt = prompt  # Fall back to static prompt
                    except ValidationError as e:
                        # If prompt building fails, use static prompt
                        logger.warning(f"Dynamic prompt build failed, using static: {e}")
                        current_prompt = prompt
                else:
                    current_prompt = prompt

                # Validate final prompt
                if current_prompt:
                    validate_prompt_content(current_prompt)

                # Generate random seed if requested, or use sequential seed
                if use_random_seed:
                    actual_seed = random.randint(0, MAX_SEED)
                else:
                    actual_seed = current_seed
                    current_seed += 1

                # Generate and save image (plugins are called automatically)
                image, save_path = state.generator.generate_and_save(
                    prompt=current_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    seed=actual_seed,
                )

                generated_paths.append(str(save_path))
                seeds_used.append(actual_seed)

                image_num = run * batch_size + i + 1
                logger.info(f"Image {image_num}/{params.total_images} complete: {save_path}")

        # Create info text
        active_plugin_names = [p.name for p in state.generator.plugins if p.enabled]
        plugins_info = (
            f"\n**Active Plugins:** {', '.join(active_plugin_names)}" if active_plugin_names else ""
        )

        # Format seeds display
        if params.total_images == 1:
            seeds_display = str(seeds_used[0])
            paths_display = str(generated_paths[0])
        else:
            seeds_display = f"{seeds_used[0]} - {seeds_used[-1]}"
            paths_display = f"{len(generated_paths)} images saved to output folder"

        # Dynamic prompts info
        dynamic_info = ""
        if has_dynamic:
            dynamic_info = "\n**Dynamic Prompts:** Enabled (prompts rebuilt for each image)"
            if len(prompts_used) <= 3:
                # Show all prompts if 3 or fewer
                dynamic_info += f"\n**Sample Prompts:** {', '.join(prompts_used[:3])}"
            else:
                # Show first 2 prompts as samples
                dynamic_info += f"\n**Sample Prompts:** {prompts_used[0]}, {prompts_used[1]}, ..."

        info = f"""
✅ **Generation Complete!**

**Prompt:** {prompt if not has_dynamic else "(Dynamic)"}
**Dimensions:** {width}x{height}
**Steps:** {num_steps}
**Batch Size:** {batch_size} × **Runs:** {runs} = **Total:** {params.total_images} images
**Seeds:** {seeds_display}
**Saved to:** {paths_display}{dynamic_info}{plugins_info}
        """

        # Return all generated images for display in gallery
        return generated_paths, info.strip(), str(seeds_used[-1]), state

    except ValidationError as e:
        # User-friendly validation error
        logger.warning(f"Validation error: {e}")
        error_msg = f"❌ **Validation Error**\n\n{str(e)}"
        return [], error_msg, str(seed), state

    except Exception as e:
        # Unexpected error
        logger.error(f"Error generating image: {e}", exc_info=True)
        error_msg = (
            f"❌ **Error**\n\nAn unexpected error occurred. "
            f"Check logs for details.\n\n`{str(e)}`"
        )
        return [], error_msg, str(seed), state


def toggle_plugin_ui(
    plugin_name: str, enabled: bool, state: UIState, **plugin_config
) -> tuple[gr.Group, UIState]:
    """Toggle a plugin on/off and update its configuration.

    Args:
        plugin_name: Name of the plugin
        enabled: Whether to enable the plugin
        state: UI state
        **plugin_config: Plugin-specific configuration

    Returns:
        Tuple of (visibility_update, updated_state)
    """
    state = toggle_plugin_state(state, plugin_name, enabled, **plugin_config)
    return gr.update(visible=enabled), state


def toggle_save_metadata_handler(
    enabled: bool, folder: str, prefix: str, state: UIState
) -> tuple[gr.Group, UIState]:
    """Handle SaveMetadata plugin toggle with configuration.

    Args:
        enabled: Whether to enable the plugin
        folder: Metadata folder name
        prefix: Filename prefix
        state: UI state

    Returns:
        Tuple of (visibility_update, updated_state)
    """
    vis_update, new_state = toggle_plugin_ui(
        "SaveMetadata", enabled, state, folder_name=folder, filename_prefix=prefix
    )
    return vis_update, new_state


def update_plugin_config_handler(
    enabled: bool, folder: str, prefix: str, state: UIState
) -> UIState:
    """Update SaveMetadata plugin configuration when settings change.

    Args:
        enabled: Whether plugin is enabled
        folder: Metadata folder name
        prefix: Filename prefix
        state: UI state

    Returns:
        Updated UI state
    """
    if enabled:
        _, new_state = toggle_plugin_ui(
            "SaveMetadata", enabled, state, folder_name=folder, filename_prefix=prefix
        )
        return new_state
    return state

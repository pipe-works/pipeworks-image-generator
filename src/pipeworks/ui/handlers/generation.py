"""Image generation and plugin management handlers."""

import logging
import random

import gradio as gr

from pipeworks.core.config import config
from pipeworks.core.model_adapters import model_registry

from ..models import ASPECT_RATIOS, MAX_SEED, GenerationParams, SegmentConfig, UIState
from ..state import initialize_ui_state, switch_model
from ..state import toggle_plugin as toggle_plugin_state
from ..validation import (
    ValidationError,
    validate_generation_params,
    validate_prompt_content,
    validate_segments,
)

logger = logging.getLogger(__name__)


def switch_model_handler(
    model_name: str, state: UIState
) -> tuple[str, gr.update, gr.update, UIState]:
    """Handle model switching from the UI.

    Args:
        model_name: Name of the model adapter to switch to
        state: UI state

    Returns:
        Tuple of (status_message, image_edit_group_update, text_to_image_group_update, updated_state)
    """
    try:
        logger.info(f"UI requesting model switch to: {model_name}")

        # Check if already using this model
        if state.current_model_name == model_name:
            # Determine current model type for UI visibility
            is_image_edit = _is_image_edit_model(model_name)
            return (
                f"✅ Already using {model_name}",
                gr.update(visible=is_image_edit),
                gr.update(visible=not is_image_edit),
                state,
            )

        # Switch the model
        state = switch_model(state, model_name)

        # Determine model type for UI visibility
        is_image_edit = _is_image_edit_model(model_name)

        success_msg = f"✅ Successfully switched to **{model_name}**"
        if is_image_edit:
            success_msg += "\n\n📸 **Image editing mode** - Upload an image and provide editing instructions"
        else:
            success_msg += "\n\n✨ **Text-to-image mode** - Describe the image you want to generate"

        logger.info(f"Model switch successful: {model_name} (image_edit={is_image_edit})")

        return (
            success_msg,
            gr.update(visible=is_image_edit),  # Show image edit group for image-edit models
            gr.update(visible=not is_image_edit),  # Show text-to-image group for others
            state,
        )

    except Exception as e:
        logger.error(f"Failed to switch model: {e}", exc_info=True)
        error_msg = f"❌ Failed to switch model: {str(e)}"
        # Keep current UI state on error
        is_image_edit = _is_image_edit_model(state.current_model_name)
        return (
            error_msg,
            gr.update(visible=is_image_edit),
            gr.update(visible=not is_image_edit),
            state,
        )


def _is_image_edit_model(model_name: str) -> bool:
    """Check if a model is an image-edit type model.

    Args:
        model_name: Name of the model adapter

    Returns:
        True if model is image-edit type, False otherwise
    """
    # Check the model type from the registry
    try:
        adapter_class = model_registry.get_adapter_class(model_name)
        if adapter_class:
            return getattr(adapter_class, "model_type", "text-to-image") == "image-edit"
    except Exception as e:
        logger.warning(f"Could not determine model type for {model_name}: {e}")

    # Fallback: Check name patterns
    return "edit" in model_name.lower() or "qwen" in model_name.lower()


def get_available_models() -> list[str]:
    """Get list of available model adapters.

    Returns:
        List of model adapter names
    """
    return model_registry.list_available()


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
    input_image: str | None = None,
    instruction: str | None = None,
) -> tuple[list[str], str, str, UIState]:
    """Generate or edit image(s) from the UI inputs.

    Args:
        prompt: Text prompt (used for text-to-image if no dynamic segments)
        width: Image width (text-to-image only)
        height: Image height (text-to-image only)
        num_steps: Number of inference steps
        batch_size: Number of images per run
        runs: Number of runs to execute
        seed: Random seed
        use_random_seed: Whether to use a random seed
        start: Start segment configuration
        middle: Middle segment configuration
        end: End segment configuration
        state: UI state
        input_image: Input image path for image editing (optional)
        instruction: Editing instruction for image editing (optional)

    Returns:
        Tuple of (image_paths, info_text, seed_used, updated_state)
    """
    # Import here to avoid circular dependency
    from pathlib import Path
    from PIL import Image

    from .prompt import build_combined_prompt

    try:
        # Initialize state
        state = initialize_ui_state(state)

        # Determine if this is image editing or text-to-image
        is_image_edit = state.model_adapter.model_type == "image-edit"

        # Validate based on model type
        if is_image_edit:
            # Image editing workflow - require input_image
            if not input_image:
                error_msg = (
                    f"❌ **Missing Input Image**\n\n"
                    f"The model **{state.current_model_name}** requires an input image.\n\n"
                    f"Please upload an image above to edit."
                )
                return [], error_msg, str(seed), state

            if not instruction or not instruction.strip():
                error_msg = (
                    f"❌ **Missing Editing Instruction**\n\n"
                    f"Please provide an instruction describing how you want to edit the image."
                )
                return [], error_msg, str(seed), state
        else:
            # Text-to-image workflow - no input image needed
            pass

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
                        # Pass run index for Sequential mode support
                        current_prompt = build_combined_prompt(start, middle, end, state, run_index=run)
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

                # Validate final prompt/instruction
                if is_image_edit:
                    current_instruction = instruction
                    if current_instruction:
                        validate_prompt_content(current_instruction)
                else:
                    if current_prompt:
                        validate_prompt_content(current_prompt)

                # Generate random seed if requested, or use sequential seed
                if use_random_seed:
                    actual_seed = random.randint(0, MAX_SEED)
                else:
                    actual_seed = current_seed
                    current_seed += 1

                # Generate and save image based on model type
                if is_image_edit:
                    # Image editing workflow
                    input_img = Image.open(input_image)
                    image, save_path = state.generator.generate_and_save(
                        input_image=input_img,
                        instruction=current_instruction,
                        num_inference_steps=num_steps,
                        seed=actual_seed,
                    )
                else:
                    # Text-to-image workflow
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

        # Create info text based on model type
        if is_image_edit:
            info = f"""
✅ **Image Editing Complete!**

**Instruction:** {instruction}
**Input Image:** {Path(input_image).name}
**Steps:** {num_steps}
**Batch Size:** {batch_size} × **Runs:** {runs} = **Total:** {params.total_images} images
**Seeds:** {seeds_display}
**Saved to:** {paths_display}{plugins_info}
            """
        else:
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

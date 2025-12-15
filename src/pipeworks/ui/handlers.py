"""Event handlers for Gradio UI components."""

import logging
import random
from pathlib import Path

import gradio as gr

from pipeworks.core.config import config

from .models import ASPECT_RATIOS, MAX_SEED, GenerationParams, SegmentConfig, UIState
from .state import initialize_ui_state
from .state import toggle_plugin as toggle_plugin_state
from .validation import (
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


def analyze_prompt(prompt: str, state: UIState) -> tuple[str, UIState]:
    """Analyze prompt tokenization and return formatted results.

    Args:
        prompt: Text prompt to analyze
        state: UI state (contains tokenizer)

    Returns:
        Tuple of (formatted_markdown, updated_state)
    """
    if not prompt or prompt.strip() == "":
        return "*Enter a prompt to see tokenization analysis*", state

    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Analyze the prompt
        analysis = state.tokenizer_analyzer.analyze(prompt)

        # Format results
        token_count = analysis["token_count"]
        tokens = analysis["tokens"]
        formatted_tokens = state.tokenizer_analyzer.format_tokens(tokens)

        # Build markdown output
        result = f"""
**Token Count:** {token_count}

**Tokenized Output:**
```
{formatted_tokens}
```
"""

        if analysis["special_tokens"]:
            special = ", ".join(analysis["special_tokens"])
            result += f"\n**Special Tokens Found:** {special}\n"

        return result.strip(), state

    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}", exc_info=True)
        return f"*Error analyzing prompt: {str(e)}*", state


def get_items_in_path(current_path: str, state: UIState) -> tuple[gr.Dropdown, str, UIState]:
    """Get folders and files at the current path level.

    Args:
        current_path: Current path being browsed (empty string for root)
        state: UI state (contains prompt_builder)

    Returns:
        Tuple of (updated_dropdown, display_path, updated_state)
    """
    # Initialize state if needed
    state = initialize_ui_state(state)

    folders, files = state.prompt_builder.get_items_in_path(current_path)

    # Build choices list: folders first (with prefix), then files
    choices = ["(None)"]

    # Add ".." to go up a level if not at root
    if current_path:
        choices.append("📁 ..")

    # Add folders with folder emoji
    for folder in folders:
        choices.append(f"📁 {folder}")

    # Add files
    choices.extend(files)

    # Display path for user reference
    display_path = f"/{current_path}" if current_path else "/inputs"

    return gr.update(choices=choices, value="(None)"), display_path, state


def navigate_file_selection(
    selected: str, current_path: str, state: UIState
) -> tuple[gr.Dropdown, str, UIState]:
    """Handle folder navigation when an item is selected.

    Args:
        selected: The selected item from dropdown
        current_path: Current path being browsed
        state: UI state

    Returns:
        Tuple of (updated_dropdown, new_path, updated_state)
    """
    # If (None) selected, do nothing
    if selected == "(None)":
        dropdown, display, state = get_items_in_path(current_path, state)
        return dropdown, current_path, state

    # Check if it's a folder (starts with folder emoji)
    if selected.startswith("📁 "):
        folder_name = selected[2:].strip()  # Remove emoji and whitespace

        if folder_name == "..":
            # Go up one level
            if current_path:
                new_path = str(Path(current_path).parent)
                if new_path == ".":
                    new_path = ""
            else:
                new_path = ""
        else:
            # Navigate into folder
            new_path = str(Path(current_path) / folder_name) if current_path else folder_name

        # Update dropdown with new path contents
        dropdown, display, state = get_items_in_path(new_path, state)
        return dropdown, new_path, state
    else:
        # It's a file - keep it selected but don't navigate
        return gr.update(), current_path, state


def build_combined_prompt(
    start: SegmentConfig, middle: SegmentConfig, end: SegmentConfig, state: UIState
) -> str:
    """Build a combined prompt from multiple segments.

    Args:
        start: Start segment configuration
        middle: Middle segment configuration
        end: End segment configuration
        state: UI state (contains prompt_builder)

    Returns:
        Combined prompt string or error message
    """
    # Initialize state if needed
    state = initialize_ui_state(state)

    segments = []

    # Helper to add segment
    def add_segment(segment: SegmentConfig):
        # Add user text if provided
        if segment.text and segment.text.strip():
            segments.append(("text", segment.text.strip()))

        # Add file selection if configured
        if segment.is_configured():
            # Get full file path
            full_path = state.prompt_builder.get_full_path(segment.path, segment.file)

            if segment.mode == "Random Line":
                segments.append(("file_random", full_path))
            elif segment.mode == "Specific Line":
                segments.append(("file_specific", f"{full_path}|{segment.line}"))
            elif segment.mode == "Line Range":
                segments.append(("file_range", f"{full_path}|{segment.line}|{segment.range_end}"))
            elif segment.mode == "All Lines":
                segments.append(("file_all", full_path))
            elif segment.mode == "Random Multiple":
                segments.append(("file_random_multi", f"{full_path}|{segment.count}"))

    # Add segments in order
    add_segment(start)
    add_segment(middle)
    add_segment(end)

    # Build the final prompt
    try:
        result = state.prompt_builder.build_prompt(segments)
        return result if result else ""
    except Exception as e:
        logger.error(f"Error building prompt: {e}", exc_info=True)
        # Don't return error as prompt - raise it
        raise ValidationError(f"Failed to build prompt: {str(e)}")


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


# Gallery Browser Handlers


def load_gallery_folder(
    selected_item: str, current_path: str, state: UIState
) -> tuple[gr.Dropdown, str, list[str], UIState]:
    """Navigate folders or load images from current path.

    Args:
        selected_item: Item selected from dropdown (folder or image)
        current_path: Current path being browsed
        state: UI state

    Returns:
        Tuple of (dropdown_update, new_path, gallery_images, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Handle folder navigation
        if selected_item and selected_item.startswith("📁"):
            folder_name = selected_item[2:].strip()  # Remove emoji prefix

            # Skip if it's just a placeholder
            if not folder_name or folder_name in ["(No folders)", "(Error)", "- Select folder --"]:
                return gr.update(), current_path, state.gallery_images, state

            if folder_name == "..":
                # Go up one level
                if current_path:
                    new_path = str(Path(current_path).parent)
                    if new_path == ".":
                        new_path = ""
                else:
                    new_path = ""
            else:
                # Navigate into folder
                new_path = str(Path(current_path) / folder_name) if current_path else folder_name

            # Update current path in state
            state.gallery_current_path = new_path

            # Get items in new path
            folders, _ = state.gallery_browser.get_items_in_path(new_path)

            # Build dropdown choices
            choices = ["-- Select folder --"]  # Neutral first choice
            if new_path:  # Add parent navigation if not at root
                choices.append("📁 ..")

            # Add folders with emoji
            for folder in folders:
                choices.append(f"📁 {folder}")

            # Scan for images in new path
            images = state.gallery_browser.scan_images(new_path)
            state.gallery_images = images

            # Update dropdown with new choices and set to neutral selection
            logger.info(f"Navigated to: {new_path}, {len(images)} images, {len(folders)} folders")
            return gr.update(choices=choices, value="-- Select folder --"), new_path, images, state

        else:
            # Not a folder selection, just return current state
            return gr.update(), current_path, state.gallery_images, state

    except Exception as e:
        logger.error(f"Error loading gallery folder: {e}", exc_info=True)
        return gr.update(), current_path, state.gallery_images, state


def select_gallery_image(
    evt: gr.SelectData, show_json: str, state: UIState
) -> tuple[str, str, UIState]:
    """Display selected image with its metadata.

    Args:
        evt: Gradio SelectData event containing selected index
        show_json: Which metadata format to show ("Text (.txt)" or "JSON (.json)")
        state: UI state

    Returns:
        Tuple of (image_path, metadata_markdown, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Get selected index from event
        selected_index = evt.index

        # Check if we have images cached
        if not state.gallery_images or selected_index >= len(state.gallery_images):
            return None, "*No image selected*", state

        # Get image path
        image_path = state.gallery_images[selected_index]
        image_name = Path(image_path).name

        # Store selected index in state
        state.gallery_selected_index = selected_index

        # Read metadata based on toggle
        if "JSON" in show_json:
            json_data = state.gallery_browser.read_json_metadata(image_path)
            metadata_md = state.gallery_browser.format_metadata_json(json_data, image_name)
        else:
            txt_content = state.gallery_browser.read_txt_metadata(image_path)
            metadata_md = state.gallery_browser.format_metadata_txt(txt_content, image_name)

        return image_path, metadata_md, state

    except Exception as e:
        logger.error(f"Error selecting gallery image: {e}", exc_info=True)
        return None, f"*Error loading image: {str(e)}*", state


def refresh_gallery(current_path: str, state: UIState) -> tuple[list[str], UIState]:
    """Refresh image list in current path.

    Args:
        current_path: Current path being browsed
        state: UI state

    Returns:
        Tuple of (gallery_images, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Scan for images
        images = state.gallery_browser.scan_images(current_path)
        state.gallery_images = images

        return images, state

    except Exception as e:
        logger.error(f"Error refreshing gallery: {e}", exc_info=True)
        return [], state


def toggle_metadata_format(show_json: str, state: UIState) -> tuple[str, UIState]:
    """Switch between .txt and .json metadata view.

    Args:
        show_json: Which metadata format to show ("Text (.txt)" or "JSON (.json)")
        state: UI state

    Returns:
        Tuple of (metadata_markdown, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Check if we have a selected image
        if state.gallery_selected_index is None or not state.gallery_images:
            return "*Select an image to view metadata*", state

        if state.gallery_selected_index >= len(state.gallery_images):
            return "*No image selected*", state

        # Get selected image
        image_path = state.gallery_images[state.gallery_selected_index]
        image_name = Path(image_path).name

        # Read metadata based on toggle
        if "JSON" in show_json:
            json_data = state.gallery_browser.read_json_metadata(image_path)
            metadata_md = state.gallery_browser.format_metadata_json(json_data, image_name)
        else:
            txt_content = state.gallery_browser.read_txt_metadata(image_path)
            metadata_md = state.gallery_browser.format_metadata_txt(txt_content, image_name)

        return metadata_md, state

    except Exception as e:
        logger.error(f"Error toggling metadata format: {e}", exc_info=True)
        return f"*Error loading metadata: {str(e)}*", state


def initialize_gallery_browser(state: UIState) -> tuple[gr.Dropdown, str, list[str], UIState]:
    """Initialize gallery browser on tab load.

    Args:
        state: UI state

    Returns:
        Tuple of (dropdown_update, current_path, gallery_images, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Skip if already initialized to prevent infinite loop
        if state.gallery_initialized:
            return (
                gr.update(),
                state.gallery_current_path,
                state.gallery_images,
                state,
            )

        # Mark as initialized
        state.gallery_initialized = True

        # Start at root of outputs directory
        current_path = ""
        state.gallery_current_path = current_path

        # Get folders and images at root
        folders, _ = state.gallery_browser.get_items_in_path(current_path)

        # Build dropdown choices
        choices = ["-- Select folder --"]  # Neutral first choice
        for folder in folders:
            choices.append(f"📁 {folder}")

        # Scan for images at root
        images = state.gallery_browser.scan_images(current_path)
        state.gallery_images = images

        logger.info(f"Initialized gallery: {len(images)} images, {len(folders)} folders at root")
        return gr.update(choices=choices, value="-- Select folder --"), current_path, images, state

    except Exception as e:
        logger.error(f"Error initializing gallery browser: {e}", exc_info=True)
        return gr.update(choices=["(Error)"]), "", [], state

"""Gradio UI for Pipeworks Image Generator - Refactored Version."""

import logging
import random
from typing import List, Tuple
from pathlib import Path

import gradio as gr

from pipeworks.core.config import config
from pipeworks.plugins.base import plugin_registry
# Import all plugins to ensure they're registered
from pipeworks.plugins import SaveMetadataPlugin

# Import new refactored modules
from .models import (
    GenerationParams,
    SegmentConfig,
    UIState,
    ASPECT_RATIOS,
    MAX_SEED,
    DEFAULT_SEED
)
from .components import SegmentUI, update_mode_visibility, create_three_segments
from .validation import (
    ValidationError,
    validate_generation_params,
    validate_segments,
    validate_prompt_content
)
from .state import initialize_ui_state, toggle_plugin as toggle_plugin_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_aspect_ratio(ratio_name: str) -> Tuple[gr.Number, gr.Number]:
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


def analyze_prompt(prompt: str, state: UIState) -> Tuple[str, UIState]:
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


def get_items_in_path(current_path: str, state: UIState) -> Tuple[gr.Dropdown, str, UIState]:
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
    selected: str,
    current_path: str,
    state: UIState
) -> Tuple[gr.Dropdown, str, UIState]:
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
    start: SegmentConfig,
    middle: SegmentConfig,
    end: SegmentConfig,
    state: UIState
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
    state: UIState
) -> Tuple[List[str], str, str, UIState]:
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
            use_random_seed=use_random_seed
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
        active_plugin_names = [
            p.name for p in state.generator.plugins if p.enabled
        ]
        plugins_info = (
            f"\n**Active Plugins:** {', '.join(active_plugin_names)}"
            if active_plugin_names else ""
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
        error_msg = f"❌ **Error**\n\nAn unexpected error occurred. Check logs for details.\n\n`{str(e)}`"
        return [], error_msg, str(seed), state


def toggle_plugin_ui(
    plugin_name: str,
    enabled: bool,
    state: UIState,
    **plugin_config
) -> Tuple[gr.Group, UIState]:
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


def create_ui() -> gr.Blocks:
    """Create the Gradio UI with refactored architecture.

    Returns:
        Gradio Blocks app
    """
    # Custom CSS for plugin section
    custom_css = """
    .plugin-section {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 12px;
    }
    """

    app = gr.Blocks(title="Pipeworks Image Generator", css=custom_css)

    with app:
        # Session state - one instance per user
        ui_state = gr.State(UIState())

        gr.Markdown(
            """
            # Pipeworks Image Generator
            ### Programmatic image generation with Z-Image-Turbo
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                gr.Markdown("### Generation Settings")

                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    value="A serene mountain landscape at sunset with vibrant colors",
                )

                # Prompt Builder
                with gr.Accordion("Prompt Builder", open=False):
                    gr.Markdown(
                        "*Build prompts by combining text and random lines from files "
                        "in the `inputs/` directory*\n\n"
                        "*Click folders (📁) to navigate, select files to use*"
                    )

                    # Initialize file browser choices
                    initial_choices = ["(None)"]
                    try:
                        from pipeworks.core.prompt_builder import PromptBuilder
                        temp_pb = PromptBuilder(config.inputs_dir)
                        folders, files = temp_pb.get_items_in_path("")
                        for folder in folders:
                            initial_choices.append(f"📁 {folder}")
                        initial_choices.extend(files)
                    except Exception as e:
                        logger.error(f"Error initializing file browser: {e}")

                    # Create three segment components (eliminates code duplication!)
                    start_segment, middle_segment, end_segment = create_three_segments(
                        initial_choices
                    )

                    # Build Button
                    build_prompt_btn = gr.Button("Build Prompt", variant="secondary")

                # Tokenizer Analyzer
                with gr.Accordion("Tokenizer Analyzer", open=False):
                    tokenizer_output = gr.Markdown(
                        value="*Enter a prompt to see tokenization analysis*",
                        label="Analysis"
                    )

                # Aspect Ratio Preset Selector
                aspect_ratio_dropdown = gr.Dropdown(
                    label="Aspect Ratio Preset",
                    choices=list(ASPECT_RATIOS.keys()),
                    value="Custom",
                    info="Select a preset or choose Custom to manually adjust sliders",
                )

                with gr.Row():
                    width_slider = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=config.default_width,
                        label="Width",
                    )
                    height_slider = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=config.default_height,
                        label="Height",
                    )

                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=config.num_inference_steps,
                    label="Inference Steps",
                    info="Z-Image-Turbo works best with 9 steps",
                )

                with gr.Row():
                    batch_input = gr.Number(
                        label="Batch Size",
                        value=1,
                        precision=0,
                        minimum=1,
                        maximum=100,
                        info="Images per run",
                    )
                    runs_input = gr.Number(
                        label="Runs",
                        value=1,
                        precision=0,
                        minimum=1,
                        maximum=100,
                        info="Number of runs to execute",
                    )
                    seed_input = gr.Number(
                        label="Seed",
                        value=DEFAULT_SEED,
                        precision=0,
                        minimum=0,
                        maximum=MAX_SEED,
                    )
                    random_seed_checkbox = gr.Checkbox(
                        label="Random Seed",
                        value=False,
                        info="Generate a new random seed each time",
                    )

                # Plugins Section
                with gr.Accordion("Plugins", open=True):
                    with gr.Group(elem_classes="plugin-section"):
                        # Save Metadata Plugin
                        with gr.Group():
                            save_metadata_check = gr.Checkbox(
                                label="Save Metadata (.txt + .json)",
                                value=False,
                                info="Save prompt and generation parameters to files",
                            )

                            with gr.Group(visible=False) as metadata_settings:
                                metadata_folder = gr.Textbox(
                                    label="Metadata Subfolder",
                                    value="metadata",
                                    placeholder="Leave empty to save alongside images",
                                    info="Subfolder within outputs directory",
                                )
                                metadata_prefix = gr.Textbox(
                                    label="Filename Prefix",
                                    value="",
                                    placeholder="Optional prefix for metadata files",
                                )

                            # Plugin toggle handler
                            def toggle_save_metadata(enabled, folder, prefix, state):
                                vis_update, new_state = toggle_plugin_ui(
                                    "SaveMetadata",
                                    enabled,
                                    state,
                                    folder_name=folder,
                                    filename_prefix=prefix
                                )
                                return vis_update, new_state

                            save_metadata_check.change(
                                fn=toggle_save_metadata,
                                inputs=[save_metadata_check, metadata_folder, metadata_prefix, ui_state],
                                outputs=[metadata_settings, ui_state],
                            )

                            # Update plugin config when settings change
                            def update_plugin_config(enabled, folder, prefix, state):
                                if enabled:
                                    _, new_state = toggle_plugin_ui(
                                        "SaveMetadata",
                                        enabled,
                                        state,
                                        folder_name=folder,
                                        filename_prefix=prefix
                                    )
                                    return new_state
                                return state

                            metadata_folder.change(
                                fn=update_plugin_config,
                                inputs=[save_metadata_check, metadata_folder, metadata_prefix, ui_state],
                                outputs=[ui_state],
                            )
                            metadata_prefix.change(
                                fn=update_plugin_config,
                                inputs=[save_metadata_check, metadata_folder, metadata_prefix, ui_state],
                                outputs=[ui_state],
                            )

                generate_btn = gr.Button(
                    "Generate Image",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                # Output display
                gr.Markdown("### Generated Images")

                image_output = gr.Gallery(
                    label="Output",
                    type="filepath",
                    height=600,
                    columns=2,
                    rows=2,
                    object_fit="contain",
                )

                # Show the seed that was actually used
                seed_used = gr.Textbox(
                    label="Seed Used",
                    interactive=False,
                    value=str(DEFAULT_SEED),
                )

                # Info display
                info_output = gr.Markdown(
                    label="Generation Info",
                    value="*Ready to generate images*",
                )

        # Model info footer
        gr.Markdown(
            f"""
            ---
            **Model:** {config.model_id} | **Device:** {config.device} | **Dtype:** {config.torch_dtype}

            *Outputs saved to: {config.outputs_dir}*
            """
        )

        # Event handlers
        # File browser navigation for all three segments
        for segment in [start_segment, middle_segment, end_segment]:
            file_dropdown, path_state, path_display = segment.get_navigation_components()

            file_dropdown.change(
                fn=navigate_file_selection,
                inputs=[file_dropdown, path_state, ui_state],
                outputs=[file_dropdown, path_state, ui_state],
            ).then(
                fn=lambda path: f"/{path}" if path else "/inputs",
                inputs=[path_state],
                outputs=[path_display],
            )

        # Mode visibility handlers for all three segments
        for segment in [start_segment, middle_segment, end_segment]:
            mode_dropdown = segment.mode
            line, range_end, count = segment.get_mode_visibility_outputs()

            mode_dropdown.change(
                fn=update_mode_visibility,
                inputs=[mode_dropdown],
                outputs=[line, range_end, count],
            )

        # Build prompt button handler
        def build_and_update_prompt(*values):
            """Build prompt from segment values and update UI."""
            # Split values into segment groups (8 values each)
            start_values = values[0:8]
            middle_values = values[8:16]
            end_values = values[16:24]
            state = values[24]

            # Convert to SegmentConfig objects
            start_cfg = SegmentUI.values_to_config(*start_values)
            middle_cfg = SegmentUI.values_to_config(*middle_values)
            end_cfg = SegmentUI.values_to_config(*end_values)

            # Initialize state
            state = initialize_ui_state(state)

            # Build prompt
            try:
                prompt = build_combined_prompt(start_cfg, middle_cfg, end_cfg, state)
            except ValidationError as e:
                prompt = f"Error: {str(e)}"

            # Update titles with status
            start_title = SegmentUI.format_title("Start", start_cfg.file, start_cfg.mode, start_cfg.dynamic)
            middle_title = SegmentUI.format_title("Middle", middle_cfg.file, middle_cfg.mode, middle_cfg.dynamic)
            end_title = SegmentUI.format_title("End", end_cfg.file, end_cfg.mode, end_cfg.dynamic)

            return prompt, start_title, middle_title, end_title, state

        # Collect all segment inputs
        all_segment_inputs = (
            start_segment.get_input_components() +
            middle_segment.get_input_components() +
            end_segment.get_input_components() +
            [ui_state]
        )

        build_prompt_btn.click(
            fn=build_and_update_prompt,
            inputs=all_segment_inputs,
            outputs=[
                prompt_input,
                start_segment.title,
                middle_segment.title,
                end_segment.title,
                ui_state
            ],
        )

        # Aspect ratio preset handler
        aspect_ratio_dropdown.change(
            fn=set_aspect_ratio,
            inputs=[aspect_ratio_dropdown],
            outputs=[width_slider, height_slider],
        )

        # Tokenizer analyzer handler
        prompt_input.change(
            fn=analyze_prompt,
            inputs=[prompt_input, ui_state],
            outputs=[tokenizer_output, ui_state],
        )

        # Trigger analysis on app load
        app.load(
            fn=analyze_prompt,
            inputs=[prompt_input, ui_state],
            outputs=[tokenizer_output, ui_state],
        )

        # Generate button handler
        def generate_wrapper(*values):
            """Wrapper to convert segment values to SegmentConfig objects."""
            # Extract values
            prompt = values[0]
            width = values[1]
            height = values[2]
            num_steps = values[3]
            batch_size = values[4]
            runs = values[5]
            seed = values[6]
            use_random_seed = values[7]

            # Segment values (8 values each)
            start_values = values[8:16]
            middle_values = values[16:24]
            end_values = values[24:32]
            state = values[32]

            # Convert to SegmentConfig objects
            start_cfg = SegmentUI.values_to_config(*start_values)
            middle_cfg = SegmentUI.values_to_config(*middle_values)
            end_cfg = SegmentUI.values_to_config(*end_values)

            # Call generate_image with clean parameters
            return generate_image(
                prompt, width, height, num_steps, batch_size, runs, seed, use_random_seed,
                start_cfg, middle_cfg, end_cfg, state
            )

        # Collect all inputs for generation
        generation_inputs = [
            prompt_input,
            width_slider,
            height_slider,
            steps_slider,
            batch_input,
            runs_input,
            seed_input,
            random_seed_checkbox,
        ] + all_segment_inputs

        generate_btn.click(
            fn=generate_wrapper,
            inputs=generation_inputs,
            outputs=[image_output, info_output, seed_used, ui_state],
        )

    return app


def main():
    """Main entry point for the application."""
    logger.info("Starting Pipeworks Image Generator (Refactored)...")
    logger.info(f"Configuration: {config.model_dump()}")

    # Log available plugins
    available_plugins = plugin_registry.list_available()
    logger.info(f"Available plugins: {available_plugins}")

    # Create and launch UI
    app = create_ui()

    logger.info(
        f"Launching Gradio UI on {config.gradio_server_name}:{config.gradio_server_port}"
    )

    app.launch(
        server_name=config.gradio_server_name,
        server_port=config.gradio_server_port,
        share=config.gradio_share,
        show_error=True,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()

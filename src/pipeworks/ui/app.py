"""Gradio UI for Pipeworks Image Generator - Refactored Version."""

import logging

import gradio as gr

from pipeworks.core.config import config
from pipeworks.plugins.base import plugin_registry

from .components import SegmentUI, create_three_segments, update_mode_visibility
from .handlers import (
    analyze_prompt,
    apply_gallery_filter,
    build_combined_prompt,
    generate_image,
    get_available_models,
    initialize_gallery_browser,
    load_gallery_folder,
    move_favorites_to_catalog,
    navigate_file_selection,
    refresh_gallery,
    select_gallery_image,
    set_aspect_ratio,
    switch_gallery_root,
    switch_model_handler,
    toggle_favorite,
    toggle_metadata_format,
    toggle_save_metadata_handler,
    update_plugin_config_handler,
)
from .models import ASPECT_RATIOS, DEFAULT_SEED, MAX_SEED, UIState
from .state import initialize_ui_state
from .validation import ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_ui() -> tuple[gr.Blocks, str]:
    """Create the Gradio UI with refactored architecture.

    Returns:
        Tuple of (Gradio Blocks app, custom CSS string)
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

    app = gr.Blocks(title="Pipeworks Image Generator")

    with app:
        # Session state - one instance per user
        ui_state = gr.State(UIState())

        gr.Markdown(
            """
            # Pipeworks Image Generator
            ### Multi-model AI image generation and editing
            """
        )

        with gr.Tabs():
            with gr.Tab("Generate", id="generate_tab"):
                # Main generation UI
                create_generation_tab(ui_state)

            with gr.Tab("Gallery Browser", id="gallery_tab") as gallery_tab:
                # Gallery browser UI
                gallery_components = create_gallery_tab(ui_state)

                # Initialize gallery when tab is selected
                gallery_tab.select(
                    fn=initialize_gallery_browser,
                    inputs=[ui_state],
                    outputs=[
                        gallery_components["folder_dropdown"],
                        gallery_components["current_path_state"],
                        gallery_components["gallery"],
                        ui_state,
                    ],
                )

    return app, custom_css


def create_generation_tab(ui_state):
    """Create the main generation tab UI.

    Args:
        ui_state: UI state component
    """
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            gr.Markdown("### Generation Settings")

            # Model Selection
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=get_available_models(),
                value=config.default_model_adapter,
                info="Select AI model for generation",
            )
            model_status = gr.Markdown(
                value=f"✅ **Current:** {config.default_model_adapter}"
            )

            # Image Editing Section (visible only for image-edit models like Qwen)
            with gr.Group(visible=False) as image_edit_group:
                gr.Markdown("### Image Editing")
                input_image = gr.Image(
                    label="Input Image",
                    type="filepath",
                    sources=["upload", "clipboard"],
                    height=300,
                )
                instruction_input = gr.Textbox(
                    label="Editing Instruction",
                    placeholder="Describe what changes you want to make to the image...",
                    lines=3,
                    value="change the sky to a sunset with vibrant orange and pink colors",
                    info="Natural language instruction for editing the image",
                )

            # Text-to-Image Section (visible only for text-to-image models like Z-Image-Turbo)
            with gr.Group(visible=True) as text_to_image_group:
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
                start_segment, middle_segment, end_segment = create_three_segments(initial_choices)

                # Build Button
                build_prompt_btn = gr.Button("Build Prompt", variant="secondary")

            # Tokenizer Analyzer
            with gr.Accordion("Tokenizer Analyzer", open=False):
                tokenizer_output = gr.Markdown(
                    value="*Enter a prompt to see tokenization analysis*", label="Analysis"
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
                        save_metadata_check.change(
                            fn=toggle_save_metadata_handler,
                            inputs=[
                                save_metadata_check,
                                metadata_folder,
                                metadata_prefix,
                                ui_state,
                            ],
                            outputs=[metadata_settings, ui_state],
                        )

                        # Update plugin config when settings change
                        metadata_folder.change(
                            fn=update_plugin_config_handler,
                            inputs=[
                                save_metadata_check,
                                metadata_folder,
                                metadata_prefix,
                                ui_state,
                            ],
                            outputs=[ui_state],
                        )
                        metadata_prefix.change(
                            fn=update_plugin_config_handler,
                            inputs=[
                                save_metadata_check,
                                metadata_folder,
                                metadata_prefix,
                                ui_state,
                            ],
                            outputs=[ui_state],
                        )

            # Model selection event handler (updates UI visibility based on model type)
            model_dropdown.change(
                fn=switch_model_handler,
                inputs=[model_dropdown, ui_state],
                outputs=[
                    model_status,
                    image_edit_group,
                    text_to_image_group,
                    ui_state,
                ],
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
            **Model:** {config.model_id} | **Device:** {config.device} |
            **Dtype:** {config.torch_dtype}

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
            line, range_end, count, sequential_start_line = segment.get_mode_visibility_outputs()

            mode_dropdown.change(
                fn=update_mode_visibility,
                inputs=[mode_dropdown],
                outputs=[line, range_end, count, sequential_start_line],
            )

        # Build prompt button handler
        def build_and_update_prompt(*values):
            """Build prompt from segment values and update UI."""
            # Split values into segment groups (9 values each: text, path, file, mode, line, range_end, count, dynamic, sequential_start_line)
            start_values = values[0:9]
            middle_values = values[9:18]
            end_values = values[18:27]
            state = values[27]

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
            start_title = SegmentUI.format_title(
                "Start", start_cfg.file, start_cfg.mode, start_cfg.dynamic
            )
            middle_title = SegmentUI.format_title(
                "Middle", middle_cfg.file, middle_cfg.mode, middle_cfg.dynamic
            )
            end_title = SegmentUI.format_title("End", end_cfg.file, end_cfg.mode, end_cfg.dynamic)

            return prompt, start_title, middle_title, end_title, state

        # Collect all segment inputs
        all_segment_inputs = (
            start_segment.get_input_components()
            + middle_segment.get_input_components()
            + end_segment.get_input_components()
            + [ui_state]
        )

        build_prompt_btn.click(
            fn=build_and_update_prompt,
            inputs=all_segment_inputs,
            outputs=[
                prompt_input,
                start_segment.title,
                middle_segment.title,
                end_segment.title,
                ui_state,
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

        # Generate button handler
        def generate_wrapper(*values):
            """Wrapper to convert segment values to SegmentConfig objects."""
            # Extract values - now includes image editing inputs
            input_img = values[0]
            instruction = values[1]
            prompt = values[2]
            width = values[3]
            height = values[4]
            num_steps = values[5]
            batch_size = values[6]
            runs = values[7]
            seed = values[8]
            use_random_seed = values[9]

            # Segment values (9 values each: text, path, file, mode, line, range_end, count, dynamic, sequential_start_line)
            start_values = values[10:19]
            middle_values = values[19:28]
            end_values = values[28:37]
            state = values[37]

            # Convert to SegmentConfig objects
            start_cfg = SegmentUI.values_to_config(*start_values)
            middle_cfg = SegmentUI.values_to_config(*middle_values)
            end_cfg = SegmentUI.values_to_config(*end_values)

            # Call generate_image with clean parameters (includes image editing params)
            return generate_image(
                prompt,
                width,
                height,
                num_steps,
                batch_size,
                runs,
                seed,
                use_random_seed,
                start_cfg,
                middle_cfg,
                end_cfg,
                state,
                input_image=input_img,
                instruction=instruction,
            )

        # Collect all inputs for generation (includes image editing inputs)
        generation_inputs = [
            input_image,
            instruction_input,
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


def create_gallery_tab(ui_state):
    """Create the gallery browser tab UI.

    Args:
        ui_state: UI state component

    Returns:
        Dictionary of gallery components for event handling
    """
    gr.Markdown("### Browse Generated Images")

    with gr.Row():
        root_selector = gr.Dropdown(
            label="Browse",
            choices=["📁 outputs", "📁 catalog"],
            value="📁 outputs",
            scale=1,
        )
        folder_dropdown = gr.Dropdown(
            label="Browse Folders",
            choices=["(No folders)"],
            value="(No folders)",
            scale=3,
        )
        refresh_btn = gr.Button("Refresh", size="sm", scale=1)

    with gr.Row():
        filter_dropdown = gr.Dropdown(
            label="Filter",
            choices=["All Images", "Favorites Only"],
            value="All Images",
            scale=2,
        )

    # Hidden state for current path
    current_path_state = gr.State("")

    current_path_display = gr.Markdown("**Current:** /outputs")

    with gr.Row():
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Images",
                columns=4,
                height=600,
                object_fit="cover",
                type="filepath",
                show_label=True,
            )

        with gr.Column(scale=1):
            selected_image = gr.Image(
                label="Selected Image",
                type="filepath",
                height=400,
                show_label=True,
            )

            with gr.Row():
                favorite_btn = gr.Button("☆ Favorite", size="sm", scale=1)
                move_catalog_btn = gr.Button(
                    "Move Favorites to Catalog", size="sm", scale=2, variant="primary"
                )

            catalog_info = gr.Markdown(value="", visible=True)

            metadata_toggle = gr.Radio(
                choices=["Text (.txt)", "JSON (.json)"],
                value="Text (.txt)",
                label="Metadata Format",
            )

            metadata_display = gr.Markdown(
                value="*Select an image to view metadata*",
                label="Metadata",
            )

    # Event handlers for gallery browser

    # Root selector
    root_selector.change(
        fn=switch_gallery_root,
        inputs=[root_selector, ui_state],
        outputs=[folder_dropdown, current_path_state, gallery, ui_state],
    ).then(
        fn=lambda root, path: (
            f"**Current:** /{root.replace('📁 ', '')}/{path}"
            if path
            else f"**Current:** /{root.replace('📁 ', '')}"
        ),
        inputs=[root_selector, current_path_state],
        outputs=[current_path_display],
    )

    # Folder navigation
    folder_dropdown.change(
        fn=load_gallery_folder,
        inputs=[folder_dropdown, current_path_state, ui_state],
        outputs=[folder_dropdown, current_path_state, gallery, ui_state],
    ).then(
        fn=lambda root, path: (
            f"**Current:** /{root.replace('📁 ', '')}/{path}"
            if path
            else f"**Current:** /{root.replace('📁 ', '')}"
        ),
        inputs=[root_selector, current_path_state],
        outputs=[current_path_display],
    )

    # Filter dropdown
    filter_dropdown.change(
        fn=apply_gallery_filter,
        inputs=[filter_dropdown, current_path_state, ui_state],
        outputs=[gallery, ui_state],
    )

    # Image selection - uses gr.SelectData for event
    gallery.select(
        fn=select_gallery_image,
        inputs=[metadata_toggle, ui_state],
        outputs=[selected_image, metadata_display, favorite_btn, ui_state],
    )

    # Favorite button
    favorite_btn.click(
        fn=toggle_favorite,
        inputs=[ui_state],
        outputs=[favorite_btn, catalog_info, ui_state],
    )

    # Move to catalog button
    move_catalog_btn.click(
        fn=move_favorites_to_catalog,
        inputs=[ui_state],
        outputs=[catalog_info, gallery, ui_state],
    )

    # Metadata format toggle
    metadata_toggle.change(
        fn=toggle_metadata_format,
        inputs=[metadata_toggle, ui_state],
        outputs=[metadata_display, ui_state],
    )

    # Refresh button
    refresh_btn.click(
        fn=refresh_gallery,
        inputs=[current_path_state, ui_state],
        outputs=[gallery, ui_state],
    )

    # Return components for initialization event handler
    return {
        "folder_dropdown": folder_dropdown,
        "current_path_state": current_path_state,
        "gallery": gallery,
    }


def main():
    """Main entry point for the application."""
    logger.info("Starting Pipeworks Image Generator (Refactored)...")
    logger.info(f"Configuration: {config.model_dump()}")

    # Log available plugins
    available_plugins = plugin_registry.list_available()
    logger.info(f"Available plugins: {available_plugins}")

    # Create and launch UI
    app, custom_css = create_ui()

    logger.info(f"Launching Gradio UI on {config.gradio_server_name}:{config.gradio_server_port}")

    app.launch(
        server_name=config.gradio_server_name,
        server_port=config.gradio_server_port,
        share=config.gradio_share,
        show_error=True,
        inbrowser=False,
        css=custom_css,
    )


if __name__ == "__main__":
    main()

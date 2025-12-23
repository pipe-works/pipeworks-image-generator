"""Gradio UI for Pipeworks Image Generator - Refactored Version."""

import logging

import gradio as gr

from pipeworks.core.config import config
from pipeworks.plugins.base import plugin_registry

from .components import (
    ConditionSegmentUI,
    SegmentUI,
    update_mode_visibility,
)
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
    # Image output at the very top (full width)
    gr.Markdown("### Generated Images")

    image_output = gr.Gallery(
        label="Output",
        type="filepath",
        height=400,
        columns=4,
        rows=1,
        object_fit="contain",
    )

    # Show the seed that was actually used and info
    with gr.Row():
        with gr.Column(scale=1):
            seed_used = gr.Textbox(
                label="Seed Used",
                interactive=False,
                value=str(DEFAULT_SEED),
            )
        with gr.Column(scale=2):
            # Info display
            info_output = gr.Markdown(
                value="*Ready to generate images*",
            )

    # Main content area
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
            model_status = gr.Markdown(value=f"✅ **Current:** {config.default_model_adapter}")

            # Image Editing Section (visible only for image-edit models like Qwen)
            with gr.Group(visible=False) as image_edit_group:
                gr.Markdown(
                    "### Image Editing\n"
                    "*Upload 1-3 images to edit or composite. "
                    "Examples: character + accessory, person + scene*"
                )
                with gr.Row():
                    input_image_1 = gr.Image(
                        label="Image 1 (Required)",
                        type="filepath",
                        sources=["upload", "clipboard"],
                        height=250,
                    )
                    input_image_2 = gr.Image(
                        label="Image 2 (Optional)",
                        type="filepath",
                        sources=["upload", "clipboard"],
                        height=250,
                    )
                    input_image_3 = gr.Image(
                        label="Image 3 (Optional)",
                        type="filepath",
                        sources=["upload", "clipboard"],
                        height=250,
                    )
                instruction_input = gr.Textbox(
                    label="Editing Instruction",
                    placeholder=(
                        "Describe the composition or changes "
                        "(e.g., 'character is wearing the hat', "
                        "'person in front of the background')..."
                    ),
                    lines=3,
                    value="the character is wearing the hat",
                    info="Natural language instruction for editing/compositing the images",
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

                # Row 1: Start segments
                gr.Markdown("**Start Segments**")
                with gr.Row():
                    # Start 1 is a regular segment (no conditions)
                    start_1 = SegmentUI("Start 1", initial_choices)
                    # Start 2 and Start 3 have condition generation support
                    start_2 = ConditionSegmentUI("Start 2", initial_choices)
                    start_3 = ConditionSegmentUI("Start 3", initial_choices)

                # Row 2: Mid segments
                gr.Markdown("**Mid Segments**")
                with gr.Row():
                    mid_1 = SegmentUI("Mid 1", initial_choices)
                    mid_2 = SegmentUI("Mid 2", initial_choices)
                    mid_3 = SegmentUI("Mid 3", initial_choices)

                # Row 3: End segments
                gr.Markdown("**End Segments**")
                with gr.Row():
                    end_1 = SegmentUI("End 1", initial_choices)
                    end_2 = SegmentUI("End 2", initial_choices)
                    end_3 = SegmentUI("End 3", initial_choices)

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
    # File browser navigation for all nine segments
    for segment in [start_1, start_2, start_3, mid_1, mid_2, mid_3, end_1, end_2, end_3]:
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

    # Mode visibility handlers for all nine segments
    for segment in [start_1, start_2, start_3, mid_1, mid_2, mid_3, end_1, end_2, end_3]:
        mode_dropdown = segment.mode
        line, range_end, count, sequential_start_line = segment.get_mode_visibility_outputs()

        mode_dropdown.change(
            fn=update_mode_visibility,
            inputs=[mode_dropdown],
            outputs=[line, range_end, count, sequential_start_line],
        )

    # =========================================================================
    # Condition generation handlers (Start 2 and Start 3)
    # =========================================================================

    def toggle_condition_type_handler(condition_type: str) -> tuple[str, gr.update]:
        """Show/hide condition controls and generate initial condition.

        Uses random seed by default for variety.

        Args:
            condition_type: Type of condition ("None", "Character", "Facial", "Both")

        Returns:
            Tuple of (condition_text, controls_visibility_update)
        """
        from pipeworks.ui.handlers import generate_condition_by_type

        if condition_type == "None":
            # Hide the condition controls and clear text
            return "", gr.update(visible=False)
        else:
            # Generate condition using random seed (None = system entropy)
            condition_text = generate_condition_by_type(condition_type, seed=None)

            # Show the condition controls
            return condition_text, gr.update(visible=True)

    def regenerate_condition_type_handler(condition_type: str) -> str:
        """Generate a new random condition when regenerate button is clicked.

        Args:
            condition_type: Type of condition to generate

        Returns:
            New condition text
        """
        from pipeworks.ui.handlers import generate_condition_by_type

        # Always use random seed for variety
        return generate_condition_by_type(condition_type, seed=None)

    # Wire up condition generation for Start 2
    (
        condition_type_2,
        condition_text_2,
        condition_regenerate_2,
        condition_dynamic_2,
        condition_controls_2,
    ) = start_2.get_condition_components()

    # Dropdown change: show/hide controls and generate initial condition
    condition_type_2.change(
        fn=toggle_condition_type_handler,
        inputs=[condition_type_2],
        outputs=[condition_text_2, condition_controls_2],
    )

    # Regenerate button: create new random condition
    condition_regenerate_2.click(
        fn=regenerate_condition_type_handler,
        inputs=[condition_type_2],
        outputs=[condition_text_2],
    )

    # Wire up condition generation for Start 3
    (
        condition_type_3,
        condition_text_3,
        condition_regenerate_3,
        condition_dynamic_3,
        condition_controls_3,
    ) = start_3.get_condition_components()

    # Dropdown change: show/hide controls and generate initial condition
    condition_type_3.change(
        fn=toggle_condition_type_handler,
        inputs=[condition_type_3],
        outputs=[condition_text_3, condition_controls_3],
    )

    # Regenerate button: create new random condition
    condition_regenerate_3.click(
        fn=regenerate_condition_type_handler,
        inputs=[condition_type_3],
        outputs=[condition_text_3],
    )

    # Build prompt button handler
    def build_and_update_prompt(*values):
        """Build prompt from segment values and update UI.

        Now includes condition text concatenation for Start 2 and Start 3.
        """
        # Split values into segment groups
        # (9 values each: text, path, file, mode, line, range_end, count, dynamic,
        # sequential_start_line)
        start_1_values = list(values[0:9])

        # Start 2 with conditions (3 condition values + 9 segment values)
        condition_type_2_val = values[9]
        condition_text_2_val = values[10]
        # Skip condition_dynamic_2 (index 11) - only used during generation, not preview
        start_2_values = list(values[12:21])

        # Start 3 with conditions (3 condition values + 9 segment values)
        condition_type_3_val = values[21]
        condition_text_3_val = values[22]
        # Skip condition_dynamic_3 (index 23) - only used during generation, not preview
        start_3_values = list(values[24:33])

        # Remaining segments
        mid_1_values = values[33:42]
        mid_2_values = values[42:51]
        mid_3_values = values[51:60]
        end_1_values = values[60:69]
        end_2_values = values[69:78]
        end_3_values = values[78:87]
        state = values[87]

        # Concatenate condition text with Start 2 text if condition is enabled
        # Show the condition in the prompt preview (even if dynamic - it will be
        # regenerated during generation)
        if condition_type_2_val != "None" and condition_text_2_val:
            # Condition text comes first, then user text (if any)
            original_text = start_2_values[0]  # First value is text
            if original_text and original_text.strip():
                start_2_values[0] = f"{condition_text_2_val}, {original_text}"
            else:
                start_2_values[0] = condition_text_2_val

        # Concatenate condition text with Start 3 text if condition is enabled
        if condition_type_3_val != "None" and condition_text_3_val:
            # Condition text comes first, then user text (if any)
            original_text = start_3_values[0]  # First value is text
            if original_text and original_text.strip():
                start_3_values[0] = f"{condition_text_3_val}, {original_text}"
            else:
                start_3_values[0] = condition_text_3_val

        # Convert to SegmentConfig objects
        start_1_cfg = SegmentUI.values_to_config(*start_1_values)
        start_2_cfg = SegmentUI.values_to_config(*start_2_values)
        start_3_cfg = SegmentUI.values_to_config(*start_3_values)
        mid_1_cfg = SegmentUI.values_to_config(*mid_1_values)
        mid_2_cfg = SegmentUI.values_to_config(*mid_2_values)
        mid_3_cfg = SegmentUI.values_to_config(*mid_3_values)
        end_1_cfg = SegmentUI.values_to_config(*end_1_values)
        end_2_cfg = SegmentUI.values_to_config(*end_2_values)
        end_3_cfg = SegmentUI.values_to_config(*end_3_values)

        # Initialize state
        state = initialize_ui_state(state)

        # Build prompt
        try:
            prompt = build_combined_prompt(
                start_1_cfg,
                start_2_cfg,
                start_3_cfg,
                mid_1_cfg,
                mid_2_cfg,
                mid_3_cfg,
                end_1_cfg,
                end_2_cfg,
                end_3_cfg,
                state,
            )
        except ValidationError as e:
            prompt = f"Error: {str(e)}"

        # Update titles with status
        start_1_title = SegmentUI.format_title(
            "Start 1", start_1_cfg.file, start_1_cfg.mode, start_1_cfg.dynamic
        )
        start_2_title = SegmentUI.format_title(
            "Start 2", start_2_cfg.file, start_2_cfg.mode, start_2_cfg.dynamic
        )
        start_3_title = SegmentUI.format_title(
            "Start 3", start_3_cfg.file, start_3_cfg.mode, start_3_cfg.dynamic
        )
        mid_1_title = SegmentUI.format_title(
            "Mid 1", mid_1_cfg.file, mid_1_cfg.mode, mid_1_cfg.dynamic
        )
        mid_2_title = SegmentUI.format_title(
            "Mid 2", mid_2_cfg.file, mid_2_cfg.mode, mid_2_cfg.dynamic
        )
        mid_3_title = SegmentUI.format_title(
            "Mid 3", mid_3_cfg.file, mid_3_cfg.mode, mid_3_cfg.dynamic
        )
        end_1_title = SegmentUI.format_title(
            "End 1", end_1_cfg.file, end_1_cfg.mode, end_1_cfg.dynamic
        )
        end_2_title = SegmentUI.format_title(
            "End 2", end_2_cfg.file, end_2_cfg.mode, end_2_cfg.dynamic
        )
        end_3_title = SegmentUI.format_title(
            "End 3", end_3_cfg.file, end_3_cfg.mode, end_3_cfg.dynamic
        )

        return (
            prompt,
            start_1_title,
            start_2_title,
            start_3_title,
            mid_1_title,
            mid_2_title,
            mid_3_title,
            end_1_title,
            end_2_title,
            end_3_title,
            state,
        )

    # Collect all segment inputs
    # NOTE: Start 2 and Start 3 condition components are interspersed with their segment inputs
    # (condition components already extracted above for handlers)
    all_segment_inputs = (
        start_1.get_input_components()  # Start 1 (no conditions)
        + [condition_type_2, condition_text_2, condition_dynamic_2]  # Condition inputs for Start 2
        + start_2.get_input_components()
        + [condition_type_3, condition_text_3, condition_dynamic_3]  # Condition inputs for Start 3
        + start_3.get_input_components()
        + mid_1.get_input_components()
        + mid_2.get_input_components()
        + mid_3.get_input_components()
        + end_1.get_input_components()
        + end_2.get_input_components()
        + end_3.get_input_components()
        + [ui_state]
    )

    build_prompt_btn.click(
        fn=build_and_update_prompt,
        inputs=all_segment_inputs,
        outputs=[
            prompt_input,
            start_1.title,
            start_2.title,
            start_3.title,
            mid_1.title,
            mid_2.title,
            mid_3.title,
            end_1.title,
            end_2.title,
            end_3.title,
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
        """Wrapper to convert segment values to SegmentConfig objects.

        Now includes condition text concatenation for Start 2 and Start 3.
        """
        # Extract values - includes image editing inputs (3 images + instruction)
        input_img_1 = values[0]
        input_img_2 = values[1]
        input_img_3 = values[2]
        instruction = values[3]
        prompt = values[4]
        width = values[5]
        height = values[6]
        num_steps = values[7]
        batch_size = values[8]
        runs = values[9]
        seed = values[10]
        use_random_seed = values[11]

        # Segment values
        # (9 values each: text, path, file, mode, line, range_end, count, dynamic,
        # sequential_start_line)
        # Start 1 (no conditions)
        start_1_values = list(values[12:21])

        # Start 2 with conditions (3 condition values + 9 segment values)
        condition_type_2_val = values[21]
        condition_text_2_val = values[22]
        condition_dynamic_2_val = values[23]
        start_2_values = list(values[24:33])

        # Start 3 with conditions (3 condition values + 9 segment values)
        condition_type_3_val = values[33]
        condition_text_3_val = values[34]
        condition_dynamic_3_val = values[35]
        start_3_values = list(values[36:45])

        # Remaining segments
        mid_1_values = values[45:54]
        mid_2_values = values[54:63]
        mid_3_values = values[63:72]
        end_1_values = values[72:81]
        end_2_values = values[81:90]
        end_3_values = values[90:99]
        state = values[99]

        # Concatenate condition text with Start 2 text if condition is enabled
        # BUT only if NOT dynamic (dynamic conditions are regenerated per-run inside generate_image)
        if condition_type_2_val != "None" and condition_text_2_val and not condition_dynamic_2_val:
            # Condition text comes first, then user text (if any)
            original_text = start_2_values[0]  # First value is text
            if original_text and original_text.strip():
                start_2_values[0] = f"{condition_text_2_val}, {original_text}"
            else:
                start_2_values[0] = condition_text_2_val

        # Concatenate condition text with Start 3 text if condition is enabled
        # BUT only if NOT dynamic (dynamic conditions are regenerated per-run inside generate_image)
        if condition_type_3_val != "None" and condition_text_3_val and not condition_dynamic_3_val:
            # Condition text comes first, then user text (if any)
            original_text = start_3_values[0]  # First value is text
            if original_text and original_text.strip():
                start_3_values[0] = f"{condition_text_3_val}, {original_text}"
            else:
                start_3_values[0] = condition_text_3_val

        # Convert to SegmentConfig objects
        start_1_cfg = SegmentUI.values_to_config(*start_1_values)
        start_2_cfg = SegmentUI.values_to_config(*start_2_values)
        start_3_cfg = SegmentUI.values_to_config(*start_3_values)
        mid_1_cfg = SegmentUI.values_to_config(*mid_1_values)
        mid_2_cfg = SegmentUI.values_to_config(*mid_2_values)
        mid_3_cfg = SegmentUI.values_to_config(*mid_3_values)
        end_1_cfg = SegmentUI.values_to_config(*end_1_values)
        end_2_cfg = SegmentUI.values_to_config(*end_2_values)
        end_3_cfg = SegmentUI.values_to_config(*end_3_values)

        # Collect input images (filter out None values)
        input_images = [img for img in [input_img_1, input_img_2, input_img_3] if img is not None]

        # Call generate_image with clean parameters (includes image editing params and conditions)
        return generate_image(
            prompt,
            width,
            height,
            num_steps,
            batch_size,
            runs,
            seed,
            use_random_seed,
            start_1_cfg,
            start_2_cfg,
            start_3_cfg,
            mid_1_cfg,
            mid_2_cfg,
            mid_3_cfg,
            end_1_cfg,
            end_2_cfg,
            end_3_cfg,
            state,
            input_images=input_images if input_images else None,
            instruction=instruction,
            # Condition parameters for Start 2
            condition_type_2=condition_type_2_val,
            condition_text_2=condition_text_2_val,
            condition_dynamic_2=condition_dynamic_2_val,
            # Condition parameters for Start 3
            condition_type_3=condition_type_3_val,
            condition_text_3=condition_text_3_val,
            condition_dynamic_3=condition_dynamic_3_val,
        )

    # Collect all inputs for generation (includes image editing inputs)
    generation_inputs = [
        input_image_1,
        input_image_2,
        input_image_3,
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

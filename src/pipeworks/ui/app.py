"""Gradio UI for Pipeworks Image Generator - Refactored Version."""

import logging
from typing import Any

import gradio as gr

from pipeworks.core.config import config
from pipeworks.plugins.base import plugin_registry

from .aspect_ratios import ASPECT_RATIOS
from .components import (
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
from .models import DEFAULT_SEED, MAX_SEED, UIState
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

        gr.Markdown("""
            # Pipeworks Image Generator
            ### Multi-model AI image generation and editing
            """)

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

            # Prompt Builder (Refactored to use dynamic segment plugin system)
            with gr.Accordion("Prompt Builder", open=False):
                gr.Markdown(
                    "*Build prompts by combining text and random lines from files "
                    "in the `inputs/` directory*\n\n"
                    "*Click folders (📁) to navigate, select files to use*\n\n"
                    "**Add 1-10 segments as needed. Each segment supports text, "
                    "files, and conditions.**"
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

                # Create dynamic segments using CompleteSegmentPlugin
                # We create 10 segments upfront (max capacity) and control visibility
                from pipeworks.ui.segment_plugins import CompleteSegmentPlugin

                plugin = CompleteSegmentPlugin()
                segments = []

                gr.Markdown("**Prompt Segments**")

                # Create all 10 segments upfront (visibility controlled by state)
                for i in range(10):
                    segment = plugin.create_ui(str(i), initial_choices)
                    segments.append(segment)
                    # Hide segments 1-9 initially (only show segment 0)
                    if i > 0:
                        segment.container.visible = False

                # Segment manager controls
                with gr.Row():
                    add_segment_btn = gr.Button("➕ Add Segment", variant="secondary", size="sm")
                    remove_segment_btn = gr.Button(
                        "➖ Remove Last Segment", variant="secondary", size="sm"
                    )

                segment_status = gr.Markdown("**Total: 1 segment(s)**")

                # Segment manager state (tracks visible segment indices)
                segment_manager_state = gr.State(
                    {
                        "visible_indices": [0],  # Only segment 0 visible initially
                        "next_segment_id": 1,
                        "max_segments": 10,
                        "min_segments": 1,
                    }
                )

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
                            value=True,  # Enabled by default so prompts are always saved
                            info="Save prompt and generation parameters to files",
                        )

                        # Visible by default since plugin is enabled by default
                        with gr.Group(visible=True) as metadata_settings:
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
    gr.Markdown(f"""
        ---
        **Model:** {config.model_id} | **Device:** {config.device} |
        **Dtype:** {config.torch_dtype}

        *Outputs saved to: {config.outputs_dir}*
        """)

    # =========================================================================
    # Event handlers for dynamic segments
    # =========================================================================

    # Condition generation helper functions (used by all segments)
    def toggle_condition_type_handler(condition_type: str) -> tuple[str, dict[str, Any]]:
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

    # Wire up event handlers for all 10 segments
    event_handlers = {
        "navigate_file_selection": navigate_file_selection,
        "update_mode_visibility": update_mode_visibility,
        "toggle_condition_type": toggle_condition_type_handler,
        "regenerate_condition": regenerate_condition_type_handler,
    }

    for segment in segments:
        plugin.register_events(segment, ui_state, event_handlers)

    # =========================================================================
    # Segment management button handlers (add/remove)
    # =========================================================================

    def add_segment_click_handler(
        segment_manager_state_value: dict[str, Any], ui_state_value: UIState
    ) -> tuple[
        dict[str, Any],
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        UIState,
    ]:
        """Add a new segment when the Add button is clicked.

        Args:
            segment_manager_state_value: Current segment manager state
            ui_state_value: Current UI state

        Returns:
            Tuple of (updated_segment_manager_state, status_message,
                     *10_visibility_updates, updated_ui_state)
        """

        # Check if we can add
        visible_indices = segment_manager_state_value.get("visible_indices", [0])
        max_segments = segment_manager_state_value.get("max_segments", 10)

        if len(visible_indices) >= max_segments:
            return (  # type: ignore[return-value]
                segment_manager_state_value,
                f"**Total: {len(visible_indices)} segment(s)** "
                f"(Maximum {max_segments} reached)",
                *[gr.update() for _ in range(10)],  # No visibility changes
                ui_state_value,
            )

        # Add next segment index
        next_index = len(visible_indices)
        updated_visible = visible_indices + [next_index]

        updated_state = {
            "visible_indices": updated_visible,
            "next_segment_id": segment_manager_state_value.get("next_segment_id", 1) + 1,
            "max_segments": max_segments,
            "min_segments": segment_manager_state_value.get("min_segments", 1),
        }

        # Make segments visible up to the new index
        visibility_updates = [gr.update(visible=(i in updated_visible)) for i in range(10)]

        # Update status message
        status = f"**Total: {len(updated_visible)} segment(s)**"

        return (updated_state, status, *visibility_updates, ui_state_value)  # type: ignore[return-value]

    def remove_segment_click_handler(
        segment_manager_state_value: dict[str, Any], ui_state_value: UIState
    ) -> tuple[
        dict[str, Any],
        str,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        UIState,
    ]:
        """Remove the last segment when the Remove button is clicked.

        Args:
            segment_manager_state_value: Current segment manager state
            ui_state_value: Current UI state

        Returns:
            Tuple of (updated_segment_manager_state, status_message,
                     *10_visibility_updates, updated_ui_state)
        """

        # Check if we can remove
        visible_indices = segment_manager_state_value.get("visible_indices", [0])
        min_segments = segment_manager_state_value.get("min_segments", 1)

        if len(visible_indices) <= min_segments:
            return (  # type: ignore[return-value]
                segment_manager_state_value,
                f"**Total: {len(visible_indices)} segment(s)** "
                f"(Minimum {min_segments} required)",
                *[gr.update() for _ in range(10)],  # No visibility changes
                ui_state_value,
            )

        # Remove last visible index
        updated_visible = visible_indices[:-1]

        updated_state = {
            "visible_indices": updated_visible,
            "next_segment_id": segment_manager_state_value.get("next_segment_id", 0),
            "max_segments": segment_manager_state_value.get("max_segments", 10),
            "min_segments": min_segments,
        }

        # Update visibility
        visibility_updates = [gr.update(visible=(i in updated_visible)) for i in range(10)]

        # Update status message
        status = f"**Total: {len(updated_visible)} segment(s)**"

        return (updated_state, status, *visibility_updates, ui_state_value)  # type: ignore[return-value]

    # Wire up add/remove button handlers
    add_segment_btn.click(
        fn=add_segment_click_handler,
        inputs=[segment_manager_state, ui_state],
        outputs=[
            segment_manager_state,
            segment_status,
            *[seg.container for seg in segments],  # Update visibility for all 10 segments
            ui_state,
        ],
    )

    remove_segment_btn.click(
        fn=remove_segment_click_handler,
        inputs=[segment_manager_state, ui_state],
        outputs=[
            segment_manager_state,
            segment_status,
            *[seg.container for seg in segments],  # Update visibility for all 10 segments
            ui_state,
        ],
    )

    # Build prompt button handler (refactored for dynamic segments)
    def build_and_update_prompt(*values):
        """Build prompt from segment values and update UI.

        Now works with variable number of segments (1-10) using CompleteSegmentPlugin.

        Args:
            *values: Variable number of values:
                - segment_manager_state (dict)
                - All 10 segments' input values (14 values each: 11 segment + 3 condition)
                - ui_state

        Returns:
            Tuple of (prompt, *segment_title_updates, ui_state)
        """
        # Extract segment_manager_state and ui_state
        segment_manager_state_val = values[0]
        ui_state_val = values[-1]

        # Extract values for all 10 segments (14 values each)
        # Values order: segment_manager_state, seg0[14], seg1[14], ..., seg9[14], ui_state
        segment_values_list = []
        start_idx = 1  # Skip segment_manager_state
        for i in range(10):
            seg_vals = values[start_idx : start_idx + 14]
            segment_values_list.append(seg_vals)
            start_idx += 14

        # Get visible segment indices
        visible_indices = segment_manager_state_val.get("visible_indices", [0])

        # Convert visible segments to SegmentConfig objects
        segment_configs = []
        for idx in visible_indices:
            segment_configs.append(plugin.values_to_config(*segment_values_list[idx]))

        # Initialize state
        ui_state_val = initialize_ui_state(ui_state_val)

        # Process segments with conditions (prepend condition to text)
        # This mirrors the logic in generate_image for consistency
        from dataclasses import replace

        processed_configs = []
        for seg in segment_configs:
            # If segment has a condition, prepend it to the text
            if seg.condition_type != "None":
                # Use the condition_text if static, or generate if dynamic
                # For build prompt, we use the current condition_text (no randomization)
                condition_text = seg.condition_text if seg.condition_text else ""

                if condition_text:
                    # Get delimiter and prepend condition to text
                    delimiter_value = seg.get_delimiter_value()
                    original_text = seg.text if seg.text else ""

                    if original_text and original_text.strip():
                        modified_text = f"{condition_text}{delimiter_value}{original_text}"
                    else:
                        modified_text = condition_text

                    # Create modified config with condition prepended
                    seg = replace(seg, text=modified_text)

            processed_configs.append(seg)

        # Build prompt using refactored handler (accepts list[SegmentConfig])
        try:
            prompt = build_combined_prompt(
                processed_configs,  # Pass processed configs with conditions
                ui_state_val,
            )
        except ValidationError as e:
            prompt = f"Error: {str(e)}"

        # Update titles for all 10 segments (only visible ones will show)
        segment_titles = []
        for i in range(10):
            if i in visible_indices:
                # Find the corresponding config (visible_indices may not be sequential)
                cfg_idx = visible_indices.index(i)
                cfg = segment_configs[cfg_idx]
                title = f"**Segment {i}**"
                if cfg.condition_type != "None":
                    title += f" • 🎭 {cfg.condition_type}"
                if cfg.file and cfg.file != "(None)":
                    title += f" • 📄 {cfg.file}"
                if cfg.dynamic:
                    title += " • 🔄 Dynamic"
                segment_titles.append(title)
            else:
                segment_titles.append(f"**Segment {i}**")  # Default title for hidden segments

        return (prompt, *segment_titles, ui_state_val)

    # Collect inputs: segment_manager_state + all 10 segments' inputs + ui_state
    all_segment_inputs = [segment_manager_state]
    for segment in segments:
        all_segment_inputs.extend(plugin.get_input_components(segment))
    all_segment_inputs.append(ui_state)

    build_prompt_btn.click(
        fn=build_and_update_prompt,
        inputs=all_segment_inputs,
        outputs=[
            prompt_input,
            *[seg.title for seg in segments],  # All 10 segment titles
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

    # Generate button handler (refactored for dynamic segments)
    def generate_wrapper(*values):
        """Wrapper to convert segment values to SegmentConfig objects.

        Now works with variable number of segments (1-10) using CompleteSegmentPlugin.
        """
        # Extract image editing inputs and generation params
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

        # Extract segment_manager_state and ui_state
        segment_manager_state_val = values[12]
        ui_state_val = values[-1]

        # Extract values for all 10 segments (14 values each)
        # Values order: [gen params], segment_manager_state, seg0[14], ..., seg9[14], ui_state
        segment_values_list = []
        start_idx = 13  # Skip gen params + segment_manager_state
        for i in range(10):
            seg_vals = values[start_idx : start_idx + 14]
            segment_values_list.append(seg_vals)
            start_idx += 14

        # Get visible segment indices
        visible_indices = segment_manager_state_val.get("visible_indices", [0])

        # Convert visible segments to SegmentConfig objects using CompleteSegmentPlugin
        segment_configs = []
        for idx in visible_indices:
            segment_configs.append(plugin.values_to_config(*segment_values_list[idx]))

        # Collect input images (filter out None values)
        input_images = [img for img in [input_img_1, input_img_2, input_img_3] if img is not None]

        # Call generate_image with refactored signature (accepts list[SegmentConfig])
        return generate_image(
            prompt,
            width,
            height,
            num_steps,
            batch_size,
            runs,
            seed,
            use_random_seed,
            segment_configs,  # Pass list directly
            ui_state_val,
            input_images=input_images if input_images else None,
            instruction=instruction,
        )

    # Collect all inputs for generation (includes image editing inputs + segments)
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
    ] + all_segment_inputs  # Includes segment_manager_state + all 10 segments + ui_state

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

    # Prompt display - full width below gallery for easy viewing
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
    ).then(
        # Clear selected image and metadata display after refresh to prevent stale state
        lambda: (None, "*Select an image to view metadata*"),
        outputs=[selected_image, metadata_display],
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

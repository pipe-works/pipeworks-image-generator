"""Gradio UI for Pipeworks Image Generator."""

import logging
import random
from typing import Dict, List

import gradio as gr

from pipeworks.core.config import config
from pipeworks.core.pipeline import ImageGenerator
from pipeworks.core.tokenizer import TokenizerAnalyzer
from pipeworks.core.prompt_builder import PromptBuilder
from pipeworks.plugins.base import PluginBase, plugin_registry
# Import all plugins to ensure they're registered
from pipeworks.plugins import SaveMetadataPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state for plugins
active_plugins: Dict[str, PluginBase] = {}
generator: ImageGenerator = None  # Will be initialized with plugins
tokenizer_analyzer: TokenizerAnalyzer = None  # Will be initialized on startup
prompt_builder: PromptBuilder = None  # Will be initialized on startup


def update_generator_plugins():
    """Update the generator's plugin list without recreating the generator."""
    global generator
    if generator is not None:
        plugin_list = [p for p in active_plugins.values() if p.enabled]
        generator.plugins = plugin_list
        logger.info(f"Updated generator with {len(plugin_list)} active plugins")


def toggle_plugin(plugin_name: str, enabled: bool, **plugin_config):
    """
    Toggle a plugin on/off and update its configuration.

    Args:
        plugin_name: Name of the plugin
        enabled: Whether to enable the plugin
        **plugin_config: Plugin-specific configuration
    """
    global active_plugins

    if enabled:
        # Always reinstantiate to ensure config is properly applied
        active_plugins[plugin_name] = plugin_registry.instantiate(plugin_name, **plugin_config)
    else:
        # Disable the plugin
        if plugin_name in active_plugins:
            active_plugins[plugin_name].enabled = False

    # Update generator's plugin list (doesn't recreate generator or unload model)
    update_generator_plugins()
    return gr.update(visible=enabled)


def set_aspect_ratio(ratio_name: str) -> tuple[int, int]:
    """
    Set width and height based on aspect ratio preset.

    Args:
        ratio_name: Name of the aspect ratio preset

    Returns:
        Tuple of (width, height)
    """
    aspect_ratios = {
        "Square 1:1 (1024x1024)": (1024, 1024),
        "Widescreen 16:9 (1280x720)": (1280, 720),
        "Widescreen 16:9 (1600x896)": (1600, 896),
        "Portrait 9:16 (720x1280)": (720, 1280),
        "Portrait 9:16 (896x1600)": (896, 1600),
        "Standard 3:2 (1280x832)": (1280, 832),
        "Standard 2:3 (832x1280)": (832, 1280),
        "Standard 3:2 (1536x1024)": (1536, 1024),
        "Custom": (config.default_width, config.default_height),
    }

    width, height = aspect_ratios.get(ratio_name, (config.default_width, config.default_height))
    return gr.update(value=width), gr.update(value=height)


def analyze_prompt(prompt: str) -> str:
    """
    Analyze prompt tokenization and return formatted results.

    Args:
        prompt: Text prompt to analyze

    Returns:
        Formatted markdown string with tokenization details
    """
    global tokenizer_analyzer

    if not prompt or prompt.strip() == "":
        return "*Enter a prompt to see tokenization analysis*"

    try:
        # Lazy load tokenizer on first use
        if tokenizer_analyzer is None:
            tokenizer_analyzer = TokenizerAnalyzer(
                model_id=config.model_id,
                cache_dir=config.models_dir
            )
            tokenizer_analyzer.load()

        # Analyze the prompt
        analysis = tokenizer_analyzer.analyze(prompt)

        # Format results
        token_count = analysis["token_count"]
        tokens = analysis["tokens"]
        formatted_tokens = tokenizer_analyzer.format_tokens(tokens)

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

        return result.strip()

    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}", exc_info=True)
        return f"*Error analyzing prompt: {str(e)}*"


def get_available_folders() -> List[str]:
    """Get list of available folders from inputs directory."""
    global prompt_builder
    if prompt_builder is None:
        prompt_builder = PromptBuilder(config.inputs_dir)

    folders = prompt_builder.scan_folders()
    return ["(None)"] + folders


def get_files_in_folder(folder: str) -> gr.Dropdown:
    """Get list of files in a specific folder."""
    global prompt_builder
    if prompt_builder is None:
        prompt_builder = PromptBuilder(config.inputs_dir)

    if folder == "(None)":
        return gr.update(choices=["(None)"], value="(None)")

    files = prompt_builder.get_files_in_folder(folder)
    return gr.update(choices=["(None)"] + files, value="(None)" if not files else files[0])


def update_segment_titles(
    start_folder: str,
    start_file: str,
    start_mode: str,
    start_dynamic: bool,
    middle_folder: str,
    middle_file: str,
    middle_mode: str,
    middle_dynamic: bool,
    end_folder: str,
    end_file: str,
    end_mode: str,
    end_dynamic: bool,
) -> tuple[gr.Markdown, gr.Markdown, gr.Markdown]:
    """Update segment titles with status and configuration."""

    def format_title(name: str, folder: str, file: str, mode: str, dynamic: bool) -> str:
        # Check if segment is configured
        has_config = (folder != "(None)" and file != "(None)")

        if has_config:
            # Green for configured segments
            parts = [f'<span style="color: #22c55e">**{name}**</span>']
            parts.append(f'<span style="color: #22c55e">| {mode}</span>')
            if dynamic:
                parts.append('<span style="color: #22c55e">| Dynamic</span>')
            return " ".join(parts)
        else:
            # Gray for unconfigured segments
            return f"**{name}**"

    start_title = format_title("Start Segment", start_folder, start_file, start_mode, start_dynamic)
    middle_title = format_title("Middle Segment", middle_folder, middle_file, middle_mode, middle_dynamic)
    end_title = format_title("End Segment", end_folder, end_file, end_mode, end_dynamic)

    return (
        gr.update(value=start_title),
        gr.update(value=middle_title),
        gr.update(value=end_title),
    )


def build_prompt_and_update_titles(
    start_text: str,
    start_folder: str,
    start_file: str,
    start_mode: str,
    start_line: int,
    start_range_end: int,
    start_count: int,
    start_dynamic: bool,
    middle_text: str,
    middle_folder: str,
    middle_file: str,
    middle_mode: str,
    middle_line: int,
    middle_range_end: int,
    middle_count: int,
    middle_dynamic: bool,
    end_text: str,
    end_folder: str,
    end_file: str,
    end_mode: str,
    end_line: int,
    end_range_end: int,
    end_count: int,
    end_dynamic: bool,
) -> tuple[str, gr.Markdown, gr.Markdown, gr.Markdown]:
    """Build prompt and update segment titles."""
    # Build the prompt
    prompt = build_combined_prompt(
        start_text, start_folder, start_file, start_mode, start_line, start_range_end, start_count,
        middle_text, middle_folder, middle_file, middle_mode, middle_line, middle_range_end, middle_count,
        end_text, end_folder, end_file, end_mode, end_line, end_range_end, end_count,
    )

    # Update titles
    titles = update_segment_titles(
        start_folder, start_file, start_mode, start_dynamic,
        middle_folder, middle_file, middle_mode, middle_dynamic,
        end_folder, end_file, end_mode, end_dynamic,
    )

    return (prompt, *titles)


def build_combined_prompt(
    start_text: str,
    start_folder: str,
    start_file: str,
    start_mode: str,
    start_line: int,
    start_range_end: int,
    start_count: int,
    middle_text: str,
    middle_folder: str,
    middle_file: str,
    middle_mode: str,
    middle_line: int,
    middle_range_end: int,
    middle_count: int,
    end_text: str,
    end_folder: str,
    end_file: str,
    end_mode: str,
    end_line: int,
    end_range_end: int,
    end_count: int,
) -> str:
    """
    Build a combined prompt from multiple segments.

    Args:
        start_*: Start segment parameters
        middle_*: Middle segment parameters
        end_*: End segment parameters

    Returns:
        Combined prompt string
    """
    global prompt_builder
    if prompt_builder is None:
        prompt_builder = PromptBuilder(config.inputs_dir)

    segments = []

    # Helper to add segment
    def add_segment(text, folder, file, mode, line, range_end, count):
        # Add user text if provided
        if text and text.strip():
            segments.append(("text", text.strip()))

        # Add file selection if file is chosen
        if folder and folder != "(None)" and file and file != "(None)":
            # Get full file path
            full_path = prompt_builder.get_full_path(folder, file)

            if mode == "Random Line":
                segments.append(("file_random", full_path))
            elif mode == "Specific Line":
                segments.append(("file_specific", f"{full_path}|{line}"))
            elif mode == "Line Range":
                segments.append(("file_range", f"{full_path}|{line}|{range_end}"))
            elif mode == "All Lines":
                segments.append(("file_all", full_path))
            elif mode == "Random Multiple":
                segments.append(("file_random_multi", f"{full_path}|{count}"))

    # Add segments in order
    add_segment(start_text, start_folder, start_file, start_mode, start_line, start_range_end, start_count)
    add_segment(middle_text, middle_folder, middle_file, middle_mode, middle_line, middle_range_end, middle_count)
    add_segment(end_text, end_folder, end_file, end_mode, end_line, end_range_end, end_count)

    # Build the final prompt
    try:
        result = prompt_builder.build_prompt(segments)
        return result if result else ""
    except Exception as e:
        logger.error(f"Error building prompt: {e}", exc_info=True)
        return f"Error: {str(e)}"


def generate_image(
    prompt: str,
    width: int,
    height: int,
    num_steps: int,
    batch_size: int,
    runs: int,
    seed: int,
    use_random_seed: bool,
    # Segment parameters for dynamic prompts
    start_text: str,
    start_folder: str,
    start_file: str,
    start_mode: str,
    start_line: int,
    start_range_end: int,
    start_count: int,
    start_dynamic: bool,
    middle_text: str,
    middle_folder: str,
    middle_file: str,
    middle_mode: str,
    middle_line: int,
    middle_range_end: int,
    middle_count: int,
    middle_dynamic: bool,
    end_text: str,
    end_folder: str,
    end_file: str,
    end_mode: str,
    end_line: int,
    end_range_end: int,
    end_count: int,
    end_dynamic: bool,
) -> tuple[List[str], str, str]:
    """
    Generate image(s) from the UI inputs.

    Args:
        prompt: Text prompt (used if no dynamic segments)
        width: Image width
        height: Image height
        num_steps: Number of inference steps
        batch_size: Number of images per run
        runs: Number of runs to execute
        seed: Random seed
        use_random_seed: Whether to use a random seed
        start_*, middle_*, end_*: Segment parameters for dynamic prompts

    Returns:
        Tuple of (image_paths, info_text, seed_used)
    """
    # Check if any segment is dynamic
    has_dynamic = start_dynamic or middle_dynamic or end_dynamic

    if not has_dynamic and (not prompt or prompt.strip() == ""):
        return [], "Error: Please provide a prompt or enable dynamic segments", str(seed)

    try:
        batch_size = max(1, int(batch_size))  # Ensure at least 1
        runs = max(1, int(runs))  # Ensure at least 1
        total_images = batch_size * runs

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
                    current_prompt = build_combined_prompt(
                        start_text, start_folder, start_file, start_mode, start_line, start_range_end, start_count,
                        middle_text, middle_folder, middle_file, middle_mode, middle_line, middle_range_end, middle_count,
                        end_text, end_folder, end_file, end_mode, end_line, end_range_end, end_count,
                    )
                    if current_prompt:
                        prompts_used.append(current_prompt)
                    else:
                        current_prompt = prompt  # Fall back to static prompt
                else:
                    current_prompt = prompt

                # Generate random seed if requested, or use sequential seed
                if use_random_seed:
                    actual_seed = random.randint(0, 2**32 - 1)
                else:
                    actual_seed = current_seed
                    current_seed += 1

                # Generate and save image (plugins are called automatically by the pipeline)
                image, save_path = generator.generate_and_save(
                    prompt=current_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    seed=actual_seed,
                )

                generated_paths.append(str(save_path))
                seeds_used.append(actual_seed)

                image_num = run * batch_size + i + 1
                logger.info(f"Image {image_num}/{total_images} complete (Run {run+1}, Batch {i+1}): {save_path}")

        # Create info text
        active_plugin_names = [p.name for p in generator.plugins if p.enabled]
        plugins_info = f"\n**Active Plugins:** {', '.join(active_plugin_names)}" if active_plugin_names else ""

        # Format seeds display
        if total_images == 1:
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
**Generation Complete!**

**Prompt:** {prompt if not has_dynamic else "(Dynamic)"}
**Dimensions:** {width}x{height}
**Steps:** {num_steps}
**Batch Size:** {batch_size} × **Runs:** {runs} = **Total:** {total_images} images
**Seeds:** {seeds_display}
**Saved to:** {paths_display}{dynamic_info}{plugins_info}
        """

        # Return all generated images for display in gallery
        return generated_paths, info.strip(), str(seeds_used[-1])

    except Exception as e:
        logger.error(f"Error generating image: {e}", exc_info=True)
        return [], f"Error: {str(e)}", str(seed)


def create_ui() -> tuple[gr.Blocks, str]:
    """Create the Gradio UI with dynamic plugins.

    Returns:
        Tuple of (app, custom_css)
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
                    gr.Markdown("*Build prompts by combining text and random lines from files in the `inputs/` directory*")

                    # Get available folders
                    available_folders = get_available_folders()

                    # Start Segment
                    with gr.Group():
                        start_segment_title = gr.Markdown("**Start Segment**")
                        start_text = gr.Textbox(label="Start Text", placeholder="Optional text...", lines=1)
                        with gr.Row():
                            start_folder = gr.Dropdown(label="Folder", choices=available_folders, value="(None)")
                            start_file = gr.Dropdown(label="File", choices=["(None)"], value="(None)")
                        with gr.Row():
                            start_mode = gr.Dropdown(
                                label="Mode",
                                choices=["Random Line", "Specific Line", "Line Range", "All Lines", "Random Multiple"],
                                value="Random Line"
                            )
                            start_dynamic = gr.Checkbox(
                                label="Dynamic",
                                value=False,
                                info="Rebuild this segment for each image"
                            )
                        with gr.Row():
                            start_line = gr.Number(label="Line #", value=1, minimum=1, precision=0, visible=False)
                            start_range_end = gr.Number(label="End Line #", value=1, minimum=1, precision=0, visible=False)
                            start_count = gr.Number(label="Count", value=1, minimum=1, maximum=10, precision=0, visible=False)

                    # Middle Segment
                    with gr.Group():
                        middle_segment_title = gr.Markdown("**Middle Segment**")
                        middle_text = gr.Textbox(label="Middle Text", placeholder="Optional text...", lines=1)
                        with gr.Row():
                            middle_folder = gr.Dropdown(label="Folder", choices=available_folders, value="(None)")
                            middle_file = gr.Dropdown(label="File", choices=["(None)"], value="(None)")
                        with gr.Row():
                            middle_mode = gr.Dropdown(
                                label="Mode",
                                choices=["Random Line", "Specific Line", "Line Range", "All Lines", "Random Multiple"],
                                value="Random Line"
                            )
                            middle_dynamic = gr.Checkbox(
                                label="Dynamic",
                                value=False,
                                info="Rebuild this segment for each image"
                            )
                        with gr.Row():
                            middle_line = gr.Number(label="Line #", value=1, minimum=1, precision=0, visible=False)
                            middle_range_end = gr.Number(label="End Line #", value=1, minimum=1, precision=0, visible=False)
                            middle_count = gr.Number(label="Count", value=1, minimum=1, maximum=10, precision=0, visible=False)

                    # End Segment
                    with gr.Group():
                        end_segment_title = gr.Markdown("**End Segment**")
                        end_text = gr.Textbox(label="End Text", placeholder="Optional text...", lines=1)
                        with gr.Row():
                            end_folder = gr.Dropdown(label="Folder", choices=available_folders, value="(None)")
                            end_file = gr.Dropdown(label="File", choices=["(None)"], value="(None)")
                        with gr.Row():
                            end_mode = gr.Dropdown(
                                label="Mode",
                                choices=["Random Line", "Specific Line", "Line Range", "All Lines", "Random Multiple"],
                                value="Random Line"
                            )
                            end_dynamic = gr.Checkbox(
                                label="Dynamic",
                                value=False,
                                info="Rebuild this segment for each image"
                            )
                        with gr.Row():
                            end_line = gr.Number(label="Line #", value=1, minimum=1, precision=0, visible=False)
                            end_range_end = gr.Number(label="End Line #", value=1, minimum=1, precision=0, visible=False)
                            end_count = gr.Number(label="Count", value=1, minimum=1, maximum=10, precision=0, visible=False)

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
                    choices=[
                        "Square 1:1 (1024x1024)",
                        "Widescreen 16:9 (1280x720)",
                        "Widescreen 16:9 (1600x896)",
                        "Portrait 9:16 (720x1280)",
                        "Portrait 9:16 (896x1600)",
                        "Standard 3:2 (1280x832)",
                        "Standard 2:3 (832x1280)",
                        "Standard 3:2 (1536x1024)",
                        "Custom",
                    ],
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
                        value=42,
                        precision=0,
                        minimum=0,
                        maximum=2**32 - 1,
                    )
                    random_seed_checkbox = gr.Checkbox(
                        label="Random Seed",
                        value=False,
                        info="Generate a new random seed each time",
                    )

                # Plugins Section - Collapsible and Scrollable
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
                            def toggle_save_metadata(enabled, folder, prefix):
                                toggle_plugin("SaveMetadata", enabled, folder_name=folder, filename_prefix=prefix)
                                return gr.update(visible=enabled)

                            save_metadata_check.change(
                                fn=toggle_save_metadata,
                                inputs=[save_metadata_check, metadata_folder, metadata_prefix],
                                outputs=[metadata_settings],
                            )

                            # Update plugin config when settings change
                            def update_plugin_config(enabled, folder, prefix):
                                if enabled:
                                    toggle_plugin("SaveMetadata", enabled, folder_name=folder, filename_prefix=prefix)

                            metadata_folder.change(
                                fn=update_plugin_config,
                                inputs=[save_metadata_check, metadata_folder, metadata_prefix],
                            )
                            metadata_prefix.change(
                                fn=update_plugin_config,
                                inputs=[save_metadata_check, metadata_folder, metadata_prefix],
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
                    value="42",
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
        # Prompt Builder folder change handlers
        start_folder.change(
            fn=get_files_in_folder,
            inputs=[start_folder],
            outputs=[start_file],
        )

        middle_folder.change(
            fn=get_files_in_folder,
            inputs=[middle_folder],
            outputs=[middle_file],
        )

        end_folder.change(
            fn=get_files_in_folder,
            inputs=[end_folder],
            outputs=[end_file],
        )

        # Prompt Builder mode change handlers
        def update_mode_visibility(mode):
            """Update visibility of line number inputs based on mode."""
            return (
                gr.update(visible=mode in ["Specific Line", "Line Range"]),
                gr.update(visible=mode == "Line Range"),
                gr.update(visible=mode == "Random Multiple"),
            )

        # Start segment mode handler
        start_mode.change(
            fn=update_mode_visibility,
            inputs=[start_mode],
            outputs=[start_line, start_range_end, start_count],
        )

        # Middle segment mode handler
        middle_mode.change(
            fn=update_mode_visibility,
            inputs=[middle_mode],
            outputs=[middle_line, middle_range_end, middle_count],
        )

        # End segment mode handler
        end_mode.change(
            fn=update_mode_visibility,
            inputs=[end_mode],
            outputs=[end_line, end_range_end, end_count],
        )

        # Build prompt button handler
        build_prompt_btn.click(
            fn=build_prompt_and_update_titles,
            inputs=[
                start_text, start_folder, start_file, start_mode, start_line, start_range_end, start_count, start_dynamic,
                middle_text, middle_folder, middle_file, middle_mode, middle_line, middle_range_end, middle_count, middle_dynamic,
                end_text, end_folder, end_file, end_mode, end_line, end_range_end, end_count, end_dynamic,
            ],
            outputs=[prompt_input, start_segment_title, middle_segment_title, end_segment_title],
        )

        # Aspect ratio preset handler
        aspect_ratio_dropdown.change(
            fn=set_aspect_ratio,
            inputs=[aspect_ratio_dropdown],
            outputs=[width_slider, height_slider],
        )

        # Tokenizer analyzer handler (updates on change)
        prompt_input.change(
            fn=analyze_prompt,
            inputs=[prompt_input],
            outputs=[tokenizer_output],
        )

        # Trigger analysis on app load with default prompt
        app.load(
            fn=analyze_prompt,
            inputs=[prompt_input],
            outputs=[tokenizer_output],
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                width_slider,
                height_slider,
                steps_slider,
                batch_input,
                runs_input,
                seed_input,
                random_seed_checkbox,
                # Segment parameters for dynamic prompts
                start_text, start_folder, start_file, start_mode, start_line, start_range_end, start_count, start_dynamic,
                middle_text, middle_folder, middle_file, middle_mode, middle_line, middle_range_end, middle_count, middle_dynamic,
                end_text, end_folder, end_file, end_mode, end_line, end_range_end, end_count, end_dynamic,
            ],
            outputs=[image_output, info_output, seed_used],
        )

    return app, custom_css


def main():
    """Main entry point for the application."""
    global generator

    logger.info("Starting Pipeworks Image Generator...")
    logger.info(f"Configuration: {config.model_dump()}")

    # Initialize generator with no plugins (plugins are added via UI)
    generator = ImageGenerator(config, plugins=[])

    # Pre-load model on startup
    logger.info("Pre-loading model...")
    try:
        generator.load_model()
    except Exception as e:
        logger.error(f"Failed to pre-load model: {e}")
        logger.warning("Model will be loaded on first generation attempt")

    # Log available plugins
    available_plugins = plugin_registry.list_available()
    logger.info(f"Available plugins: {available_plugins}")

    # Create and launch UI
    app, custom_css = create_ui()

    logger.info(
        f"Launching Gradio UI on {config.gradio_server_name}:{config.gradio_server_port}"
    )

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

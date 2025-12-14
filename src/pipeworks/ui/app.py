"""Gradio UI for Pipeworks Image Generator."""

import logging
import random
from typing import Dict, List

import gradio as gr

from pipeworks.core.config import config
from pipeworks.core.pipeline import ImageGenerator
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


def reinitialize_generator():
    """Reinitialize the generator with current active plugins."""
    global generator
    plugin_list = [p for p in active_plugins.values() if p.enabled]
    generator = ImageGenerator(config, plugins=plugin_list)
    logger.info(f"Generator reinitialized with {len(plugin_list)} active plugins")


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

    # Reinitialize generator with updated plugins
    reinitialize_generator()
    return gr.update(visible=enabled)


def generate_image(
    prompt: str,
    width: int,
    height: int,
    num_steps: int,
    seed: int,
    use_random_seed: bool,
) -> tuple[str, str, str]:
    """
    Generate an image from the UI inputs.

    Args:
        prompt: Text prompt
        width: Image width
        height: Image height
        num_steps: Number of inference steps
        seed: Random seed
        use_random_seed: Whether to use a random seed

    Returns:
        Tuple of (image_path, info_text, seed_used)
    """
    if not prompt or prompt.strip() == "":
        return None, "Error: Please provide a prompt", str(seed)

    try:
        # Generate random seed if requested
        actual_seed = random.randint(0, 2**32 - 1) if use_random_seed else seed

        # Generate and save image (plugins are called automatically by the pipeline)
        image, save_path = generator.generate_and_save(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            seed=actual_seed,
        )

        # Create info text
        active_plugin_names = [p.name for p in generator.plugins if p.enabled]
        plugins_info = f"\n**Active Plugins:** {', '.join(active_plugin_names)}" if active_plugin_names else ""

        info = f"""
**Generation Complete!**

**Prompt:** {prompt}
**Dimensions:** {width}x{height}
**Steps:** {num_steps}
**Seed:** {actual_seed}
**Saved to:** {save_path}{plugins_info}
        """

        return str(save_path), info.strip(), str(actual_seed)

    except Exception as e:
        logger.error(f"Error generating image: {e}", exc_info=True)
        return None, f"Error: {str(e)}", str(seed)


def create_ui() -> tuple[gr.Blocks, str]:
    """Create the Gradio UI with menu and dynamic plugins.

    Returns:
        Tuple of (app, custom_css)
    """

    # Use custom CSS for menu-like appearance
    custom_css = """
    .menu-bar {
        background-color: #1f2937;
        padding: 8px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
    }
    .menu-item {
        display: inline-block;
        color: #e5e7eb;
        margin-right: 20px;
        font-weight: 500;
        cursor: pointer;
    }
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
        # Menu Bar
        with gr.Row(elem_classes="menu-bar"):
            gr.HTML('<div class="menu-item">File</div><div class="menu-item">Edit</div><div class="menu-item">Plugins</div><div class="menu-item">Settings</div>')

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

                # Info display
                info_output = gr.Markdown(
                    label="Generation Info",
                    value="*Ready to generate images*",
                )

            with gr.Column(scale=1):
                # Output display
                gr.Markdown("### Generated Image")

                image_output = gr.Image(
                    label="Output",
                    type="filepath",
                    height=600,
                )

                # Show the seed that was actually used
                seed_used = gr.Textbox(
                    label="Seed Used",
                    interactive=False,
                    value="42",
                )

        # Example prompts
        with gr.Accordion("Example Prompts", open=False):
            gr.Examples(
                examples=[
                    ["A serene mountain landscape at sunset with vibrant colors"],
                    ["Young woman in red traditional dress, photorealistic portrait"],
                    ["Modern architectural building with glass facade and clean lines"],
                    ["Cute cat sleeping on a cozy blanket, soft lighting"],
                    ["Abstract geometric pattern with bold colors and shapes"],
                    ["Futuristic cityscape at night with neon lights"],
                ],
                inputs=prompt_input,
                label="Click to use example prompts",
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
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                width_slider,
                height_slider,
                steps_slider,
                seed_input,
                random_seed_checkbox,
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

"""Gradio UI for Pipeworks Image Generator."""

import logging
import random
from pathlib import Path

import gradio as gr

from pipeworks.core.config import config
from pipeworks.core.pipeline import ImageGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize generator
generator = ImageGenerator(config)


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

        # Generate and save image
        image, save_path = generator.generate_and_save(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            seed=actual_seed,
        )

        # Create info text
        info = f"""
**Generation Complete!**

**Prompt:** {prompt}
**Dimensions:** {width}x{height}
**Steps:** {num_steps}
**Seed:** {actual_seed}
**Saved to:** {save_path}
        """

        return str(save_path), info.strip(), str(actual_seed)

    except Exception as e:
        logger.error(f"Error generating image: {e}", exc_info=True)
        return None, f"Error: {str(e)}", str(seed)


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""

    with gr.Blocks(
        title="Pipeworks Image Generator",
        theme=gr.themes.Soft(),
    ) as app:
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

    return app


def main():
    """Main entry point for the application."""
    logger.info("Starting Pipeworks Image Generator...")
    logger.info(f"Configuration: {config.model_dump()}")

    # Pre-load model on startup
    logger.info("Pre-loading model...")
    try:
        generator.load_model()
    except Exception as e:
        logger.error(f"Failed to pre-load model: {e}")
        logger.warning("Model will be loaded on first generation attempt")

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
    )


if __name__ == "__main__":
    main()

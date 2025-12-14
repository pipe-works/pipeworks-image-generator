"""Plugin for saving generation metadata to text and JSON files."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from pipeworks.plugins.base import PluginBase, plugin_registry

logger = logging.getLogger(__name__)


class SaveMetadataPlugin(PluginBase):
    """
    Save generation metadata to .txt and .json files.

    This plugin saves:
    - .txt file with the prompt
    - .json file with all generation parameters

    Configuration:
        folder_name: Custom folder name within outputs directory (optional)
        filename_prefix: Prefix for generated files (optional)
    """

    name = "SaveMetadata"
    description = "Save prompt and parameters to .txt and .json files"
    version = "0.1.0"

    def __init__(self, **config):
        super().__init__(**config)
        self.folder_name = config.get("folder_name", None)
        self.filename_prefix = config.get("filename_prefix", "")

    def on_after_save(
        self, image: Image.Image, save_path: Path, params: Dict[str, Any]
    ) -> None:
        """
        Save metadata files after the image has been saved.

        Args:
            image: The saved image
            save_path: Path where the image was saved
            params: Generation parameters
        """
        if not self.enabled:
            return

        try:
            # Determine the output directory and final image path
            if self.folder_name:
                # Use custom folder within the image's parent directory
                output_dir = save_path.parent / self.folder_name
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save image to the metadata subfolder
                new_image_path = output_dir / save_path.name
                image.save(new_image_path)
                logger.info(f"Saved image to: {new_image_path}")

                # Use the new path for metadata reference
                final_image_path = new_image_path
            else:
                # Use the same directory as the image
                output_dir = save_path.parent
                final_image_path = save_path

            # Generate base filename
            base_name = final_image_path.stem  # Image filename without extension
            if self.filename_prefix:
                base_name = f"{self.filename_prefix}_{base_name}"

            # Save prompt to .txt file
            txt_path = output_dir / f"{base_name}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(params.get("prompt", ""))

            logger.info(f"Saved prompt to: {txt_path}")

            # Prepare metadata for JSON
            metadata = {
                "prompt": params.get("prompt", ""),
                "width": params.get("width"),
                "height": params.get("height"),
                "num_inference_steps": params.get("num_inference_steps"),
                "seed": params.get("seed"),
                "guidance_scale": params.get("guidance_scale"),
                "model_id": params.get("model_id"),
                "timestamp": datetime.now().isoformat(),
                "image_path": str(final_image_path),
            }

            # Add any additional params
            for key, value in params.items():
                if key not in metadata:
                    metadata[key] = value

            # Save metadata to .json file
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved metadata to: {json_path}")

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}", exc_info=True)


# Register the plugin
plugin_registry.register(SaveMetadataPlugin)

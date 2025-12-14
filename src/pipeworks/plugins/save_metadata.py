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

    def on_before_save(
        self, image: Image.Image, save_path: Path, params: Dict[str, Any]
    ) -> tuple[Image.Image, Path]:
        """
        Modify the save path to use the metadata folder if configured.

        Args:
            image: Image to be saved
            save_path: Proposed save path
            params: Generation parameters

        Returns:
            Tuple of (image, modified save path)
        """
        if not self.enabled or not self.folder_name:
            return image, save_path

        # Redirect save path to metadata subfolder
        output_dir = save_path.parent / self.folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        new_save_path = output_dir / save_path.name

        logger.info(f"Redirecting save to: {new_save_path}")
        return image, new_save_path

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
            # Use the same directory as the saved image
            output_dir = save_path.parent

            # Generate base filename
            base_name = save_path.stem  # Image filename without extension
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
                "image_path": str(save_path),
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

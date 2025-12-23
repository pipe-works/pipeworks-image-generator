"""Data models for Pipeworks UI state and parameters."""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SegmentConfig:
    """Configuration for a single prompt segment (start/middle/end).

    This represents the state of one segment in the prompt builder,
    including text input, file selection, and mode settings.
    """

    text: str = ""
    path: str = ""
    file: str = "(None)"
    mode: str = "Random Line"
    line: int = 1
    range_end: int = 1
    count: int = 1
    dynamic: bool = False
    sequential_start_line: int = 1  # Starting line for Sequential mode

    # Character/Facial condition generation (for Start segments)
    condition_type: str = "None"  # "None", "Character", "Facial", or "Both"
    condition_text: str = ""  # Generated condition text
    condition_dynamic: bool = False  # Regenerate condition per image

    def is_configured(self) -> bool:
        """Check if segment has a valid file selected.

        Returns:
            True if a file is selected and it's not a folder or placeholder
        """
        return bool(self.file and self.file != "(None)" and not self.file.startswith("📁"))

    def has_content(self) -> bool:
        """Check if segment has any content (text or file).

        Returns:
            True if segment has text or a configured file
        """
        return bool(self.text and self.text.strip()) or self.is_configured()


@dataclass
class GenerationParams:
    """Parameters for image generation.

    This dataclass encapsulates all the parameters needed for generating
    images, with built-in validation logic.
    """

    prompt: str
    width: int
    height: int
    num_steps: int
    batch_size: int
    runs: int
    seed: int
    use_random_seed: bool

    def validate(self) -> None:
        """Validate generation parameters.

        Raises:
            ValueError: If any parameter is invalid, with descriptive message
        """
        # Validate batch size
        if self.batch_size < 1 or self.batch_size > 100:
            raise ValueError(f"Batch size must be 1-100, got {self.batch_size}")

        # Validate runs
        if self.runs < 1 or self.runs > 100:
            raise ValueError(f"Runs must be 1-100, got {self.runs}")

        # Validate total images
        total = self.batch_size * self.runs
        if total > 1000:
            raise ValueError(
                f"Total images ({total}) exceeds maximum of 1000. " f"Reduce batch size or runs."
            )

        # Validate dimensions are multiples of 64
        if self.width % 64 != 0:
            raise ValueError(f"Width must be multiple of 64, got {self.width}")
        if self.height % 64 != 0:
            raise ValueError(f"Height must be multiple of 64, got {self.height}")

        # Validate dimension ranges
        if self.width < 512 or self.width > 2048:
            raise ValueError(f"Width must be 512-2048, got {self.width}")
        if self.height < 512 or self.height > 2048:
            raise ValueError(f"Height must be 512-2048, got {self.height}")

        # Validate inference steps
        if self.num_steps < 1 or self.num_steps > 50:
            raise ValueError(f"Inference steps must be 1-50, got {self.num_steps}")

        # Validate seed
        if self.seed < 0 or self.seed > 2**32 - 1:
            raise ValueError(f"Seed must be 0 to {2**32 - 1}, got {self.seed}")

    @property
    def total_images(self) -> int:
        """Calculate total number of images to generate."""
        return self.batch_size * self.runs


@dataclass
class UIState:
    """Session state for the Gradio UI.

    This represents all the stateful objects that need to be maintained
    per user session. Each user gets their own UIState instance to ensure
    thread safety and isolation.

    Attributes
    ----------
    model_adapter : Any | None
        Current model adapter instance (ModelAdapterBase)
    current_model_name : str
        Name of the currently selected model adapter
    tokenizer_analyzer : Any | None
        TokenizerAnalyzer instance for prompt analysis
    prompt_builder : Any | None
        PromptBuilder instance for file-based prompts
    active_plugins : dict[str, Any]
        Dictionary of active plugin instances
    gallery_browser : Any | None
        GalleryBrowser instance for image browsing
    favorites_db : Any | None
        FavoritesDB instance for tracking favorites
    catalog_manager : Any | None
        CatalogManager instance for archiving favorites
    """

    # Model adapter
    model_adapter: Any | None = None  # ModelAdapterBase instance
    current_model_name: str = "Z-Image-Turbo"  # Currently selected model

    # Core components
    tokenizer_analyzer: Any | None = None  # TokenizerAnalyzer instance
    prompt_builder: Any | None = None  # PromptBuilder instance
    active_plugins: dict[str, Any] = field(default_factory=dict)  # Dict[str, PluginBase]

    # Gallery browser state
    gallery_browser: Any | None = None  # GalleryBrowser instance
    gallery_current_path: str = ""  # Current subfolder in outputs/
    gallery_images: list[str] = field(default_factory=list)  # Cached image list
    gallery_selected_index: int | None = None  # Currently selected image index
    gallery_initialized: bool = False  # Track if gallery has been initialized

    # Favorites and catalog state
    favorites_db: Any | None = None  # FavoritesDB instance
    catalog_manager: Any | None = None  # CatalogManager instance
    gallery_filter: str = "all"  # "all" or "favorites"
    gallery_root: str = "outputs"  # Current root: "outputs" or "catalog"

    def is_initialized(self) -> bool:
        """Check if the state has been initialized with core components.

        Returns:
            True if model_adapter, tokenizer, and prompt builder are loaded
        """
        return (
            self.model_adapter is not None
            and self.tokenizer_analyzer is not None
            and self.prompt_builder is not None
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UIState(initialized={self.is_initialized()}, "
            f"model={self.current_model_name}, "
            f"plugins={len(self.active_plugins)})"
        )


# Constants for UI
SEGMENT_MODES = [
    "Random Line",
    "Specific Line",
    "Line Range",
    "All Lines",
    "Random Multiple",
    "Sequential",
]

ASPECT_RATIOS = {
    "Square 1:1 (1024x1024)": (1024, 1024),
    "Widescreen 16:9 (1280x720)": (1280, 720),
    "Widescreen 16:9 (1600x896)": (1600, 896),
    "Portrait 9:16 (720x1280)": (720, 1280),
    "Portrait 9:16 (896x1600)": (896, 1600),
    "Standard 3:2 (1280x832)": (1280, 832),
    "Standard 2:3 (832x1280)": (832, 1280),
    "Standard 3:2 (1536x1024)": (1536, 1024),
    "Custom": None,  # Will use config defaults
}

# UI Constants
MAX_SEED = 2**32 - 1
DEFAULT_SEED = 42

# Condition generation types
CONDITION_TYPES = [
    "None",  # No condition generation
    "Character",  # Generate character conditions (physique, wealth, etc.)
    "Facial",  # Generate facial signal conditions
    "Both",  # Generate both character and facial conditions
]

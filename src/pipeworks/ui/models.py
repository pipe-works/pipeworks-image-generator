"""Data models for Pipeworks UI state and parameters."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

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
        return bool(self.text.strip()) or self.is_configured()


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
                f"Total images ({total}) exceeds maximum of 1000. "
                f"Reduce batch size or runs."
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
    """
    generator: Optional[Any] = None  # ImageGenerator instance
    tokenizer_analyzer: Optional[Any] = None  # TokenizerAnalyzer instance
    prompt_builder: Optional[Any] = None  # PromptBuilder instance
    active_plugins: Dict[str, Any] = field(default_factory=dict)  # Dict[str, PluginBase]

    def is_initialized(self) -> bool:
        """Check if the state has been initialized with core components.

        Returns:
            True if generator, tokenizer, and prompt builder are loaded
        """
        return (
            self.generator is not None
            and self.tokenizer_analyzer is not None
            and self.prompt_builder is not None
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UIState(initialized={self.is_initialized()}, "
            f"plugins={len(self.active_plugins)})"
        )


# Constants for UI
SEGMENT_MODES = [
    "Random Line",
    "Specific Line",
    "Line Range",
    "All Lines",
    "Random Multiple"
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

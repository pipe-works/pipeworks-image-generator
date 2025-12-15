"""Validation utilities for Pipeworks UI inputs."""

from pathlib import Path
from typing import Tuple
import logging

from .models import GenerationParams, SegmentConfig

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """User-friendly validation error.

    This exception is raised when user input fails validation.
    The message is intended to be displayed directly to the user.
    """
    pass


def validate_generation_params(params: GenerationParams) -> None:
    """Validate generation parameters with user-friendly messages.

    Args:
        params: Generation parameters to validate

    Raises:
        ValidationError: If validation fails with user-friendly message
    """
    try:
        params.validate()
    except ValueError as e:
        raise ValidationError(str(e)) from e


def validate_segment_path(path: str, file: str, base_dir: Path) -> Path:
    """Validate that a segment file path is safe and exists.

    This function ensures:
    1. A file is actually selected (not placeholder or folder)
    2. The path is within the base directory (security)
    3. The file exists on disk

    Args:
        path: Relative path from base_dir
        file: Filename
        base_dir: Base directory (inputs_dir)

    Returns:
        Absolute resolved path if valid

    Raises:
        ValidationError: If path is invalid, unsafe, or file doesn't exist
    """
    # Check if file is selected
    if not file or file == "(None)":
        raise ValidationError("No file selected")

    # Check if it's a folder (not a file)
    if file.startswith("📁"):
        raise ValidationError("Selected item is a folder, not a file")

    # Build full path
    try:
        if path:
            full_path = (base_dir / path / file).resolve()
        else:
            full_path = (base_dir / file).resolve()
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid path: {e}") from e

    # Security: Ensure path is within base_dir (prevent path traversal attacks)
    try:
        base_resolved = base_dir.resolve()
        if not str(full_path).startswith(str(base_resolved)):
            logger.warning(f"Path traversal attempt detected: {full_path}")
            raise ValidationError("Invalid path: outside of inputs directory")
    except (ValueError, OSError) as e:
        raise ValidationError(f"Error validating path: {e}") from e

    # Check file exists
    if not full_path.exists():
        raise ValidationError(f"File not found: {file}")

    # Check it's actually a file, not a directory
    if not full_path.is_file():
        raise ValidationError(f"Path is not a file: {file}")

    return full_path


def validate_segments(
    segments: Tuple[SegmentConfig, SegmentConfig, SegmentConfig],
    base_dir: Path,
    prompt: str
) -> None:
    """Validate all segment configurations.

    Ensures that either:
    - A static prompt is provided, OR
    - At least one segment has content (text or file)

    Also validates that any configured file segments point to valid files.

    Args:
        segments: Tuple of (start, middle, end) segment configurations
        base_dir: Base directory for resolving file paths
        prompt: Static prompt text

    Raises:
        ValidationError: If validation fails with user-friendly message
    """
    start, middle, end = segments
    segment_names = ["Start", "Middle", "End"]

    # Check if any segment has dynamic flag set
    has_dynamic = any(seg.dynamic for seg in segments)

    # Check if any segment has content
    has_segment_content = any(seg.has_content() for seg in segments)

    # Validate that we have some input
    if not has_segment_content and (not prompt or not prompt.strip()):
        raise ValidationError(
            "Please provide either a prompt or configure at least one segment "
            "in the Prompt Builder"
        )

    # Validate each configured segment's file path
    for i, segment in enumerate(segments):
        if segment.is_configured():
            try:
                validate_segment_path(segment.path, segment.file, base_dir)
            except ValidationError as e:
                segment_name = segment_names[i]
                raise ValidationError(f"{segment_name} segment: {e}") from e

    # If dynamic segments are used, ensure at least one segment is configured
    if has_dynamic:
        has_configured_dynamic = any(
            seg.dynamic and seg.is_configured() for seg in segments
        )
        if not has_configured_dynamic:
            raise ValidationError(
                "Dynamic mode is enabled but no segments have files configured. "
                "Please select files for segments marked as dynamic."
            )


def validate_prompt_content(prompt: str, max_length: int = 1000) -> None:
    """Validate prompt text content.

    Args:
        prompt: Prompt text to validate
        max_length: Maximum allowed prompt length

    Raises:
        ValidationError: If prompt is too long or contains invalid content
    """
    if len(prompt) > max_length:
        raise ValidationError(
            f"Prompt is too long ({len(prompt)} characters). "
            f"Maximum is {max_length} characters."
        )

    # Check for obviously invalid prompts
    if prompt.strip().lower() in ["error:", "error", "none", "null"]:
        raise ValidationError("Invalid prompt content")


def sanitize_filename_input(text: str) -> str:
    """Sanitize user input for use in filenames.

    Args:
        text: User input text

    Returns:
        Sanitized text safe for filenames
    """
    # Remove potentially problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '_')

    # Limit length
    return text[:100]

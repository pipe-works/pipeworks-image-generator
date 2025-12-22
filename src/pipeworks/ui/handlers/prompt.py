"""Prompt builder and file navigation handlers."""

import logging
from pathlib import Path

import gradio as gr

from ..models import SegmentConfig, UIState
from ..state import initialize_ui_state
from ..validation import ValidationError

logger = logging.getLogger(__name__)


def get_items_in_path(current_path: str, state: UIState) -> tuple[gr.Dropdown, str, UIState]:
    """Get folders and files at the current path level.

    Args:
        current_path: Current path being browsed (empty string for root)
        state: UI state (contains prompt_builder)

    Returns:
        Tuple of (updated_dropdown, display_path, updated_state)
    """
    # Initialize state if needed
    state = initialize_ui_state(state)

    folders, files = state.prompt_builder.get_items_in_path(current_path)

    # Build choices list: folders first (with prefix), then files
    choices = ["(None)"]

    # Add ".." to go up a level if not at root
    if current_path:
        choices.append("📁 ..")

    # Add folders with folder emoji
    for folder in folders:
        choices.append(f"📁 {folder}")

    # Add files
    choices.extend(files)

    # Display path for user reference
    display_path = f"/{current_path}" if current_path else "/inputs"

    return gr.update(choices=choices, value="(None)"), display_path, state


def navigate_file_selection(
    selected: str, current_path: str, state: UIState
) -> tuple[gr.Dropdown, str, UIState]:
    """Handle folder navigation when an item is selected.

    Args:
        selected: The selected item from dropdown
        current_path: Current path being browsed
        state: UI state

    Returns:
        Tuple of (updated_dropdown, new_path, updated_state)
    """
    # If (None) selected, do nothing
    if selected == "(None)":
        dropdown, display, state = get_items_in_path(current_path, state)
        return dropdown, current_path, state

    # Check if it's a folder (starts with folder emoji)
    if selected.startswith("📁 "):
        folder_name = selected[2:].strip()  # Remove emoji and whitespace

        if folder_name == "..":
            # Go up one level
            if current_path:
                new_path = str(Path(current_path).parent)
                if new_path == ".":
                    new_path = ""
            else:
                new_path = ""
        else:
            # Navigate into folder
            new_path = str(Path(current_path) / folder_name) if current_path else folder_name

        # Update dropdown with new path contents
        dropdown, display, state = get_items_in_path(new_path, state)
        return dropdown, new_path, state
    else:
        # It's a file - keep it selected but don't navigate
        return gr.update(), current_path, state


def build_combined_prompt(
    start_1: SegmentConfig,
    start_2: SegmentConfig,
    start_3: SegmentConfig,
    mid_1: SegmentConfig,
    mid_2: SegmentConfig,
    mid_3: SegmentConfig,
    end_1: SegmentConfig,
    end_2: SegmentConfig,
    end_3: SegmentConfig,
    state: UIState,
    run_index: int = 0,
) -> str:
    """Build a combined prompt from multiple segments.

    Args:
        start_1: Start segment 1 configuration
        start_2: Start segment 2 configuration
        start_3: Start segment 3 configuration
        mid_1: Mid segment 1 configuration
        mid_2: Mid segment 2 configuration
        mid_3: Mid segment 3 configuration
        end_1: End segment 1 configuration
        end_2: End segment 2 configuration
        end_3: End segment 3 configuration
        state: UI state (contains prompt_builder)
        run_index: Zero-indexed run number (for Sequential mode)

    Returns:
        Combined prompt string or error message
    """
    # Initialize state if needed
    state = initialize_ui_state(state)

    segments = []

    # Helper to add segment
    def add_segment(segment: SegmentConfig):
        # Add user text if provided
        if segment.text and segment.text.strip():
            segments.append(("text", segment.text.strip()))

        # Add file selection if configured
        if segment.is_configured():
            # Get full file path
            full_path = state.prompt_builder.get_full_path(segment.path, segment.file)

            if segment.mode == "Random Line":
                segments.append(("file_random", full_path))
            elif segment.mode == "Specific Line":
                segments.append(("file_specific", f"{full_path}|{segment.line}"))
            elif segment.mode == "Line Range":
                segments.append(("file_range", f"{full_path}|{segment.line}|{segment.range_end}"))
            elif segment.mode == "All Lines":
                segments.append(("file_all", full_path))
            elif segment.mode == "Random Multiple":
                segments.append(("file_random_multi", f"{full_path}|{segment.count}"))
            elif segment.mode == "Sequential":
                segments.append(
                    (
                        "file_sequential",
                        f"{full_path}|{segment.sequential_start_line}|{run_index}",
                    )
                )

    # Add segments in order (Start 1-3, Mid 1-3, End 1-3)
    add_segment(start_1)
    add_segment(start_2)
    add_segment(start_3)
    add_segment(mid_1)
    add_segment(mid_2)
    add_segment(mid_3)
    add_segment(end_1)
    add_segment(end_2)
    add_segment(end_3)

    # Build the final prompt
    try:
        result = state.prompt_builder.build_prompt(segments)
        return result if result else ""
    except Exception as e:
        logger.error(f"Error building prompt: {e}", exc_info=True)
        # Don't return error as prompt - raise it
        raise ValidationError(f"Failed to build prompt: {str(e)}")

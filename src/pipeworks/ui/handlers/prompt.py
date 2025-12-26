"""Prompt builder and file navigation handlers."""

import logging
from pathlib import Path
from typing import Any

import gradio as gr

from ..models import SegmentConfig, UIState
from ..state import initialize_ui_state
from ..validation import ValidationError

logger = logging.getLogger(__name__)


def get_items_in_path(
    current_path: str, state: UIState
) -> tuple[dict[str, Any], str, dict[str, Any], UIState]:
    """Get folders and files at the current path level.

    Args:
        current_path: Current path being browsed (empty string for root)
        state: UI state (contains prompt_builder)

    Returns:
        Tuple of (updated_dropdown, display_path, line_count_update, updated_state)
    """
    # Initialize state if needed
    state = initialize_ui_state(state)

    # Check if prompt_builder is available
    if state.prompt_builder is None:
        return gr.update(), current_path, gr.update(), state

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

    # Hide line count when browsing folders
    line_count_update = gr.update(value="", visible=False)

    return gr.update(choices=choices, value="(None)"), display_path, line_count_update, state


def navigate_file_selection(
    selected: str, current_path: str, state: UIState
) -> tuple[dict[str, Any], str, dict[str, Any], UIState]:
    """Handle folder navigation when an item is selected.

    Args:
        selected: The selected item from dropdown
        current_path: Current path being browsed
        state: UI state

    Returns:
        Tuple of (updated_dropdown, new_path, line_count_update, updated_state)
    """
    # Initialize state if needed
    state = initialize_ui_state(state)

    # If (None) selected, do nothing
    if selected == "(None)":
        dropdown, display, line_count, state = get_items_in_path(current_path, state)
        return dropdown, current_path, line_count, state

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
        dropdown, display, line_count, state = get_items_in_path(new_path, state)
        return dropdown, new_path, line_count, state
    else:
        # It's a file - display line count
        if state.prompt_builder is None:
            return gr.update(), current_path, gr.update(), state

        # Get full path to the file
        full_path = state.prompt_builder.get_full_path(current_path, selected)

        # Get file info (includes line count)
        file_info = state.prompt_builder.get_file_info(full_path)

        if file_info["exists"]:
            line_count = file_info["line_count"]
            line_count_text = f"**Lines:** {line_count}"
            line_count_update = gr.update(value=line_count_text, visible=True)
        else:
            line_count_update = gr.update(value="**File not found**", visible=True)

        return gr.update(), current_path, line_count_update, state


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

    # Check if prompt_builder is available
    if state.prompt_builder is None:
        raise ValidationError("Prompt builder not initialized")

    segments = []

    # Helper to add segment
    def add_segment(segment: SegmentConfig):
        """Add a segment to the prompt, respecting text_order and delimiter.

        The segment's delimiter is appended at the END of the segment's content,
        not used to join with other segments.

        Handles four scenarios:
        1. No content: Skip
        2. Only text: Add text directly with delimiter appended
        3. Only file: Add file as lazy tuple with delimiter info
        4. Both text and file: Resolve file content early, combine with text using
           delimiter and text_order, add as single text segment with delimiter appended

        Args:
            segment: SegmentConfig with text, file, and settings
        """
        has_text = segment.text and segment.text.strip()
        has_file = segment.is_configured()

        # Get the actual delimiter value from the label
        delimiter_value = segment.get_delimiter_value()

        # Case 1: No content - skip
        if not has_text and not has_file:
            return

        # Case 2: Only text - add directly with delimiter appended at end
        if has_text and not has_file:
            content_with_delimiter = segment.text.strip() + delimiter_value
            segments.append(("text", content_with_delimiter))
            return

        # Case 3: Only file - resolve file content and append delimiter at end
        if has_file and not has_text:
            if state.prompt_builder is None:
                return
            full_path = state.prompt_builder.get_full_path(segment.path, segment.file)

            # Resolve file content based on mode
            file_content = ""
            if segment.mode == "Random Line":
                file_content = state.prompt_builder.get_random_line(full_path)
            elif segment.mode == "Specific Line":
                file_content = state.prompt_builder.get_specific_line(full_path, segment.line)
            elif segment.mode == "Line Range":
                file_content = state.prompt_builder.get_line_range(
                    full_path, segment.line, segment.range_end, delimiter=delimiter_value
                )
            elif segment.mode == "All Lines":
                file_content = state.prompt_builder.get_all_lines(
                    full_path, delimiter=delimiter_value
                )
            elif segment.mode == "Random Multiple":
                file_content = state.prompt_builder.get_random_lines(
                    full_path, segment.count, delimiter=delimiter_value
                )
            elif segment.mode == "Sequential":
                file_content = state.prompt_builder.get_sequential_line(
                    full_path, segment.sequential_start_line, run_index
                )

            # If file read failed, skip this segment
            if not file_content:
                return

            # Append delimiter at the END of the file content
            content_with_delimiter = file_content + delimiter_value
            segments.append(("text", content_with_delimiter))
            return

        # Case 4: Both text and file - resolve file early and combine
        # Append delimiter at the END of the combined content
        if has_text and has_file:
            if state.prompt_builder is None:
                content_with_delimiter = segment.text.strip() + delimiter_value
                segments.append(("text", content_with_delimiter))
                return
            full_path = state.prompt_builder.get_full_path(segment.path, segment.file)

            # Resolve file content based on mode
            # Use delimiter_value for joining multiple lines within files
            file_content = ""
            if segment.mode == "Random Line":
                file_content = state.prompt_builder.get_random_line(full_path)
            elif segment.mode == "Specific Line":
                file_content = state.prompt_builder.get_specific_line(full_path, segment.line)
            elif segment.mode == "Line Range":
                file_content = state.prompt_builder.get_line_range(
                    full_path, segment.line, segment.range_end, delimiter=delimiter_value
                )
            elif segment.mode == "All Lines":
                file_content = state.prompt_builder.get_all_lines(
                    full_path, delimiter=delimiter_value
                )
            elif segment.mode == "Random Multiple":
                file_content = state.prompt_builder.get_random_lines(
                    full_path, segment.count, delimiter=delimiter_value
                )
            elif segment.mode == "Sequential":
                file_content = state.prompt_builder.get_sequential_line(
                    full_path, segment.sequential_start_line, run_index
                )

            # If file read failed, fall back to text only
            if not file_content:
                content_with_delimiter = segment.text.strip() + delimiter_value
                segments.append(("text", content_with_delimiter))
                return

            # Combine text and file based on text_order
            # The delimiter is placed ONLY at the end of the segment, not between text and file
            if segment.text_order == "text_first":
                combined = f"{segment.text.strip()} {file_content}"
            else:  # file_first
                combined = f"{file_content} {segment.text.strip()}"

            # Append delimiter at the END of the combined segment
            combined_with_end_delimiter = combined + delimiter_value
            segments.append(("text", combined_with_end_delimiter))

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
    # NOTE: Each segment has delimiter appended, pass delimiter="" to concatenate
    try:
        result = state.prompt_builder.build_prompt(segments, delimiter="")
        return result if result else ""
    except Exception as e:
        logger.error(f"Error building prompt: {e}", exc_info=True)
        # Don't return error as prompt - raise it
        raise ValidationError(f"Failed to build prompt: {str(e)}")

"""Reusable UI components for Pipeworks Gradio interface."""

import gradio as gr

from .models import SEGMENT_MODES, SegmentConfig


class SegmentUI:
    """Reusable UI component for a prompt segment.

    This component encapsulates all the UI elements for a single segment
    (Start, Middle, or End) in the prompt builder. This eliminates the
    code duplication that existed with three separate blocks of identical code.

    Each segment has:
    - Title (with status indicator)
    - Text input
    - File browser (with hierarchical navigation)
    - Mode selector
    - Dynamic checkbox
    - Mode-specific inputs (line numbers, ranges, counts)
    """

    def __init__(self, name: str, initial_choices: list[str]):
        """Initialize a segment UI component.

        Args:
            name: Name of the segment (e.g., "Start", "Middle", "End")
            initial_choices: Initial file/folder choices for dropdown
        """
        self.name = name

        with gr.Group():
            # Title with status indicator (updated via event handlers)
            self.title = gr.Markdown(f"**{name} Segment**")

            # Text input for manual text entry
            self.text = gr.Textbox(label=f"{name} Text", placeholder="Optional text...", lines=1)

            # Current path display (shows where user is in folder hierarchy)
            self.path_display = gr.Textbox(label="Current Path", value="/inputs", interactive=False)

            # File/folder browser dropdown
            self.file = gr.Dropdown(
                label="File/Folder Browser", choices=initial_choices, value="(None)"
            )

            # Hidden state to track current navigation path
            self.path_state = gr.State(value="")

            # Mode and dynamic options
            with gr.Row():
                self.mode = gr.Dropdown(label="Mode", choices=SEGMENT_MODES, value="Random Line")
                self.dynamic = gr.Checkbox(
                    label="Dynamic", value=False, info="Rebuild this segment for each image"
                )

            # Mode-specific inputs (visibility controlled by mode selection)
            with gr.Row():
                self.line = gr.Number(
                    label="Line #", value=1, minimum=1, precision=0, visible=False
                )
                self.range_end = gr.Number(
                    label="End Line #", value=1, minimum=1, precision=0, visible=False
                )
                self.count = gr.Number(
                    label="Count", value=1, minimum=1, maximum=10, precision=0, visible=False
                )
                self.sequential_start_line = gr.Number(
                    label="Start Line #",
                    value=1,
                    minimum=1,
                    precision=0,
                    visible=False,
                    info="Starting line for sequential mode",
                )

    def get_all_components(self) -> list[gr.components.Component]:
        """Return all Gradio components in this segment.

        Returns:
            List of all components (for bulk operations)
        """
        return [
            self.title,
            self.text,
            self.path_display,
            self.file,
            self.path_state,
            self.mode,
            self.dynamic,
            self.line,
            self.range_end,
            self.count,
            self.sequential_start_line,
        ]

    def get_input_components(self) -> list[gr.components.Component]:
        """Return components used as function inputs.

        Returns:
            List of components that should be passed as inputs to handlers
        """
        return [
            self.text,
            self.path_state,
            self.file,
            self.mode,
            self.line,
            self.range_end,
            self.count,
            self.dynamic,
            self.sequential_start_line,
        ]

    def get_output_components(self) -> list[gr.components.Component]:
        """Return components used as function outputs.

        Returns:
            List of components that can be updated by handlers
        """
        return [self.title, self.path_display, self.file, self.path_state]

    def get_navigation_components(self) -> tuple[gr.Dropdown, gr.State, gr.Textbox]:
        """Return components needed for file browser navigation.

        Returns:
            Tuple of (file_dropdown, path_state, path_display)
        """
        return self.file, self.path_state, self.path_display

    def get_mode_visibility_outputs(
        self,
    ) -> tuple[gr.Number, gr.Number, gr.Number, gr.Number]:
        """Return components that change visibility based on mode.

        Returns:
            Tuple of (line, range_end, count, sequential_start_line) number inputs
        """
        return self.line, self.range_end, self.count, self.sequential_start_line

    @staticmethod
    def values_to_config(
        text: str,
        path: str,
        file: str,
        mode: str,
        line: int,
        range_end: int,
        count: int,
        dynamic: bool,
        sequential_start_line: int,
    ) -> SegmentConfig:
        """Convert UI component values to SegmentConfig dataclass.

        Args:
            text: Text input value
            path: Current path state
            file: Selected file
            mode: Selected mode
            line: Line number input
            range_end: Range end input
            count: Count input
            dynamic: Dynamic checkbox state
            sequential_start_line: Starting line for Sequential mode

        Returns:
            SegmentConfig instance with all values
        """
        return SegmentConfig(
            text=text,
            path=path,
            file=file,
            mode=mode,
            line=int(line) if line else 1,
            range_end=int(range_end) if range_end else 1,
            count=int(count) if count else 1,
            dynamic=dynamic,
            sequential_start_line=int(sequential_start_line) if sequential_start_line else 1,
        )

    @staticmethod
    def format_title(name: str, file: str, mode: str, dynamic: bool) -> str:
        """Format segment title with status indicators.

        Args:
            name: Segment name
            file: Selected file
            mode: Selected mode
            dynamic: Whether dynamic mode is enabled

        Returns:
            Formatted HTML string for title
        """
        # Check if segment is configured (file is selected and not a folder)
        has_config = file and file != "(None)" and not file.startswith("📁")

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


def update_mode_visibility(mode: str) -> tuple[gr.Number, gr.Number, gr.Number, gr.Number]:
    """Update visibility of line number inputs based on selected mode.

    Args:
        mode: Selected mode (e.g., "Random Line", "Specific Line", "Sequential")

    Returns:
        Tuple of gr.update() calls for (line, range_end, count, sequential_start_line)
    """
    return (
        gr.update(visible=mode in ["Specific Line", "Line Range"]),
        gr.update(visible=mode == "Line Range"),
        gr.update(visible=mode == "Random Multiple"),
        gr.update(visible=mode == "Sequential"),
    )


def create_three_segments(initial_choices: list[str]) -> tuple[SegmentUI, SegmentUI, SegmentUI]:
    """Create the three segment UI components (Start, Middle, End).

    Args:
        initial_choices: Initial file/folder choices for dropdowns

    Returns:
        Tuple of (start_segment, middle_segment, end_segment)
    """
    start = SegmentUI("Start", initial_choices)
    middle = SegmentUI("Middle", initial_choices)
    end = SegmentUI("End", initial_choices)

    return start, middle, end


def create_nine_segments(
    initial_choices: list[str],
) -> tuple[
    SegmentUI,
    SegmentUI,
    SegmentUI,
    SegmentUI,
    SegmentUI,
    SegmentUI,
    SegmentUI,
    SegmentUI,
    SegmentUI,
]:
    """Create nine segment UI components arranged in 3x3 grid.

    Args:
        initial_choices: Initial file/folder choices for dropdowns

    Returns:
        Tuple of 9 segments: (start_1, start_2, start_3, mid_1, mid_2, mid_3, end_1, end_2, end_3)
    """
    start_1 = SegmentUI("Start 1", initial_choices)
    start_2 = SegmentUI("Start 2", initial_choices)
    start_3 = SegmentUI("Start 3", initial_choices)

    mid_1 = SegmentUI("Mid 1", initial_choices)
    mid_2 = SegmentUI("Mid 2", initial_choices)
    mid_3 = SegmentUI("Mid 3", initial_choices)

    end_1 = SegmentUI("End 1", initial_choices)
    end_2 = SegmentUI("End 2", initial_choices)
    end_3 = SegmentUI("End 3", initial_choices)

    return start_1, start_2, start_3, mid_1, mid_2, mid_3, end_1, end_2, end_3

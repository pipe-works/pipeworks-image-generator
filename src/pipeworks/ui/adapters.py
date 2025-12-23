"""Adapter functions for converting between UI values and business objects."""

from .components import SegmentUI
from .models import SegmentConfig


def convert_segment_values_to_configs(
    start_values: tuple,
    middle_values: tuple,
    end_values: tuple,
) -> tuple[SegmentConfig, SegmentConfig, SegmentConfig]:
    """Convert raw UI segment values to SegmentConfig objects.

    Args:
        start_values: 9-tuple of start segment values
            (text, path, file, mode, line, range_end, count, dynamic, sequential_start_line)
        middle_values: 9-tuple of middle segment values
        end_values: 9-tuple of end segment values

    Returns:
        Tuple of (start_cfg, middle_cfg, end_cfg)
    """
    return (
        SegmentUI.values_to_config(*start_values),
        SegmentUI.values_to_config(*middle_values),
        SegmentUI.values_to_config(*end_values),
    )


def split_segment_inputs(values: list) -> tuple[tuple, tuple, tuple, any]:
    """Split combined input list into segment groups.

    Args:
        values: List of all UI input values

    Returns:
        Tuple of (start_values, middle_values, end_values, state)
        Each segment values is a 9-tuple
            (text, path, file, mode, line, range_end, count, dynamic, sequential_start_line)
    """
    start_values = tuple(values[0:9])
    middle_values = tuple(values[9:18])
    end_values = tuple(values[18:27])
    state = values[27]
    return start_values, middle_values, end_values, state

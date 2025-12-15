"""Prompt builder utility for constructing prompts from text files and user input."""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build prompts by combining text files and user input."""

    def __init__(self, inputs_dir: Path):
        """
        Initialize the prompt builder.

        Args:
            inputs_dir: Directory containing input text files
        """
        self.inputs_dir = Path(inputs_dir)
        self._file_cache = {}  # Cache file contents

    def scan_text_files(self) -> List[str]:
        """
        Scan inputs directory for .txt files.

        Returns:
            List of relative file paths (including subdirectories)
        """
        if not self.inputs_dir.exists():
            logger.warning(f"Inputs directory does not exist: {self.inputs_dir}")
            return []

        txt_files = []
        for txt_file in self.inputs_dir.rglob("*.txt"):
            # Get relative path from inputs_dir
            relative_path = txt_file.relative_to(self.inputs_dir)
            txt_files.append(str(relative_path))

        return sorted(txt_files)

    def read_file_lines(self, file_path: str) -> List[str]:
        """
        Read lines from a text file.

        Args:
            file_path: Relative path from inputs_dir

        Returns:
            List of non-empty lines (stripped of whitespace)
        """
        # Check cache first
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        full_path = self.inputs_dir / file_path
        if not full_path.exists():
            logger.error(f"File not found: {full_path}")
            return []

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # Cache the result
            self._file_cache[file_path] = lines
            return lines

        except Exception as e:
            logger.error(f"Error reading file {full_path}: {e}")
            return []

    def get_random_line(self, file_path: str) -> str:
        """
        Get a random line from a text file.

        Args:
            file_path: Relative path from inputs_dir

        Returns:
            Random line from the file, or empty string if file is empty/not found
        """
        lines = self.read_file_lines(file_path)
        if not lines:
            return ""
        return random.choice(lines)

    def get_specific_line(self, file_path: str, line_number: int) -> str:
        """
        Get a specific line from a text file (1-indexed).

        Args:
            file_path: Relative path from inputs_dir
            line_number: Line number (1-indexed)

        Returns:
            The specified line, or empty string if out of range
        """
        lines = self.read_file_lines(file_path)
        if not lines or line_number < 1 or line_number > len(lines):
            return ""
        return lines[line_number - 1]

    def get_line_range(self, file_path: str, start: int, end: int) -> str:
        """
        Get a range of lines from a text file (1-indexed, inclusive).

        Args:
            file_path: Relative path from inputs_dir
            start: Start line number (1-indexed)
            end: End line number (1-indexed, inclusive)

        Returns:
            Lines joined with commas, or empty string if range is invalid
        """
        lines = self.read_file_lines(file_path)
        if not lines:
            return ""

        # Clamp to valid range
        start = max(1, min(start, len(lines)))
        end = max(start, min(end, len(lines)))

        selected_lines = lines[start - 1 : end]
        return ", ".join(selected_lines)

    def get_all_lines(self, file_path: str) -> str:
        """
        Get all lines from a text file.

        Args:
            file_path: Relative path from inputs_dir

        Returns:
            All lines joined with commas
        """
        lines = self.read_file_lines(file_path)
        return ", ".join(lines)

    def get_random_lines(self, file_path: str, count: int) -> str:
        """
        Get multiple random lines from a text file (without replacement).

        Args:
            file_path: Relative path from inputs_dir
            count: Number of random lines to select

        Returns:
            Random lines joined with commas
        """
        lines = self.read_file_lines(file_path)
        if not lines:
            return ""

        # Don't try to select more lines than available
        count = min(count, len(lines))
        selected = random.sample(lines, count)
        return ", ".join(selected)

    def build_prompt(self, segments: List[Tuple[str, str]]) -> str:
        """
        Build a prompt from a list of segments.

        Args:
            segments: List of (segment_type, content) tuples
                     segment_type can be: 'text', 'file_random', 'file_specific',
                                        'file_range', 'file_all', 'file_random_multi'

        Returns:
            Combined prompt string
        """
        parts = []

        for segment_type, content in segments:
            if not content or content.strip() == "":
                continue

            if segment_type == "text":
                # User text input
                parts.append(content.strip())

            elif segment_type == "file_random":
                # Random line from file
                result = self.get_random_line(content)
                if result:
                    parts.append(result)

            elif segment_type == "file_specific":
                # Specific line from file (format: "filepath|line_number")
                try:
                    filepath, line_num = content.split("|")
                    result = self.get_specific_line(filepath, int(line_num))
                    if result:
                        parts.append(result)
                except ValueError:
                    logger.error(f"Invalid file_specific format: {content}")

            elif segment_type == "file_range":
                # Range of lines (format: "filepath|start|end")
                try:
                    filepath, start, end = content.split("|")
                    result = self.get_line_range(filepath, int(start), int(end))
                    if result:
                        parts.append(result)
                except ValueError:
                    logger.error(f"Invalid file_range format: {content}")

            elif segment_type == "file_all":
                # All lines from file
                result = self.get_all_lines(content)
                if result:
                    parts.append(result)

            elif segment_type == "file_random_multi":
                # Multiple random lines (format: "filepath|count")
                try:
                    filepath, count = content.split("|")
                    result = self.get_random_lines(filepath, int(count))
                    if result:
                        parts.append(result)
                except ValueError:
                    logger.error(f"Invalid file_random_multi format: {content}")

        # Join all parts with comma-space separator
        return ", ".join(parts)

    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a text file.

        Args:
            file_path: Relative path from inputs_dir

        Returns:
            Dictionary with file metadata (line_count, preview, etc.)
        """
        lines = self.read_file_lines(file_path)

        return {
            "line_count": len(lines),
            "preview": lines[:5] if lines else [],  # First 5 lines
            "exists": len(lines) > 0,
        }

    def clear_cache(self):
        """Clear the file content cache."""
        self._file_cache.clear()

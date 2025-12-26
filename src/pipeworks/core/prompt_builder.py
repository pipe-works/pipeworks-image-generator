"""Prompt builder utility for constructing prompts from text files and user input.

This module provides the PromptBuilder class, which enables flexible prompt
construction by combining text from multiple sources:
- Direct user input (text strings)
- Random lines from text files
- Specific lines or line ranges from text files
- Multiple random lines from text files

The prompt builder is designed to work with a directory of text files
(typically the inputs/ directory) where each file contains prompt fragments,
keywords, or descriptions. This allows for:
- Reusable prompt libraries
- Randomized prompt generation for variety
- Consistent prompt styling across generations

File Organization
-----------------
The inputs directory can have a nested structure:
    inputs/
    ├── characters.txt
    ├── environments.txt
    ├── styles/
    │   ├── realistic.txt
    │   ├── anime.txt
    │   └── abstract.txt
    └── modifiers/
        ├── lighting.txt
        └── camera.txt

Each text file should contain one item per line:
    # styles/realistic.txt
    photorealistic
    hyperrealistic
    realistic photograph
    professional photograph
    high quality photo

Selection Modes
---------------
The PromptBuilder supports multiple selection modes for each file:

1. **Random Line**: Pick one random line from the file
2. **Specific Line**: Select line N (1-indexed)
3. **Line Range**: Select lines N through M (inclusive)
4. **All Lines**: Use all lines from the file
5. **N Random Lines**: Pick N random lines without replacement

Caching
-------
File contents are cached in memory after first read to improve performance
when the same file is accessed multiple times. Call clear_cache() if files
are modified during runtime.

Usage Example
-------------
Basic usage:

    >>> from pipeworks.core.prompt_builder import PromptBuilder
    >>> from pathlib import Path
    >>> builder = PromptBuilder(Path("inputs"))
    >>>
    >>> # Get a random line from a file
    >>> style = builder.get_random_line("styles/realistic.txt")
    >>> print(style)
    'photorealistic'
    >>>
    >>> # Build a complete prompt
    >>> segments = [
    ...     ("text", "a beautiful landscape"),
    ...     ("file_random", "styles/realistic.txt"),
    ...     ("file_random", "modifiers/lighting.txt"),
    ... ]
    >>> prompt = builder.build_prompt(segments)
    >>> print(prompt)
    'a beautiful landscape, photorealistic, golden hour lighting'

Integration with UI
-------------------
The prompt builder integrates with the Gradio UI to provide:
- Folder navigation dropdown
- File selection dropdown
- Mode selection (random/specific/range/all)
- Live preview of file contents
- Three prompt segments (start/middle/end) for flexible composition

See Also
--------
- ui/components.py: SegmentUI class for UI integration
- ui/handlers.py: Prompt builder event handlers
"""

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build prompts by combining text files and user input.

    The PromptBuilder provides a file-based approach to prompt construction,
    allowing users to organize reusable prompt fragments in text files and
    combine them in flexible ways.

    The builder maintains a cache of file contents for performance, and supports
    various selection modes for retrieving lines from files.

    Attributes
    ----------
    inputs_dir : Path
        Directory containing input text files
    _file_cache : dict
        Cache of file contents (path -> list of lines)

    Notes
    -----
    - File paths are relative to inputs_dir
    - Empty lines are stripped from files
    - All lines are trimmed of whitespace
    - File cache persists across multiple calls
    - Thread-safety is not guaranteed (single-threaded use expected)

    Examples
    --------
    Create a builder and scan for files:

        >>> builder = PromptBuilder(Path("inputs"))
        >>> files = builder.scan_text_files()
        >>> print(files)
        ['characters.txt', 'styles/realistic.txt', 'modifiers/lighting.txt']

    Get random lines:

        >>> line1 = builder.get_random_line("characters.txt")
        >>> line2 = builder.get_random_line("styles/realistic.txt")
        >>> prompt = f"{line1}, {line2}"

    Build a complete prompt:

        >>> segments = [
        ...     ("text", "a wizard"),
        ...     ("file_random", "modifiers/clothing.txt"),
        ...     ("file_random", "environments.txt"),
        ... ]
        >>> prompt = builder.build_prompt(segments)
    """

    def __init__(self, inputs_dir: Path):
        """
        Initialize the prompt builder.

        Args:
            inputs_dir: Directory containing input text files
        """
        self.inputs_dir = Path(inputs_dir)
        self._file_cache: dict[str, list[str]] = {}  # Cache file contents

    def scan_text_files(self) -> list[str]:
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

    def scan_folders(self) -> list[str]:
        """
        Scan inputs directory for folders containing .txt files.

        Returns:
            List of folder names (including "(Root)" for files in inputs/)
        """
        if not self.inputs_dir.exists():
            logger.warning(f"Inputs directory does not exist: {self.inputs_dir}")
            return ["(Root)"]

        folders = set()

        # Check for files directly in inputs/ directory
        has_root_files = any(self.inputs_dir.glob("*.txt"))
        if has_root_files:
            folders.add("(Root)")

        # Find all folders containing .txt files
        for txt_file in self.inputs_dir.rglob("*.txt"):
            relative_path = txt_file.relative_to(self.inputs_dir)
            if relative_path.parent != Path("."):
                # Get the top-level folder
                top_folder = str(relative_path.parts[0])
                folders.add(top_folder)

        return sorted(list(folders))

    def get_items_in_path(self, path: str = "") -> tuple[list[str], list[str]]:
        """
        Get folders and files at a specific path level (non-recursive).

        Args:
            path: Relative path from inputs_dir (empty string for root)

        Returns:
            Tuple of (folders, files) at this level only
        """
        if not self.inputs_dir.exists():
            return [], []

        current_path = self.inputs_dir / path if path else self.inputs_dir
        if not current_path.exists():
            return [], []

        folders = []
        files = []

        try:
            for item in sorted(current_path.iterdir()):
                if item.is_dir():
                    # Only add if directory contains .txt files (directly or in subdirectories)
                    if any(item.rglob("*.txt")):
                        folders.append(item.name)
                elif item.suffix == ".txt":
                    files.append(item.name)
        except PermissionError:
            logger.error(f"Permission denied accessing: {current_path}")

        return folders, files

    def get_files_in_folder(self, folder: str) -> list[str]:
        """
        Get all .txt files in a specific folder.

        Args:
            folder: Folder name, or "(Root)" for files in inputs/ root

        Returns:
            List of filenames (not full paths)
        """
        if not self.inputs_dir.exists():
            return []

        files = []

        if folder == "(Root)":
            # Get files directly in inputs/ directory
            for txt_file in self.inputs_dir.glob("*.txt"):
                files.append(txt_file.name)
        else:
            # Get files in the specified folder (and subfolders)
            folder_path = self.inputs_dir / folder
            if folder_path.exists():
                for txt_file in folder_path.rglob("*.txt"):
                    # Get relative path from the folder
                    relative_path = txt_file.relative_to(folder_path)
                    files.append(str(relative_path))

        return sorted(files)

    def get_full_path(self, folder: str, filename: str) -> str:
        """
        Get the full relative path for a file given folder and filename.

        Args:
            folder: Folder name, path, or "(Root)"
            filename: Filename

        Returns:
            Full relative path from inputs_dir
        """
        if not filename or filename == "(None)":
            return ""

        # Handle (None) or empty folder
        if not folder or folder == "(None)" or folder == "(Root)":
            return filename

        # Combine folder path and filename
        return f"{folder}/{filename}" if folder else filename

    def read_file_lines(self, file_path: str) -> list[str]:
        """Read lines from a text file with caching.

        This method implements a simple in-memory cache to avoid repeatedly
        reading the same file from disk. The cache is a dictionary mapping
        file paths to their parsed line lists.

        Args:
            file_path: Relative path from inputs_dir

        Returns:
            List of non-empty lines (stripped of whitespace)

        Notes:
            - Empty lines are automatically filtered out
            - Leading/trailing whitespace is removed from each line
            - Results are cached in memory for subsequent calls
            - Cache persists for the lifetime of the PromptBuilder instance
        """
        # Check cache first to avoid redundant file I/O
        # This significantly improves performance for repeated access
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        # Construct full absolute path
        full_path = self.inputs_dir / file_path
        if not full_path.exists():
            logger.error(f"File not found: {full_path}")
            return []

        try:
            # Read file with UTF-8 encoding (supports international characters)
            with open(full_path, encoding="utf-8") as f:
                # Strip whitespace and filter out empty lines
                # This ensures consistent behavior regardless of file formatting
                lines = [line.strip() for line in f.readlines() if line.strip()]

            # Cache the parsed result for future calls
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

    def get_sequential_line(self, file_path: str, start_line: int, run_index: int) -> str:
        """
        Get a sequential line from a text file based on run index.

        This method is designed for batch generation where each run should use
        the next sequential line from a file. The line number is calculated as:
        line_number = start_line + run_index

        Args:
            file_path: Relative path from inputs_dir
            start_line: Starting line number (1-indexed)
            run_index: Zero-indexed run number (0, 1, 2, ...)

        Returns:
            The line at (start_line + run_index), or empty string if out of range

        Examples:
            >>> # Run 0 uses line 3, Run 1 uses line 4, Run 2 uses line 5
            >>> builder.get_sequential_line("styles.txt", start_line=3, run_index=0)
            'photorealistic'
            >>> builder.get_sequential_line("styles.txt", start_line=3, run_index=1)
            'hyperrealistic'
        """
        line_number = start_line + run_index
        return self.get_specific_line(file_path, line_number)

    def get_line_range(self, file_path: str, start: int, end: int, delimiter: str = ", ") -> str:
        """
        Get a range of lines from a text file (1-indexed, inclusive).

        Args:
            file_path: Relative path from inputs_dir
            start: Start line number (1-indexed)
            end: End line number (1-indexed, inclusive)
            delimiter: Delimiter for joining lines (default: ", ")

        Returns:
            Lines joined with delimiter, or empty string if range is invalid
        """
        lines = self.read_file_lines(file_path)
        if not lines:
            return ""

        # Clamp to valid range
        start = max(1, min(start, len(lines)))
        end = max(start, min(end, len(lines)))

        selected_lines = lines[start - 1 : end]
        return delimiter.join(selected_lines)

    def get_all_lines(self, file_path: str, delimiter: str = ", ") -> str:
        """
        Get all lines from a text file.

        Args:
            file_path: Relative path from inputs_dir
            delimiter: Delimiter for joining lines (default: ", ")

        Returns:
            All lines joined with delimiter
        """
        lines = self.read_file_lines(file_path)
        return delimiter.join(lines)

    def get_random_lines(self, file_path: str, count: int, delimiter: str = ", ") -> str:
        """
        Get multiple random lines from a text file (without replacement).

        Args:
            file_path: Relative path from inputs_dir
            count: Number of random lines to select
            delimiter: Delimiter for joining lines (default: ", ")

        Returns:
            Random lines joined with delimiter
        """
        lines = self.read_file_lines(file_path)
        if not lines:
            return ""

        # Don't try to select more lines than available
        count = min(count, len(lines))
        selected = random.sample(lines, count)
        return delimiter.join(selected)

    def _strip_trailing_delimiter(self, text: str, delimiter: str) -> str:
        """
        Strip trailing delimiter from text to avoid double punctuation.

        This method removes trailing occurrences of the delimiter (with or without
        trailing whitespace) to prevent double punctuation when joining segments.

        Args:
            text: Text to process
            delimiter: Delimiter to strip from the end

        Returns:
            Text with trailing delimiter removed

        Examples:
            >>> pb._strip_trailing_delimiter("hello,", ", ")
            'hello'
            >>> pb._strip_trailing_delimiter("hello, ", ", ")
            'hello'
            >>> pb._strip_trailing_delimiter("hello|", " | ")
            'hello'
        """
        if not text or not delimiter:
            return text

        # First, strip any trailing whitespace
        text = text.rstrip()

        # Strip the full delimiter if present (e.g., ", ")
        if text.endswith(delimiter):
            text = text[: -len(delimiter)].rstrip()
            return text

        # Strip just the delimiter without whitespace (e.g., "," from ", ")
        delimiter_no_space = delimiter.strip()
        if delimiter_no_space and text.endswith(delimiter_no_space):
            text = text[: -len(delimiter_no_space)].rstrip()

        return text

    def build_prompt(self, segments: list[tuple[str, str]], delimiter: str = ", ") -> str:
        """Build a prompt from a list of segments.

        This is the main orchestration method that combines multiple prompt
        segments into a single prompt string. Each segment can be either
        direct text or a reference to file content with a specific selection mode.

        The method processes segments in order, extracts content based on the
        segment type, and combines non-empty results with the specified delimiter.
        Trailing delimiters are automatically stripped from each segment to avoid
        double punctuation.

        Args:
            segments: List of (segment_type, content) tuples
                     segment_type can be:
                     - 'text': Direct user input text
                     - 'file_random': Random line from file
                     - 'file_specific': Specific line (format: "filepath|line_number")
                     - 'file_range': Line range (format: "filepath|start|end")
                     - 'file_all': All lines from file
                     - 'file_random_multi': N random lines (format: "filepath|count")
            delimiter: Delimiter for joining segments (default: ", ")

        Returns:
            Combined prompt string with segments joined by the delimiter

        Notes:
            - Empty segments are automatically skipped
            - Trailing delimiters are stripped from each segment to prevent double punctuation
            - Invalid segment formats are logged as errors
            - File operations use the cached read_file_lines method
            - Results are deterministic except for random selections

        Examples:
            >>> segments = [
            ...     ("text", "a wizard"),
            ...     ("file_random", "clothing.txt"),
            ...     ("file_specific", "colors.txt|3"),
            ...     ("file_random_multi", "modifiers.txt|2"),
            ... ]
            >>> prompt = builder.build_prompt(segments)
            >>> print(prompt)
            'a wizard, wearing robes, blue, highly detailed, 4k'
        """
        parts = []

        # Process each segment in order
        for segment_type, content in segments:
            # Skip empty segments
            if not content or content.strip() == "":
                continue

            if segment_type == "text":
                # Direct user text input - strip trailing delimiter to avoid double punctuation
                part = self._strip_trailing_delimiter(content.strip(), delimiter)
                if part:
                    parts.append(part)

            elif segment_type == "file_random":
                # Random line from file - provides variation
                result = self.get_random_line(content)
                if result:
                    part = self._strip_trailing_delimiter(result, delimiter)
                    if part:
                        parts.append(part)

            elif segment_type == "file_specific":
                # Specific line from file (format: "filepath|line_number")
                # Used when user wants precise control
                try:
                    filepath, line_num = content.split("|")
                    result = self.get_specific_line(filepath, int(line_num))
                    if result:
                        part = self._strip_trailing_delimiter(result, delimiter)
                        if part:
                            parts.append(part)
                except ValueError:
                    logger.error(f"Invalid file_specific format: {content}")

            elif segment_type == "file_range":
                # Range of lines (format: "filepath|start|end")
                # Useful for combining related modifiers
                try:
                    filepath, start, end = content.split("|")
                    result = self.get_line_range(filepath, int(start), int(end), delimiter)
                    if result:
                        part = self._strip_trailing_delimiter(result, delimiter)
                        if part:
                            parts.append(part)
                except ValueError:
                    logger.error(f"Invalid file_range format: {content}")

            elif segment_type == "file_all":
                # All lines from file - kitchen sink approach
                result = self.get_all_lines(content, delimiter)
                if result:
                    part = self._strip_trailing_delimiter(result, delimiter)
                    if part:
                        parts.append(part)

            elif segment_type == "file_random_multi":
                # Multiple random lines (format: "filepath|count")
                # Balance between variation and control
                try:
                    filepath, count = content.split("|")
                    result = self.get_random_lines(filepath, int(count), delimiter)
                    if result:
                        part = self._strip_trailing_delimiter(result, delimiter)
                        if part:
                            parts.append(part)
                except ValueError:
                    logger.error(f"Invalid file_random_multi format: {content}")

            elif segment_type == "file_sequential":
                # Sequential line (format: "filepath|start_line|run_index")
                # Used for batch processing where each run uses next sequential line
                try:
                    filepath, start_line, run_index = content.split("|")
                    result = self.get_sequential_line(filepath, int(start_line), int(run_index))
                    if result:
                        part = self._strip_trailing_delimiter(result, delimiter)
                        if part:
                            parts.append(part)
                except ValueError:
                    logger.error(f"Invalid file_sequential format: {content}")

        # Join all parts with the specified delimiter
        # This creates a natural prompt structure that works well with diffusion models
        return delimiter.join(parts)

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

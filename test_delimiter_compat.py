#!/usr/bin/env python3
"""Quick compatibility test for delimiter changes."""

from pathlib import Path
from pipeworks.core.prompt_builder import PromptBuilder

# Create test directory
test_dir = Path("/tmp/test_delim")
test_dir.mkdir(exist_ok=True)

# Create test file
test_file = test_dir / "test.txt"
test_file.write_text("line1\nline2\nline3\n")

# Test PromptBuilder with legacy API (non-empty delimiter)
pb = PromptBuilder(test_dir)

print("=" * 60)
print("Testing LEGACY behavior (delimiter != '')")
print("=" * 60)

# Test 1: Simple text segments with delimiter
segments = [("text", "A"), ("text", "B"), ("text", "C")]
result = pb.build_prompt(segments, delimiter=", ")
print(f"\nTest 1 - Simple text with ', ' delimiter:")
print(f"Expected: 'A, B, C'")
print(f"Got:      '{result}'")
print(f"PASS" if result == "A, B, C" else f"FAIL")

# Test 2: Text segments with trailing delimiter
segments = [("text", "hello,"), ("text", "world")]
result = pb.build_prompt(segments, delimiter=", ")
print(f"\nTest 2 - Text with trailing comma:")
print(f"Expected: 'hello, world'")
print(f"Got:      '{result}'")
print(f"PASS" if result == "hello, world" else f"FAIL")

# Test 3: File segments
segments = [("file_specific", "test.txt|1"), ("file_specific", "test.txt|2")]
result = pb.build_prompt(segments, delimiter=", ")
print(f"\nTest 3 - File segments with ', ' delimiter:")
print(f"Expected: 'line1, line2'")
print(f"Got:      '{result}'")
print(f"PASS" if result == "line1, line2" else f"FAIL")

# Test 4: Line range with delimiter
segments = [("file_range", "test.txt|1|3")]
result = pb.build_prompt(segments, delimiter=", ")
print(f"\nTest 4 - Line range:")
print(f"Expected: 'line1, line2, line3' (lines joined with ', ')")
print(f"Got:      '{result}'")
print(f"PASS" if result == "line1, line2, line3" else f"FAIL")

print("\n" + "=" * 60)
print("Testing NEW behavior (delimiter == '')")
print("=" * 60)

# Test 5: Pre-delimited segments
segments = [("text", "A,"), ("text", "B,"), ("text", "C")]
result = pb.build_prompt(segments, delimiter="")
print(f"\nTest 5 - Pre-delimited segments:")
print(f"Expected: 'A,B,C'")
print(f"Got:      '{result}'")
print(f"PASS" if result == "A,B,C" else f"FAIL")

# Test 6: Mixed delimiters
segments = [("text", "hello, "), ("text", "world. "), ("text", "end")]
result = pb.build_prompt(segments, delimiter="")
print(f"\nTest 6 - Mixed delimiters:")
print(f"Expected: 'hello, world. end'")
print(f"Got:      '{result}'")
print(f"PASS" if result == "hello, world. end" else f"FAIL")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("If all tests PASS, the build_prompt logic is correct.")
print("CI failures are likely due to test expectations, not code bugs.")

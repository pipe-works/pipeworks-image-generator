# Delimiter Changes - CI Failure Investigation

## Problem Summary

Local pre-commit hooks were passing, but CI tests are failing. The issue is that pre-commit only ran a subset of tests, missing `test_prompt_builder.py` which likely has failures.

## What Changed

### 1. Pre-commit Config (FIXED)
**Before**: Only ran 3 specific test files
**After**: Now runs all unit tests (`tests/unit/`)

### 2. Code Changes

#### `src/pipeworks/ui/handlers/prompt.py`
**Changed behavior for text+file segments:**
```python
# OLD: Used segment.delimiter between text and file
combined = f"{segment.text.strip()}{segment.delimiter}{file_content}"

# NEW: Hardcoded space between text and file, delimiter only at END
combined = f"{segment.text.strip()} {file_content}"
combined_with_end_delimiter = combined + delimiter_value
```

**Impact**: If you have a segment with both text="photo of" and file content="wizard", and delimiter=", ":
- OLD: `"photo of, wizard, "` (delimiter between AND at end)
- NEW: `"photo of wizard, "` (space between, delimiter only at end)

#### `src/pipeworks/core/prompt_builder.py`
**Added special case for empty delimiter:**
```python
if delimiter == "":
    # Don't strip! Handler already appended delimiters
    parts.append(content)
else:
    # Legacy behavior: strip and join with delimiter
    part = self._strip_trailing_delimiter(content.strip(), delimiter)
    parts.append(part)
```

**Impact**: Legacy API (non-empty delimiter) should still work correctly.

## How to Investigate CI Failures

### Step 1: Run the specific failing tests locally

Since you have Python 3.11 but need 3.12+, you may need to use Docker or a virtual environment:

```bash
# If you have Python 3.12+ available
pytest tests/unit/test_prompt_builder.py -v

# Or run all unit tests to see full picture
pytest tests/unit/ -v
```

### Step 2: Look for these specific test patterns

Tests that might be failing:

1. **Tests checking combined text+file segments**
   - Search for: `text=.*file=.*delimiter`
   - These expect the OLD behavior where delimiter was between text and file

2. **Tests in `test_prompt_builder.py` using `build_prompt` directly**
   - Lines 710-939 have many delimiter-related tests
   - These should PASS (legacy behavior preserved)
   - But check for edge cases

3. **Integration between handler and prompt_builder**
   - Tests in `test_handler_delimiters.py` expect NEW behavior
   - Tests in `test_prompt_builder.py` expect OLD behavior
   - Potential mismatch if tests share fixtures

### Step 3: Check for these specific failures

Run this command to see if specific tests are failing:

```bash
# Test the specific problem areas
pytest tests/unit/test_prompt_builder.py::TestPromptBuilderBuildPrompt::test_build_prompt_custom_delimiter -v
pytest tests/unit/test_prompt_builder.py::TestPromptBuilderBuildPrompt::test_build_prompt_double_punctuation_text_segment_fixed -v
pytest tests/unit/test_prompt_builder.py::TestPromptBuilderBuildPrompt::test_build_prompt_delimiter_parameter_used -v

# Test the new delimiter tests
pytest tests/unit/test_handler_delimiters.py -v

# Check for any import errors
pytest tests/unit/ --collect-only
```

## Expected vs Actual Behavior

### Legacy API (should still work)

```python
from pipeworks.core.prompt_builder import PromptBuilder

pb = PromptBuilder(inputs_dir)

# Test 1: Simple text joining
segments = [("text", "A"), ("text", "B")]
result = pb.build_prompt(segments, delimiter=", ")
# Expected: "A, B"

# Test 2: Text with trailing delimiter
segments = [("text", "hello,"), ("text", "world")]
result = pb.build_prompt(segments, delimiter=", ")
# Expected: "hello, world"  (trailing comma stripped)

# Test 3: File segments
segments = [("file_specific", "file.txt|1"), ("file_specific", "file.txt|2")]
result = pb.build_prompt(segments, delimiter=", ")
# Expected: "line1, line2"
```

### New Handler API

```python
from pipeworks.ui.handlers.prompt import build_combined_prompt
from pipeworks.ui.models import SegmentConfig

# Each segment appends its own delimiter
seg1 = SegmentConfig(text="hello", delimiter="Comma-Space (, )")
seg2 = SegmentConfig(text="world", delimiter="Period (.)")

# Handler resolves to:
# segments = [("text", "hello, "), ("text", "world.")]
# Then calls: pb.build_prompt(segments, delimiter="")
# Result: "hello, world."
```

## Potential Issues to Check

### Issue 1: Space between text and file within segments

If you have tests that check for specific formatting of text+file segments:

```python
# OLD behavior
seg = SegmentConfig(
    text="photo of",
    file="wizard.txt",  # contains "wizard"
    delimiter="Comma-Space (, )"
)
# OLD result: "photo of, wizard, "
# NEW result: "photo of wizard, "  ← Space, not delimiter!
```

**Fix**: Update test expectations or restore delimiter between text and file.

### Issue 2: Empty delimiter handling

If tests pass `delimiter=""` and expect stripping:

```python
# This now has special behavior
segments = [("text", "hello,"), ("text", "world.")]
result = pb.build_prompt(segments, delimiter="")
# Result: "hello,world." (NO stripping, NO joining)
```

**Fix**: These tests need to use `delimiter=" "` for legacy behavior.

### Issue 3: Delimiter mapping

Tests might use old delimiter values directly instead of labels:

```python
# OLD (might fail if test uses raw value)
segment = SegmentConfig(delimiter=", ")

# NEW (should use label)
segment = SegmentConfig(delimiter="Comma-Space (, )")
```

## Recommended Actions

1. **First**, see actual CI logs to identify specific failing tests
2. **Run tests locally** with the updated pre-commit config
3. **Check if failures are in**:
   - `test_prompt_builder.py` (legacy API compatibility)
   - `test_handler_delimiters.py` (new handler behavior)
   - Integration between the two

4. **Most likely fix**: Tests in `test_prompt_builder.py` that combine text+file might expect the old delimiter placement between text and file

5. **Quick validation**: Run the `test_delimiter_compat.py` script I created (requires Python 3.12+)

## Next Steps

1. Get the actual CI test output
2. Identify which specific tests are failing
3. Determine if it's:
   - Expected behavior change (update tests)
   - Bug in code (fix code)
   - Test fixture issue (fix test setup)

Let me know the specific test failures and I can provide targeted fixes!

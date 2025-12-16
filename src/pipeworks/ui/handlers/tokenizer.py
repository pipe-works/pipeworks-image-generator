"""Tokenizer analysis handlers."""

import logging

from ..models import UIState
from ..state import initialize_ui_state

logger = logging.getLogger(__name__)


def analyze_prompt(prompt: str, state: UIState) -> tuple[str, UIState]:
    """Analyze prompt tokenization and return formatted results.

    Args:
        prompt: Text prompt to analyze
        state: UI state (contains tokenizer)

    Returns:
        Tuple of (formatted_markdown, updated_state)
    """
    if not prompt or prompt.strip() == "":
        return "*Enter a prompt to see tokenization analysis*", state

    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Analyze the prompt
        analysis = state.tokenizer_analyzer.analyze(prompt)

        # Format results
        token_count = analysis["token_count"]
        tokens = analysis["tokens"]
        formatted_tokens = state.tokenizer_analyzer.format_tokens(tokens)

        # Build markdown output
        result = f"""
**Token Count:** {token_count}

**Tokenized Output:**
```
{formatted_tokens}
```
"""

        if analysis["special_tokens"]:
            special = ", ".join(analysis["special_tokens"])
            result += f"\n**Special Tokens Found:** {special}\n"

        return result.strip(), state

    except Exception as e:
        logger.error(f"Error analyzing prompt: {e}", exc_info=True)
        return f"*Error analyzing prompt: {str(e)}*", state

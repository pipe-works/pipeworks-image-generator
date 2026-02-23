"""Three-part prompt template compilation for the Image Generator.

The prompt system composes a final prompt from three user-selectable parts
(prepend, main scene, append) interleaved with fixed boilerplate sections
that establish a consistent aesthetic.

Template Structure
------------------
The compiled prompt follows this exact layout::

    [Prepend Style Value]

    [Fixed: Ledgerfall pamphleteer aesthetic boilerplate]

    Main Scene:

    [Manual Prompt or Automated Preset Value]

    [Fixed: Mood/atmosphere boilerplate]

    [Append Modifier Value]

    [Fixed: Colour palette directive]

Each section is separated by double newlines.  Empty prepend or append
values are silently omitted (no blank sections in the output).

Usage
-----
::

    compiled = build_prompt(
        prepend_value="In a classical oil painting style.",
        main_scene="A goblin repairing a clockwork automaton.",
        append_value="Award-winning quality, 8K resolution.",
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fixed boilerplate sections.
# These are constants rather than configuration because they define the core
# aesthetic identity of the Pipe-Works image generation system.  Users
# control variation through the three variable parts (prepend, scene, append).
# ---------------------------------------------------------------------------

_STYLE_BOILERPLATE = (
    "A horizontal, aged parchment poster in a whimsical Ledgerfall pamphleteer style. "
    "Hand-inked illustration with slightly crooked linework, warm sepia tones, soft ink bleed, "
    "delicate marginalia and decorative corner flourishes. The aesthetic feels like an old guild "
    "notice pinned in a goblin workshop — part lab diagram, part folklore warning. Detailed "
    "cross-hatching, old woodcut engraving texture, medieval fantasy sketchbook energy. Uneven "
    "bold lettering, hand-drawn banners, playful annotations and scattered scraps of paper."
)

_MOOD_BOILERPLATE = (
    "The air hangs heavy with the weight of unspoken tension. A low hum vibrates through the "
    "room, a constant reminder of unseen forces at play. The individual's posture is rigid, "
    "their gaze fixed on some distant point beyond the immediate surroundings. A subtle shift "
    "in their demeanor suggests a simmering frustration, masked by an attempt to maintain "
    "composure."
)

_COLOUR_BOILERPLATE = (
    "Color: Muted parchment background with soft brown and ink-black linework. "
    "Very limited colour palette, mostly sepia and faded parchment tones."
)


def build_prompt(
    prepend_value: str,
    main_scene: str,
    append_value: str,
) -> str:
    """Compile the full prompt from its three variable parts.

    The three caller-supplied values are interleaved with the fixed
    boilerplate sections to produce a single prompt string.  Empty or
    whitespace-only prepend/append values are silently omitted so that the
    output never contains empty sections or stray delimiters.

    Args:
        prepend_value: Style prefix text (e.g. "In a watercolour style.").
            Pass an empty string to omit.
        main_scene: The primary scene description.  This is always included
            and forms the creative core of the prompt.
        append_value: Post-processing modifier text (e.g. "cinematic
            colour grade").  Pass an empty string to omit.

    Returns:
        The fully compiled prompt string with sections separated by double
        newlines (``\\n\\n``).

    Examples:
        >>> build_prompt("Oil painting.", "A goblin workshop.", "8K detail.")
        'Oil painting.\\n\\n[boilerplate]...\\n\\nMain Scene:\\n\\nA goblin workshop.\\n\\n...'

        >>> build_prompt("", "A sunset.", "")  # No prepend or append
        '[boilerplate]...\\n\\nMain Scene:\\n\\nA sunset.\\n\\n...'
    """
    parts: list[str] = []

    # --- Prepend (optional) ------------------------------------------------
    # The prepend value sets the overall artistic style.  When the user
    # selects "None" in the UI the value arrives as an empty string.
    stripped_prepend = prepend_value.strip()
    if stripped_prepend:
        parts.append(stripped_prepend)

    # --- Fixed style boilerplate -------------------------------------------
    # Always included to establish the Ledgerfall pamphleteer aesthetic.
    parts.append(_STYLE_BOILERPLATE)

    # --- Main scene header + content ---------------------------------------
    parts.append("Main Scene:")
    parts.append(main_scene.strip())

    # --- Fixed mood boilerplate --------------------------------------------
    # Always included to set atmosphere and tension.
    parts.append(_MOOD_BOILERPLATE)

    # --- Append (optional) -------------------------------------------------
    # Post-processing modifiers like "cinematic colour grade" or
    # "award-winning quality".
    stripped_append = append_value.strip()
    if stripped_append:
        parts.append(stripped_append)

    # --- Fixed colour palette directive ------------------------------------
    # Always included to constrain the colour palette to parchment tones.
    parts.append(_COLOUR_BOILERPLATE)

    # Join all sections with double newlines for clear visual separation.
    return "\n\n".join(parts)

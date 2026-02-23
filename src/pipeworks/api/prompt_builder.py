"""Three-part prompt template compilation for the Image Generator.

The prompt system composes a final prompt from three user-selectable parts
(prepend, main scene, append) optionally interleaved with fixed boilerplate
sections that establish a consistent aesthetic.

Boilerplate Behaviour
---------------------
When a section uses **template** mode, the fixed boilerplate for that section
is included automatically — this is the stable, tested aesthetic.  When a
section uses **manual** mode, the boilerplate is omitted so the user has
full control over the prompt text.

Template Structure (all template mode)::

    [Prepend Style Value]

    [Fixed: Ledgerfall pamphleteer aesthetic boilerplate]

    Main Scene:

    [Manual Prompt or Automated Preset Value]

    [Fixed: Mood/atmosphere boilerplate]

    [Append Modifier Value]

    [Fixed: Colour palette directive]

Manual Structure (all manual mode)::

    [Manual Prepend Text]

    Main Scene:

    [Manual Prompt]

    [Manual Append Text]

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
    *,
    prepend_mode: str = "template",
    append_mode: str = "template",
) -> str:
    """Compile the full prompt from its three variable parts.

    The three caller-supplied values are combined into a single prompt
    string.  When a section is in **template** mode, the corresponding
    fixed boilerplate is included to establish the Pipe-Works aesthetic.
    When a section is in **manual** mode, the boilerplate is omitted so
    the user has full control.

    Args:
        prepend_value: Style prefix text (e.g. "In a watercolour style.").
            Pass an empty string to omit.
        main_scene: The primary scene description.  This is always included
            and forms the creative core of the prompt.
        append_value: Post-processing modifier text (e.g. "cinematic
            colour grade").  Pass an empty string to omit.
        prepend_mode: ``"template"`` to include the style boilerplate,
            ``"manual"`` to omit it.
        append_mode: ``"template"`` to include the mood and colour
            boilerplate, ``"manual"`` to omit it.

    Returns:
        The fully compiled prompt string with sections separated by double
        newlines (``\\n\\n``).
    """
    parts: list[str] = []

    # --- Prepend (optional) ------------------------------------------------
    # The prepend value sets the overall artistic style.  When the user
    # selects "None" in the UI the value arrives as an empty string.
    stripped_prepend = prepend_value.strip()
    if stripped_prepend:
        parts.append(stripped_prepend)

    # --- Fixed style boilerplate (template mode only) ----------------------
    # Included to establish the Ledgerfall pamphleteer aesthetic.  Omitted
    # in manual mode so the user's prepend text stands alone.
    if prepend_mode == "template":
        parts.append(_STYLE_BOILERPLATE)

    # --- Main scene header + content ---------------------------------------
    # The "Main Scene:" header is included in template mode to separate the
    # boilerplate from the scene.  In full-manual mode it serves no purpose.
    if prepend_mode == "template" or append_mode == "template":
        parts.append("Main Scene:")
    parts.append(main_scene.strip())

    # --- Fixed mood boilerplate (template mode only) -----------------------
    # Sets atmosphere and tension.  Omitted in manual append mode.
    if append_mode == "template":
        parts.append(_MOOD_BOILERPLATE)

    # --- Append (optional) -------------------------------------------------
    # Post-processing modifiers like "cinematic colour grade" or
    # "award-winning quality".
    stripped_append = append_value.strip()
    if stripped_append:
        parts.append(stripped_append)

    # --- Fixed colour palette directive (template mode only) ---------------
    # Constrains the colour palette to parchment tones.  Omitted in manual
    # append mode.
    if append_mode == "template":
        parts.append(_COLOUR_BOILERPLATE)

    # Join all sections with double newlines for clear visual separation.
    return "\n\n".join(parts)

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

Each section is separated by double newlines.  Empty prepend, scene, or
append values are silently omitted (no blank sections in the output).

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

import random
import re

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

_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")
_PLACEHOLDER_RANDOM = random.SystemRandom()


def expand_prompt_placeholders(
    text: str,
    *,
    rng: random.Random | random.SystemRandom | None = None,
) -> str:
    """Expand ``{a|b|c}`` placeholders using an isolated RNG.

    Each placeholder is replaced independently. Empty options are ignored and
    malformed placeholders are left unchanged. Placeholder expansion is
    intentionally isolated from the module-level ``random`` state so prompt
    variation does not perturb unrelated randomness elsewhere in the app.
    """
    if not text:
        return text

    resolved_rng = rng or _PLACEHOLDER_RANDOM

    def _replace(match: re.Match[str]) -> str:
        options = [option.strip() for option in match.group(1).split("|")]
        valid_options = [option for option in options if option]
        if not valid_options:
            return match.group(0)
        return resolved_rng.choice(valid_options)

    expanded = text
    for _ in range(10):
        updated = _PLACEHOLDER_PATTERN.sub(_replace, expanded)
        if updated == expanded:
            break
        expanded = updated
    return expanded


def resolve_prompt_variants(
    prepend_value: str,
    main_scene: str,
    append_value: str,
    *,
    rng: random.Random | random.SystemRandom | None = None,
) -> tuple[str, str, str]:
    """Expand placeholders for the three variable prompt sections once."""
    return (
        expand_prompt_placeholders(prepend_value, rng=rng),
        expand_prompt_placeholders(main_scene, rng=rng),
        expand_prompt_placeholders(append_value, rng=rng),
    )


def build_prompt(
    prepend_value: str,
    main_scene: str,
    append_value: str,
    *,
    prepend_mode: str = "template",
    append_mode: str = "template",
    expand_placeholders: bool = True,
    rng: random.Random | random.SystemRandom | None = None,
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
        main_scene: The primary scene description.  Pass an empty string to
            omit it.
        append_value: Post-processing modifier text (e.g. "cinematic
            colour grade").  Pass an empty string to omit.
        prepend_mode: ``"template"`` to include the style boilerplate,
            ``"manual"`` to omit it.
        append_mode: ``"template"`` to include the mood and colour
            boilerplate, ``"manual"`` to omit it.
        expand_placeholders: When ``True``, expand ``{a|b|c}`` placeholder
            groups before assembling the prompt.
        rng: Optional isolated RNG to use for placeholder selection.

    Returns:
        The fully compiled prompt string with sections separated by double
        newlines (``\\n\\n``).
    """
    if expand_placeholders:
        prepend_value, main_scene, append_value = resolve_prompt_variants(
            prepend_value,
            main_scene,
            append_value,
            rng=rng,
        )

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

    # --- Main scene header + content (optional) ----------------------------
    # The "Main Scene:" header is included only when scene text is present
    # and either side uses template mode to separate the boilerplate from
    # the scene. In full-manual mode it serves no purpose.
    stripped_scene = main_scene.strip()
    if stripped_scene:
        if prepend_mode == "template" or append_mode == "template":
            parts.append("Main Scene:")
        parts.append(stripped_scene)

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

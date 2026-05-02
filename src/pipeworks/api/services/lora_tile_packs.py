"""Bundled tile-pack loader for the LoRA Dataset tab.

Tile-packs are static JSON fixtures shipped with the image-generator
package. They feed the non-location categories of the LoRA Dataset
tab — character sheet, facial expressions, body actions — without
promoting render directives to canonical mud-server policies.

The packs are deliberately one-consumer (the LoRA Dataset tab) and
training-only. Locations remain canonical because they describe world
content; tile-packs describe how to render the character for ML
training and have no place in the world model.

Loading is per-request: packs are small JSON files (kilobytes) and
re-reading them on each call avoids the need for a cache-invalidation
strategy if an operator edits a pack on disk during development. If
the packs grow large enough for this to matter the loader can adopt
the same lru_cache pattern as the prompt catalog.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeworks.api.models_lora import LoraTileSpec

logger = logging.getLogger(__name__)

_PACK_FILES: dict[str, str] = {
    "character_sheet": "lora_character_sheet.json",
    "facial_expression": "lora_facial_expressions.json",
    "body_action": "lora_body_actions.json",
}


def load_tile_pack(*, data_dir: Path, kind: str) -> list[LoraTileSpec]:
    """Load and validate one tile-pack from disk.

    Returns an empty list if the pack file is missing, malformed, or
    contains no entries — callers should treat that as "no tiles
    available for this kind" rather than an error, since placeholder
    kinds ship with empty packs by design.
    """
    filename = _PACK_FILES.get(kind)
    if filename is None:
        return []

    path = data_dir / filename
    if not path.exists():
        return []

    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load LoRA tile-pack %s: %s", path, exc)
        return []

    raw_tiles = payload.get("tiles") if isinstance(payload, dict) else None
    if not isinstance(raw_tiles, list):
        return []

    tiles: list[LoraTileSpec] = []
    for raw in raw_tiles:
        if not isinstance(raw, dict):
            continue
        try:
            tiles.append(LoraTileSpec.model_validate(raw))
        except Exception as exc:  # noqa: BLE001  (pack parsing is best-effort)
            logger.warning("Skipping malformed tile in %s: %s", filename, exc)
            continue
    return tiles


def load_all_tile_packs(*, data_dir: Path) -> dict[str, list[LoraTileSpec]]:
    """Return every supported pack keyed by kind."""
    return {kind: load_tile_pack(data_dir=data_dir, kind=kind) for kind in _PACK_FILES}


def find_tile_spec(*, data_dir: Path, kind: str, key: str) -> LoraTileSpec | None:
    """Look up one tile-spec by ``(kind, key)``.

    Used by the run-creation flow to resolve a selected tile key into
    its full record (including section_label and snapshotted text).
    """
    for tile in load_tile_pack(data_dir=data_dir, kind=kind):
        if tile.key == key:
            return tile
    return None

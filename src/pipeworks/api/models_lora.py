"""Pydantic models for the LoRA Dataset tab.

A LoRA dataset run is a persistent, curated batch of generations. Each
tile in the run is a *tile-spec*: a small fragment that gets appended
to the run's pinned consistency stack to produce one image. Tile-specs
come from two sources:

- **Canonical mud-server policies** for ``location`` tiles — locations
  exist in the world model and are authored through policy-workbench.
- **Bundled JSON tile-packs** for non-world tile kinds (character
  sheet, facial expressions, body actions) — these are render
  directives for ML training, not world content, so they live as
  static fixtures in the image-generator package rather than diluting
  the canonical policy surface.

Each run carries:

- a snapshot of the prompt-v3 sections that were active in the composer
  at run-creation time (the "consistency stack"), frozen on disk so the
  run is reproducible even after the live composer is changed
- a slot per selected tile-spec, keyed by the tile's ``key``, holding
  generation params and per-tile state
- an atomic JSON manifest written to ``outputs_dir/lora_runs/<run_id>/``
  alongside per-tile PNG and caption TXT files

The shape mirrors character-forge's ``RunManifest`` / ``SlotState`` in
spirit but drops anchor-variant and scene-pack fields. Forge code is
not imported.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from pipeworks.api.models import PromptSection

SlotStatus = Literal["pending", "running", "done", "failed", "cancelled"]
TileKind = Literal["location", "character_sheet", "facial_expression", "body_action"]


class LoraRunSlot(BaseModel):
    """One tile in a LoRA dataset run.

    A slot is keyed by the tile-spec's ``key`` and carries the
    snapshotted source identity and the canonical text (from mud-server
    or bundled JSON, depending on ``tile_kind``) plus per-tile generation
    state. ``excluded=True`` removes the slot from dataset export
    without deleting the underlying files, letting the operator curate
    out drifted tiles.
    """

    tile_kind: TileKind = Field(
        default="location",
        description=(
            "Source category of the tile-spec. ``location`` is sourced "
            "from canonical mud-server policies; the others are sourced "
            "from bundled JSON tile-packs in the image-generator."
        ),
    )
    tile_source_id: str = Field(
        ...,
        description=(
            "Stable identifier within the source. For locations this is "
            "the canonical policy id (with variant suffix), e.g. "
            "'location:image.locations.environment:cozy_inn:v1'. For "
            "bundled tile-packs this is the pack key prefixed with the "
            "kind, e.g. 'character_sheet:turnaround'."
        ),
    )
    tile_key: str = Field(
        ...,
        description="Short human-readable slug, e.g. 'cozy_inn' or 'turnaround'. Used as slot id.",
    )
    tile_label: str = Field(
        ...,
        description="Display label for UI, derived from the tile source.",
    )
    tile_text: str = Field(
        ...,
        description=(
            "Snapshot of the tile-spec's canonical text at run-creation time. "
            "Frozen on disk so a run remains reproducible even if the "
            "upstream content is later edited."
        ),
    )
    section_label: str = Field(
        default="Location",
        description=(
            "Prompt section header used when the tile_text is appended to "
            "the consistency stack. Locations use 'Location'; character "
            "sheet tiles use 'Character Sheet'; expressions and actions "
            "use their respective labels. Per-tile so a single run can "
            "mix kinds with appropriate prompt structure."
        ),
    )
    compiled_prompt: str = Field(
        default="",
        description=("Full prompt-v3 compilation (consistency stack + the slot's tile section)."),
    )
    seed: int | None = Field(
        default=None,
        description="Seed used for this tile. None until the slot has been generated.",
    )
    image_filename: str | None = Field(
        default=None,
        description="Tile PNG filename relative to the run dir (e.g. '01_cozy_inn.png').",
    )
    caption_filename: str | None = Field(
        default=None,
        description="Tile caption TXT filename relative to the run dir.",
    )
    status: SlotStatus = Field(
        default="pending",
        description="Per-slot generation lifecycle.",
    )
    error: str | None = Field(
        default=None,
        description="Error detail if the slot's last generation attempt failed.",
    )
    excluded: bool = Field(
        default=False,
        description=(
            "Operator opt-out flag. Excluded slots are skipped at dataset "
            "export but their files remain on disk for re-inclusion."
        ),
    )


class LoraRunParams(BaseModel):
    """Generation parameters frozen at run-creation time.

    Mirrors the relevant subset of ``GenerateRequest`` fields. Pinning
    these on the manifest keeps the run reproducible and lets each tile
    be (re)generated with consistent settings.
    """

    model_id: str
    aspect_ratio_id: str
    width: int
    height: int
    steps: int
    guidance: float
    scheduler: str | None = None
    base_seed: int = Field(
        ...,
        description=(
            "Base seed for the run. With ``share_seed_across_tiles`` true "
            "(the default for LoRA runs) every tile uses ``base_seed`` "
            "verbatim, so all tiles initialise from identical noise — "
            "the strongest character lock available without i2i. With it "
            "false, per-tile seed is ``base_seed + slot_order_index`` for "
            "more pose/angle variance at the cost of identity drift."
        ),
    )
    share_seed_across_tiles: bool = Field(
        default=True,
        description=(
            "When true, every pending slot uses ``base_seed`` for "
            "generation; when false, each slot uses "
            "``base_seed + slot_order_index``. Toggling this on a run "
            "recomputes seeds for slots still in ``pending`` state but "
            "leaves done/failed slots' historical seeds untouched."
        ),
    )
    negative_prompt: str | None = None


class LoraRunManifest(BaseModel):
    """On-disk manifest for one LoRA dataset run.

    Persisted at ``outputs_dir/lora_runs/<run_id>/manifest.json`` via
    atomic write (tmp file plus ``os.replace``). Browser polling can
    read the manifest while a worker thread is updating it.
    """

    run_id: str = Field(..., description="UUID identifying this run.")
    schema_version: Literal[2] = Field(
        default=2,
        description="Manifest schema version. Bumped to 2 with the tile-kind generalisation.",
    )
    created_at: float = Field(..., description="UNIX timestamp at run creation.")
    updated_at: float = Field(..., description="UNIX timestamp of last manifest write.")
    status: Literal["pending", "running", "complete", "cancelled", "failed"] = Field(
        default="pending",
        description="Run-level lifecycle state.",
    )
    cancel_requested: bool = Field(
        default=False,
        description=(
            "Best-effort cancellation flag. The generation worker checks "
            "this between tiles and stops cleanly when set."
        ),
    )
    params: LoraRunParams
    pinned_sections: list[PromptSection] = Field(
        ...,
        description=(
            "Snapshot of the prompt-v3 consistency stack. Each slot's "
            "compiled prompt is built from these sections plus a tile "
            "section appended for the slot."
        ),
    )
    slots: dict[str, LoraRunSlot] = Field(
        default_factory=dict,
        description="Per-tile slot state, keyed by tile_key.",
    )
    slot_order: list[str] = Field(
        default_factory=list,
        description=(
            "Stable ordering of tile_keys. Mirrors the order tiles "
            "were selected at run-creation; per-tile seeds and tile "
            "filename prefixes are derived from this order."
        ),
    )


class LoraRunCreateRequest(BaseModel):
    """Request body for ``POST /api/lora-dataset/runs``.

    The frontend snapshots the current prompt-v3 composer state and
    pins it to the run alongside the operator's selected tile-specs
    and generation parameters. Tile-specs are split by source so the
    backend can resolve each from its appropriate registry.
    """

    model_id: str
    aspect_ratio_id: str
    width: int
    height: int
    steps: int
    guidance: float
    scheduler: str | None = None
    seed: int | None = Field(
        default=None,
        description="Optional base seed; server picks one if omitted.",
    )
    negative_prompt: str | None = None
    pinned_sections: list[PromptSection] = Field(
        default_factory=list,
        description=(
            "Prompt-v3 sections snapshotted from the composer. Empty "
            "lists are allowed but will produce thin tiles."
        ),
    )
    location_policy_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical location policy ids selected for this run, e.g. "
            "['location:image.locations.environment:cozy_inn:v1', ...]. "
            "Identifiers carry the variant suffix; the run snapshots "
            "the canonical text so subsequent edits do not affect this run."
        ),
    )
    character_sheet_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Bundled character-sheet tile keys selected for this run, "
            "e.g. ['turnaround']. Resolved against the bundled "
            "``lora_character_sheet.json`` tile-pack at run creation."
        ),
    )
    share_seed_across_tiles: bool = Field(
        default=True,
        description=(
            "Default true: every tile in the run uses ``base_seed`` "
            "verbatim, locking character identity across the dataset. "
            "Set false to revert to ``base_seed + slot_order_index`` "
            "per-tile seeds for more pose/angle variance."
        ),
    )


class LoraRunPatch(BaseModel):
    """Request body for ``PATCH /api/lora-dataset/runs/{id}``.

    Currently only flips the seed-strategy flag, but is shaped as a
    partial-update model so future run-level toggles can ride on the
    same endpoint without a route proliferation.
    """

    share_seed_across_tiles: bool | None = Field(
        default=None,
        description=(
            "If set, toggle the run's seed strategy. Only pending slots' "
            "seeds are recomputed; done/failed slots keep their historical "
            "seed value so curation history is not rewritten."
        ),
    )


class LoraRunSlotPatch(BaseModel):
    """Request body for ``PATCH /api/lora-dataset/runs/{id}/slots/{key}``."""

    excluded: bool | None = Field(
        default=None,
        description="If set, toggle the slot's excluded flag.",
    )


class LoraTileSpec(BaseModel):
    """One tile-spec from a bundled JSON tile-pack.

    Returned by ``GET /api/lora-dataset/tile-packs`` so the frontend can
    populate per-kind pickers without bundled JS data.
    """

    key: str = Field(..., description="Stable tile-pack key, e.g. 'turnaround'.")
    label: str = Field(..., description="Display label for UI.")
    text: str = Field(..., description="Tile-spec prompt fragment.")
    section_label: str = Field(
        ...,
        description=(
            "Prompt section header used when this tile-spec is appended "
            "to a run's consistency stack."
        ),
    )
    aspect_ratio_hint: str | None = Field(
        default=None,
        description=(
            "Optional hint to the operator about the aspect ratio that "
            "best suits this tile-spec, e.g. '16:9' for a wide turnaround "
            "sheet. Currently advisory only — runs use a single aspect "
            "ratio across all tiles."
        ),
    )


class LoraTilePacksResponse(BaseModel):
    """Response body for ``GET /api/lora-dataset/tile-packs``."""

    character_sheet: list[LoraTileSpec] = Field(default_factory=list)
    facial_expression: list[LoraTileSpec] = Field(default_factory=list)
    body_action: list[LoraTileSpec] = Field(default_factory=list)


class LoraDatasetExportResult(BaseModel):
    """Response body for ``POST /api/lora-dataset/runs/{id}/dataset``."""

    run_id: str
    dataset_dir: str
    pairs_copied: int
    excluded: int
    skipped: int

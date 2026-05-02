"""Pydantic models for the LoRA Dataset tab.

A LoRA dataset run is a persistent, curated batch of generations whose
diversity axis is the canonical ``location`` policy and whose consistency
axis is everything else in the prompt-v3 composition (species, clothing,
tone profile, etc.). Each run carries:

- a snapshot of the prompt-v3 sections that were active in the composer
  at run-creation time (the "consistency stack"), frozen on disk so the
  run is reproducible even after the live composer is changed
- a slot per selected location, keyed by the location's canonical
  ``policy_key``, holding generation params and per-tile state
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


class LoraRunSlot(BaseModel):
    """One tile in a LoRA dataset run.

    A slot is keyed by the canonical location ``policy_key`` and carries
    the snapshotted location identity (policy_id + variant + canonical
    text) plus per-tile generation state. ``excluded=True`` removes the
    slot from dataset export without deleting the underlying files,
    letting the operator curate out drifted tiles.
    """

    location_policy_id: str = Field(
        ...,
        description=(
            "Canonical location identifier, e.g. "
            "'location:image.locations.environment:cozy_inn'."
        ),
    )
    location_variant: str = Field(
        default="v1",
        description="Canonical location variant pinned at run-creation time.",
    )
    location_key: str = Field(
        ...,
        description="Short human-readable slug, e.g. 'cozy_inn'. Used as slot id.",
    )
    location_label: str = Field(
        ...,
        description="Display label for UI, derived from the location policy.",
    )
    location_text: str = Field(
        ...,
        description=(
            "Snapshot of the canonical location text at run-creation time. "
            "Frozen on disk so a run remains reproducible even if the "
            "canonical content is later edited upstream."
        ),
    )
    compiled_prompt: str = Field(
        default="",
        description=(
            "Full prompt-v3 compilation (consistency stack + the slot's " "location section)."
        ),
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
            "Base seed for the run. Per-tile seed = base_seed + slot_order. "
            "Determinism is intentional so regenerated tiles can be made "
            "different from their first attempt by varying the slot's seed."
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
    schema_version: Literal[1] = Field(
        default=1,
        description="Manifest schema version. Bump on incompatible changes.",
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
            "compiled prompt is built from these sections plus a "
            "location section appended for the slot's location."
        ),
    )
    location_section_label: str = Field(
        default="Location",
        description=(
            "Label used for the appended location section in each tile's "
            "compiled prompt. Defaults to 'Location' but operators can "
            "override it (rare)."
        ),
    )
    slots: dict[str, LoraRunSlot] = Field(
        default_factory=dict,
        description="Per-location slot state, keyed by location_key.",
    )
    slot_order: list[str] = Field(
        default_factory=list,
        description=(
            "Stable ordering of location_keys. Mirrors the order locations "
            "were selected at run-creation; per-tile seeds and tile "
            "filename prefixes are derived from this order."
        ),
    )


class LoraRunCreateRequest(BaseModel):
    """Request body for ``POST /api/lora-dataset/runs``.

    The frontend snapshots the current prompt-v3 composer state and
    pins it to the run alongside the operator's selected locations and
    generation parameters.
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
    location_section_label: str = Field(default="Location")
    location_policy_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Canonical location policy ids selected for this run, e.g. "
            "['location:image.locations.environment:cozy_inn:v1', ...]. "
            "Identifiers carry the variant suffix; the run snapshots "
            "the canonical text so subsequent edits do not affect this run."
        ),
    )


class LoraRunSlotPatch(BaseModel):
    """Request body for ``PATCH /api/lora-dataset/runs/{id}/slots/{key}``."""

    excluded: bool | None = Field(
        default=None,
        description="If set, toggle the slot's excluded flag.",
    )


class LoraDatasetExportResult(BaseModel):
    """Response body for ``POST /api/lora-dataset/runs/{id}/dataset``."""

    run_id: str
    dataset_dir: str
    pairs_copied: int
    excluded: int
    skipped: int

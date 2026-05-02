"""LoRA Dataset tab API routes.

Persistent, curated batches whose tiles come from two sources:

- **Locations** — canonical mud-server policies, world content. The
  diversity axis when training on character-in-environment tiles.
- **Bundled tile-packs** — static JSON fixtures shipped with the
  image-generator (character-sheet, facial expressions, body actions).
  Render directives for ML training, not world content; they live in
  the package rather than diluting the canonical policy surface.

Each run is its own directory under
``outputs_dir/lora_runs/<run_id>/`` with an atomic JSON manifest, per-tile
PNG and TXT files, and an optional ``dataset/`` export subdirectory.

The router intentionally calls the underlying generation primitive
(``ModelManager.generate``) directly rather than re-entering ``/api/generate``.
That keeps the run flow free of gallery side-effects and policy-prompt
re-resolution: tile text is already snapshotted on the manifest at
run-creation time, so no live mud-server calls are needed during the
per-tile loop.
"""

# i2i-fallback note: this router is exclusively text-to-image. If the
# seed+prompt consistency axis turns out to be inadequate for character
# LoRA quality, the natural i2i extension is FLUX.2-klein-style cascade
# from a stylized base — see character-forge for the parallel reference
# design. The seams that would have to be threaded are flagged with
# "i2i-extension-seam" comments at the run-creation flow, the per-tile
# generation call site, and the manifest write site.

from __future__ import annotations

import logging
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from pipeworks.api.models import PromptSection
from pipeworks.api.models_lora import (
    LoraDatasetExportResult,
    LoraRunCreateRequest,
    LoraRunManifest,
    LoraRunParams,
    LoraRunPatch,
    LoraRunSlot,
    LoraRunSlotPatch,
    LoraTilePacksResponse,
)
from pipeworks.api.prompt_builder import (
    build_dynamic_prompt,
    expand_prompt_placeholders,
)
from pipeworks.api.services.generation_runtime import GenerationRuntimeService
from pipeworks.api.services.lora_run_store import (
    dataset_dir_for,
    ensure_run_dir,
    iter_runs,
    new_run_id,
    read_manifest,
    run_dir_for,
    update_manifest,
    write_manifest,
)
from pipeworks.api.services.lora_tile_packs import find_tile_spec, load_all_tile_packs
from pipeworks.api.services.prompt_catalog import load_json
from pipeworks.api.services.runtime_policy import RuntimePolicyService
from pipeworks.core.model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoraDatasetRouterDependencies:
    """Dependencies required by LoRA dataset routes."""

    data_dir: Callable[[], Path]
    outputs_dir: Callable[[], Path]
    runtime_policy_service: RuntimePolicyService
    generation_runtime_service: GenerationRuntimeService
    normalize_base_url: Callable[[str | None], str]
    get_model_runtime_support: Callable[[str], tuple[bool, str | None]]


def create_lora_dataset_router(deps: LoraDatasetRouterDependencies) -> APIRouter:
    """Build APIRouter exposing the LoRA Dataset tab endpoints."""
    router = APIRouter()

    @router.get("/api/lora-dataset/tile-packs")
    async def list_tile_packs() -> dict:
        """Return bundled tile-packs by kind for the LoRA tab pickers.

        Locations are not included here — those come from the existing
        ``/api/policy-prompts`` mud-server snippet pipeline. Tile-packs
        cover the local-only kinds (character_sheet, facial_expression,
        body_action). Empty packs are returned as empty lists so the
        frontend can render placeholder UI without special-casing.
        """
        packs = load_all_tile_packs(data_dir=deps.data_dir())
        response = LoraTilePacksResponse(
            character_sheet=packs.get("character_sheet", []),
            facial_expression=packs.get("facial_expression", []),
            body_action=packs.get("body_action", []),
        )
        return response.model_dump(mode="json")

    @router.post("/api/lora-dataset/runs")
    async def create_run(req: LoraRunCreateRequest, request: Request) -> dict:
        """Create a new LoRA dataset run and snapshot its tile sources.

        Locations are resolved against the live mud-server policy snippet
        catalog so the canonical text is captured on the manifest at
        creation time. Bundled tile-pack keys (character-sheet,
        facial-expression, body-action) are resolved against the
        package-local JSON fixtures. Once snapshotted, the run is
        independent of upstream edits — reproducibility wins over
        freshness here.
        """
        if not req.location_policy_ids and not req.character_sheet_keys:
            raise HTTPException(
                status_code=400,
                detail=(
                    "At least one tile must be selected: location_policy_ids "
                    "and/or character_sheet_keys."
                ),
            )

        slots: dict[str, LoraRunSlot] = {}
        slot_order: list[str] = []

        # --- Locations (canonical mud-server policies) ----------------
        if req.location_policy_ids:
            policy_options, _, runtime_auth = (
                deps.runtime_policy_service.load_policy_prompts_for_request(
                    request=request,
                    response=None,
                    explicit_session_id=None,
                    normalize_base_url=deps.normalize_base_url,
                )
            )
            if not runtime_auth.access_granted:
                raise HTTPException(
                    status_code=401,
                    detail="LoRA dataset runs require an authenticated mud-server session.",
                )

            options_by_id = {option["id"]: option for option in policy_options}
            for policy_id in req.location_policy_ids:
                option = options_by_id.get(policy_id)
                if option is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Unknown or inaccessible location policy id: {policy_id!r}. "
                            "Make sure the location is published and the active session "
                            "has admin/superuser role."
                        ),
                    )
                # Option id format is "<policy_type>:<namespace>:<policy_key>:<variant>".
                id_parts = policy_id.split(":")
                if len(id_parts) != 4 or id_parts[0] != "location":
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Policy {policy_id!r} is not a canonical location "
                            f"(expected id 'location:<namespace>:<policy_key>:<variant>')."
                        ),
                    )
                _, _, location_key, _ = id_parts
                if not location_key or location_key in slots:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Duplicate or empty tile_key for policy {policy_id!r}.",
                    )

                slots[location_key] = LoraRunSlot(
                    tile_kind="location",
                    tile_source_id=policy_id,
                    tile_key=location_key,
                    tile_label=str(option.get("label") or location_key),
                    tile_text=str(option.get("value") or ""),
                    section_label="Location",
                )
                slot_order.append(location_key)

        # --- Character-sheet (bundled JSON tile-pack) -----------------
        for sheet_key in req.character_sheet_keys:
            tile = find_tile_spec(data_dir=deps.data_dir(), kind="character_sheet", key=sheet_key)
            if tile is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown character-sheet tile key: {sheet_key!r}. "
                        "Check the bundled lora_character_sheet.json pack."
                    ),
                )
            if tile.key in slots:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Tile key collision: {tile.key!r} is already in this run "
                        "(likely as a location). Tile keys must be unique across kinds."
                    ),
                )

            slots[tile.key] = LoraRunSlot(
                tile_kind="character_sheet",
                tile_source_id=f"character_sheet:{tile.key}",
                tile_key=tile.key,
                tile_label=tile.label,
                tile_text=tile.text,
                section_label=tile.section_label,
            )
            slot_order.append(tile.key)

        base_seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
        # i2i-extension-seam: a future i2i variant would also stage a
        # base reference image at this point (e.g. uploaded by the
        # operator or generated as a stylized base) and persist it
        # under the run dir alongside the manifest.
        run_id = new_run_id()
        now = time.time()

        # Resolve `{a|b|c}` placeholders ONCE at run creation so every tile
        # in the dataset shares the same draw. Placeholder expansion is
        # appropriate for normal /api/generate batches (each image gets
        # an independent draw — that's the whole point of the syntax) but
        # actively harmful for LoRA training, where character identity
        # must be stable across the dataset. Freezing the draw at run
        # creation gives the operator a single deterministic stack that
        # gets reused by every tile.
        resolved_pinned: list[PromptSection] = []
        for section in req.pinned_sections:
            resolved_text = (
                expand_prompt_placeholders(section.manual_text)
                if section.manual_text
                else section.manual_text
            )
            resolved_pinned.append(
                PromptSection(
                    label=section.label,
                    mode=section.mode,
                    manual_text=resolved_text,
                    automated_prompt_id=section.automated_prompt_id,
                )
            )
        resolved_negative_prompt = (
            expand_prompt_placeholders(req.negative_prompt) if req.negative_prompt else None
        )

        manifest = LoraRunManifest(
            run_id=run_id,
            created_at=now,
            updated_at=now,
            params=LoraRunParams(
                model_id=req.model_id,
                aspect_ratio_id=req.aspect_ratio_id,
                width=req.width,
                height=req.height,
                steps=req.steps,
                guidance=req.guidance,
                scheduler=req.scheduler,
                base_seed=base_seed,
                share_seed_across_tiles=req.share_seed_across_tiles,
                negative_prompt=resolved_negative_prompt,
            ),
            pinned_sections=resolved_pinned,
            slots=slots,
            slot_order=slot_order,
        )
        # Compile each slot's prompt up-front so the manifest is
        # self-describing even before any tile has run, and assign each
        # slot its planned seed using the run's seed strategy.
        for index, key in enumerate(slot_order):
            slot = manifest.slots[key]
            slot.compiled_prompt = _compile_slot_prompt(
                pinned_sections=manifest.pinned_sections,
                slot=slot,
            )
            slot.seed = _planned_seed_for_slot(
                base_seed=base_seed,
                slot_order_index=index,
                share_seed_across_tiles=req.share_seed_across_tiles,
            )

        ensure_run_dir(deps.outputs_dir(), run_id)
        write_manifest(deps.outputs_dir(), manifest)
        return manifest.model_dump(mode="json")

    @router.get("/api/lora-dataset/runs")
    async def list_runs() -> dict:
        manifests = [m.model_dump(mode="json") for m in iter_runs(deps.outputs_dir())]
        return {"runs": manifests}

    @router.get("/api/lora-dataset/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        manifest = read_manifest(deps.outputs_dir(), run_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="run not found")
        return manifest.model_dump(mode="json")

    @router.patch("/api/lora-dataset/runs/{run_id}")
    async def patch_run(run_id: str, patch: LoraRunPatch) -> dict:
        """Toggle run-level flags (currently just the seed strategy).

        Slots already in ``done`` or ``failed`` state keep their historical
        ``seed`` value so the curation record is not rewritten — only
        ``pending`` slots are updated to the new strategy.

        The endpoint refuses to act on a ``running`` run; toggling mid-flight
        would race the per-tile loop's seed reads.
        """
        existing = read_manifest(deps.outputs_dir(), run_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="run not found")
        if existing.status == "running":
            raise HTTPException(
                status_code=409,
                detail="run is currently generating; cancel before changing seed strategy.",
            )

        if patch.share_seed_across_tiles is None:
            return existing.model_dump(mode="json")

        new_strategy = patch.share_seed_across_tiles

        def _apply(manifest: LoraRunManifest) -> None:
            manifest.params.share_seed_across_tiles = new_strategy
            for index, key in enumerate(manifest.slot_order):
                slot = manifest.slots.get(key)
                if slot is None or slot.status in {"done", "failed"}:
                    continue
                slot.seed = _planned_seed_for_slot(
                    base_seed=manifest.params.base_seed,
                    slot_order_index=index,
                    share_seed_across_tiles=new_strategy,
                )

        manifest = update_manifest(deps.outputs_dir(), run_id, _apply)
        if manifest is None:
            raise HTTPException(status_code=404, detail="run not found")
        return manifest.model_dump(mode="json")

    @router.delete("/api/lora-dataset/runs/{run_id}")
    async def delete_run(run_id: str) -> dict:
        run_dir = run_dir_for(deps.outputs_dir(), run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="run not found")
        shutil.rmtree(run_dir)
        return {"success": True, "run_id": run_id}

    @router.patch("/api/lora-dataset/runs/{run_id}/slots/{slot_key}")
    async def patch_slot(run_id: str, slot_key: str, patch: LoraRunSlotPatch) -> dict:
        def _apply(manifest: LoraRunManifest) -> None:
            slot = manifest.slots.get(slot_key)
            if slot is None:
                raise HTTPException(status_code=404, detail="slot not found")
            if patch.excluded is not None:
                slot.excluded = patch.excluded

        manifest = update_manifest(deps.outputs_dir(), run_id, _apply)
        if manifest is None:
            raise HTTPException(status_code=404, detail="run not found")
        return manifest.slots[slot_key].model_dump(mode="json")

    @router.post("/api/lora-dataset/runs/{run_id}/cancel")
    async def cancel_run(run_id: str) -> dict:
        existing = read_manifest(deps.outputs_dir(), run_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="run not found")

        cancel_event = deps.generation_runtime_service.resolve_generation_cancel_event(run_id)
        if cancel_event is not None:
            cancel_event.set()

        def _flag(manifest: LoraRunManifest) -> None:
            manifest.cancel_requested = True

        update_manifest(deps.outputs_dir(), run_id, _flag)
        return {"success": True, "run_id": run_id, "status": "cancelling"}

    @router.post("/api/lora-dataset/runs/{run_id}/generate")
    async def generate_run(run_id: str, request: Request) -> dict:
        """Run the per-tile generation loop synchronously.

        Mirrors the existing ``/api/generate`` shape: the call hangs until
        the run finishes (or is cancelled), and the browser is expected
        to poll ``GET /api/lora-dataset/runs/{run_id}`` for progress.
        Manifest writes are atomic so concurrent polling is safe.
        """
        manifest = read_manifest(deps.outputs_dir(), run_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="run not found")

        return await _execute_run(
            deps=deps,
            request=request,
            run_id=run_id,
            slot_keys=list(manifest.slot_order),
        )

    @router.post("/api/lora-dataset/runs/{run_id}/slots/{slot_key}/regenerate")
    async def regenerate_slot(run_id: str, slot_key: str, request: Request) -> dict:
        """Re-run a single tile using a fresh per-slot seed.

        Each regeneration bumps the slot's seed deterministically so
        the new attempt differs from the previous one without losing
        run-level reproducibility. The slot's ``excluded`` flag is
        cleared on regeneration on the assumption that the operator is
        actively curating the tile back into the dataset.
        """
        manifest = read_manifest(deps.outputs_dir(), run_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="run not found")
        if slot_key not in manifest.slots:
            raise HTTPException(status_code=404, detail="slot not found")

        def _bump(manifest: LoraRunManifest) -> None:
            slot = manifest.slots[slot_key]
            slot_index = (
                manifest.slot_order.index(slot_key) if slot_key in manifest.slot_order else 0
            )
            # Stride bumps by slot count plus the slot's order index so two
            # slots regenerated from the same starting seed (the shared-seed
            # default state) end up with distinct seeds rather than colliding.
            previous_seed = slot.seed if slot.seed is not None else manifest.params.base_seed
            slot.seed = previous_seed + len(manifest.slots) + slot_index
            slot.status = "pending"
            slot.error = None
            slot.excluded = False

        update_manifest(deps.outputs_dir(), run_id, _bump)

        return await _execute_run(
            deps=deps,
            request=request,
            run_id=run_id,
            slot_keys=[slot_key],
        )

    @router.get("/api/lora-dataset/runs/{run_id}/files/{filename}")
    async def get_run_file(run_id: str, filename: str) -> FileResponse:
        """Serve a per-tile file from a run directory.

        Path traversal is rejected by refusing any filename that contains a
        path separator or a parent-dir component. Files outside the run
        directory cannot be reached this way.
        """
        if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
            raise HTTPException(status_code=400, detail="invalid filename")
        run_dir = run_dir_for(deps.outputs_dir(), run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="run not found")
        target = run_dir / filename
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(target)

    @router.post("/api/lora-dataset/runs/{run_id}/dataset")
    async def export_dataset(run_id: str) -> dict:
        manifest = read_manifest(deps.outputs_dir(), run_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="run not found")

        dataset_dir = dataset_dir_for(deps.outputs_dir(), run_id)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        run_dir = run_dir_for(deps.outputs_dir(), run_id)
        pairs_copied = 0
        excluded = 0
        skipped = 0
        for index, key in enumerate(manifest.slot_order):
            slot = manifest.slots.get(key)
            if slot is None or slot.status != "done":
                skipped += 1
                continue
            if slot.excluded:
                excluded += 1
                continue
            if not slot.image_filename or not slot.caption_filename:
                skipped += 1
                continue
            png_src = run_dir / slot.image_filename
            txt_src = run_dir / slot.caption_filename
            if not png_src.exists() or not txt_src.exists():
                skipped += 1
                continue
            base = f"{index:02d}_{key}"
            shutil.copyfile(png_src, dataset_dir / f"{base}.png")
            shutil.copyfile(txt_src, dataset_dir / f"{base}.txt")
            pairs_copied += 1

        result = LoraDatasetExportResult(
            run_id=run_id,
            dataset_dir=str(dataset_dir),
            pairs_copied=pairs_copied,
            excluded=excluded,
            skipped=skipped,
        )
        return result.model_dump(mode="json")

    return router


def _planned_seed_for_slot(
    *, base_seed: int, slot_order_index: int, share_seed_across_tiles: bool
) -> int:
    """Compute a slot's planned seed from the run's seed strategy.

    Shared seed is the default for LoRA runs because identical noise
    plus a frozen consistency stack gives the strongest character lock
    available without i2i. The legacy per-tile-offset strategy is kept
    behind a toggle for operators who want pose/angle variance even at
    the cost of identity drift across the dataset.
    """
    if share_seed_across_tiles:
        return base_seed
    return base_seed + slot_order_index


def _compile_slot_prompt(
    *,
    pinned_sections: list[PromptSection],
    slot: LoraRunSlot,
) -> str:
    """Build the full prompt-v3 compilation for one tile.

    The slot's tile text is appended as the final section using the
    slot's ``section_label`` as the header — this is per-tile so a
    single run can mix tile kinds (locations, character sheet,
    expressions, actions) with appropriate prompt structure.

    Placeholders are NOT re-expanded here: ``create_run`` resolves
    ``{a|b|c}`` once at run creation time and freezes the draw on
    ``manifest.pinned_sections`` so every tile in the dataset sees the
    same consistency stack.
    """
    sections: list[dict[str, str]] = []
    for pinned in pinned_sections:
        text = (pinned.manual_text or "").strip()
        if not text:
            continue
        sections.append({"label": pinned.label, "text": text})
    sections.append({"label": slot.section_label, "text": slot.tile_text})

    return build_dynamic_prompt(sections, expand_placeholders=False)


async def _execute_run(
    *,
    deps: LoraDatasetRouterDependencies,
    request: Request,
    run_id: str,
    slot_keys: list[str],
) -> dict:
    """Drive the per-tile generation loop for a run subset.

    Used for both the full-run path and per-slot regeneration. The
    function hangs until all targeted slots are ``done``, ``failed``,
    or the run is cancelled; the manifest is updated atomically after
    each slot transition so the browser sees progress in near-real-time.
    """
    outputs_dir = deps.outputs_dir()
    data_dir = deps.data_dir()
    manifest = read_manifest(outputs_dir, run_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="run not found")

    models_data = load_json(data_dir / "models.json", {"models": []})
    model_cfg = next(
        (m for m in models_data.get("models", []) if m["id"] == manifest.params.model_id), None
    )
    if not model_cfg:
        raise HTTPException(status_code=400, detail=f"Unknown model: {manifest.params.model_id}")
    hf_id = model_cfg["hf_id"]
    model_label = str(model_cfg.get("label") or manifest.params.model_id)

    is_available, unavailable_reason = deps.get_model_runtime_support(hf_id)
    if not is_available:
        raise HTTPException(
            status_code=503,
            detail=unavailable_reason or "Selected model is unavailable in this runtime.",
        )

    cache_miss = deps.generation_runtime_service.has_cached_model(hf_id) is False
    deps.generation_runtime_service.register_generation_status(
        generation_id=run_id,
        model_id=manifest.params.model_id,
        model_label=model_label,
        worker_label="local",
        batch_size=len(slot_keys),
        cache_miss=cache_miss,
    )
    cancel_event = deps.generation_runtime_service.register_generation_cancel_event(run_id)

    def _mark_running(manifest: LoraRunManifest) -> None:
        manifest.status = "running"
        manifest.cancel_requested = False
        for key in slot_keys:
            slot = manifest.slots.get(key)
            if slot is None:
                continue
            slot.status = "pending"
            slot.error = None

    update_manifest(outputs_dir, run_id, _mark_running)

    model_mgr: ModelManager = request.app.state.model_manager
    if model_mgr.current_model_id != hf_id:
        deps.generation_runtime_service.update_generation_status(
            run_id,
            phase="loading_model",
            message=f"Loading {model_label} on Luminal GPU…",
        )
        await run_in_threadpool(model_mgr.load_model, hf_id)

    completed = 0
    cancelled = False
    try:
        for tile_index, key in enumerate(slot_keys):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break

            manifest = read_manifest(outputs_dir, run_id)
            if manifest is None or manifest.cancel_requested:
                cancelled = True
                break
            slot = manifest.slots.get(key)
            if slot is None:
                continue

            slot_order_index = (
                manifest.slot_order.index(key) if key in manifest.slot_order else tile_index
            )

            def _mark_running_slot(m: LoraRunManifest, k: str = key) -> None:
                target = m.slots.get(k)
                if target is None:
                    return
                target.status = "running"
                target.error = None

            update_manifest(outputs_dir, run_id, _mark_running_slot)

            deps.generation_runtime_service.update_generation_status(
                run_id,
                phase="generating",
                message=(
                    f"Generating tile {tile_index + 1} of {len(slot_keys)} " f"({slot.tile_label})…"
                ),
                completed_count=completed,
            )

            seed_value = (
                slot.seed if slot.seed is not None else manifest.params.base_seed + tile_index
            )

            # Placeholders were resolved once at run creation
            # (``create_run`` freezes ``{a|b|c}`` draws on the manifest).
            # Each slot's compiled prompt was also built at create time;
            # rebuild here to pick up any post-creation slot edits.
            compiled_prompt = _compile_slot_prompt(
                pinned_sections=manifest.pinned_sections,
                slot=slot,
            )

            # ``params.negative_prompt`` is also resolved at run creation,
            # so it is used here as-is rather than re-expanded per tile.
            negative_prompt = manifest.params.negative_prompt

            try:
                # i2i-extension-seam: a future FLUX.2-klein i2i variant
                # would replace this txt-to-img call with a strength-
                # controlled i2i pipeline conditioned on the run's base
                # reference image (see character-forge for the pattern).
                # The slot prompt would still drive content; only the
                # pipeline call shape changes.
                image = await run_in_threadpool(
                    model_mgr.generate,
                    prompt=compiled_prompt,
                    width=manifest.params.width,
                    height=manifest.params.height,
                    steps=manifest.params.steps,
                    guidance_scale=manifest.params.guidance,
                    seed=seed_value,
                    negative_prompt=negative_prompt,
                    scheduler=manifest.params.scheduler,
                )
            except Exception as exc:  # noqa: BLE001  (caught broadly to record per-slot failure)
                logger.exception("LoRA tile generation failed for slot %s", key)
                err_text = str(exc)

                def _mark_failed(m: LoraRunManifest, k: str = key, err: str = err_text) -> None:
                    target = m.slots.get(k)
                    if target is None:
                        return
                    target.status = "failed"
                    target.error = err

                update_manifest(outputs_dir, run_id, _mark_failed)
                continue

            run_dir = run_dir_for(outputs_dir, run_id)
            tile_basename = f"{slot_order_index:02d}_{key}"
            png_filename = f"{tile_basename}.png"
            txt_filename = f"{tile_basename}.txt"
            image.save(run_dir / png_filename, format="PNG")
            (run_dir / txt_filename).write_text(slot.tile_text, encoding="utf-8")

            # i2i-extension-seam: this is also where an i2i variant
            # would record per-tile conditioning metadata (e.g. the
            # base reference image used, the i2i strength applied).
            # The manifest schema would gain optional fields rather
            # than a new schema version.
            def _mark_done(
                m: LoraRunManifest,
                k: str = key,
                seed_v: int = seed_value,
                cp: str = compiled_prompt,
                pf: str = png_filename,
                tf: str = txt_filename,
            ) -> None:
                target = m.slots.get(k)
                if target is None:
                    return
                target.status = "done"
                target.error = None
                target.seed = seed_v
                target.compiled_prompt = cp
                target.image_filename = pf
                target.caption_filename = tf

            update_manifest(outputs_dir, run_id, _mark_done)

            completed += 1

        terminal_status: str
        if cancelled:
            terminal_status = "cancelled"
        else:
            terminal_status = "complete"

        def _finalise(m: LoraRunManifest) -> None:
            failed_any = any(m.slots[k].status == "failed" for k in slot_keys if k in m.slots)
            if cancelled:
                m.status = "cancelled"
            elif failed_any:
                m.status = "failed"
            else:
                m.status = "complete"

        manifest = update_manifest(outputs_dir, run_id, _finalise)

        deps.generation_runtime_service.update_generation_status(
            run_id,
            phase=terminal_status,
            message=(
                f"Generated {completed} of {len(slot_keys)} tile(s)."
                if not cancelled
                else f"Stopped after {completed} of {len(slot_keys)} tile(s)."
            ),
            completed_count=completed,
            done=True,
        )
    finally:
        deps.generation_runtime_service.pop_generation_cancel_event(run_id)

    if manifest is None:  # defensive
        raise HTTPException(status_code=500, detail="manifest disappeared during run")

    return {
        "success": True,
        "run_id": run_id,
        "status": manifest.status,
        "completed": completed,
        "requested": len(slot_keys),
        "cancelled": cancelled,
    }

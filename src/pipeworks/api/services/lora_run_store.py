"""On-disk run-manifest storage for the LoRA Dataset tab.

Each run lives in its own directory under ``<outputs_dir>/lora_runs/<run_id>/``
and contains:

- ``manifest.json`` — Pydantic-serialised :class:`LoraRunManifest`,
  written atomically via tmp file plus ``os.replace`` so browser polling
  can safely read while a worker thread updates
- ``NN_<location_key>.png`` — per-tile PNG, written after a successful
  generation. The numeric prefix is the slot order (zero-padded to 2 digits)
- ``NN_<location_key>.txt`` — per-tile caption file, written alongside the PNG
- ``dataset/`` — created on export, contains a curated copy of non-excluded
  PNG/TXT pairs

The store is intentionally reconciliation-friendly: every read prunes
slots whose recorded image file is missing on disk and resets their state
to ``pending``. This keeps the manifest honest if a user rm's a tile
manually, mirroring the gallery store's reconciliation rule.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from pipeworks.api.models_lora import LoraRunManifest

_MANIFEST_FILENAME = "manifest.json"
_DATASET_SUBDIR = "dataset"
# Process-local lock map: one threading.Lock per run_id. Different runs
# can be written concurrently but a single run serialises updates so a
# read-modify-write cycle stays consistent under a single uvicorn worker.
# (Multi-worker deployments would need an inter-process lock; not needed
# for the current single-process FastAPI deploy.)
_run_locks: dict[str, threading.Lock] = {}
_run_locks_guard = threading.Lock()


def _get_run_lock(run_id: str) -> threading.Lock:
    with _run_locks_guard:
        existing = _run_locks.get(run_id)
        if existing is None:
            existing = threading.Lock()
            _run_locks[run_id] = existing
        return existing


def lora_runs_dir(outputs_dir: Path) -> Path:
    """Resolve the parent directory holding all LoRA run subdirs."""
    return outputs_dir / "lora_runs"


def run_dir_for(outputs_dir: Path, run_id: str) -> Path:
    """Resolve the per-run subdirectory."""
    return lora_runs_dir(outputs_dir) / run_id


def manifest_path_for(outputs_dir: Path, run_id: str) -> Path:
    """Resolve the manifest JSON path for one run."""
    return run_dir_for(outputs_dir, run_id) / _MANIFEST_FILENAME


def dataset_dir_for(outputs_dir: Path, run_id: str) -> Path:
    """Resolve the dataset export subdirectory for one run."""
    return run_dir_for(outputs_dir, run_id) / _DATASET_SUBDIR


def new_run_id() -> str:
    """Mint a fresh run identifier."""
    return str(uuid.uuid4())


def ensure_run_dir(outputs_dir: Path, run_id: str) -> Path:
    """Create and return the run directory, parents included."""
    run_dir = run_dir_for(outputs_dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(outputs_dir: Path, manifest: LoraRunManifest) -> None:
    """Persist a manifest atomically.

    Atomicity matters: the generation worker thread updates the manifest
    after each tile completes, while the browser polls
    ``GET /api/lora-dataset/runs/{id}`` for progress. A torn JSON read
    would surface as a 500 in the browser. Writing to a tmp sibling and
    ``os.replace`` keeps the read side either at the previous valid
    snapshot or the new one — never partial.
    """
    run_dir = ensure_run_dir(outputs_dir, manifest.run_id)
    target = run_dir / _MANIFEST_FILENAME
    tmp_path = run_dir / f"{_MANIFEST_FILENAME}.tmp.{os.getpid()}.{threading.get_ident()}"

    payload = manifest.model_dump(mode="json")
    payload["updated_at"] = _now_unix()
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def read_manifest(outputs_dir: Path, run_id: str) -> LoraRunManifest | None:
    """Load and reconcile a manifest against the run directory.

    Returns ``None`` if the manifest is missing or invalid (treated as
    "no such run" rather than 500ing). Reconciliation prunes references
    to tile files that have disappeared from disk and resets the
    affected slots to ``pending`` so the operator can regenerate them.
    """
    target = manifest_path_for(outputs_dir, run_id)
    if not target.exists():
        return None

    try:
        with open(target, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    try:
        manifest = LoraRunManifest.model_validate(raw)
    except ValidationError:
        return None

    if _reconcile_slots_against_disk(outputs_dir, manifest):
        write_manifest(outputs_dir, manifest)
    return manifest


def list_run_ids(outputs_dir: Path) -> list[str]:
    """Return run ids present on disk, newest manifest first.

    Sort key is the manifest's ``updated_at`` (falling back to ``created_at``,
    falling back to directory mtime for legacy/broken runs). Runs whose
    manifest fails to load are skipped silently — the alternative would
    be to surface mojibake to the UI.
    """
    parent = lora_runs_dir(outputs_dir)
    if not parent.exists():
        return []

    candidates: list[tuple[float, str]] = []
    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        manifest = read_manifest(outputs_dir, entry.name)
        if manifest is None:
            continue
        sort_key = manifest.updated_at or manifest.created_at or entry.stat().st_mtime
        candidates.append((sort_key, entry.name))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [run_id for _, run_id in candidates]


def iter_runs(outputs_dir: Path) -> Iterator[LoraRunManifest]:
    """Yield reconciled manifests in the same order as :func:`list_run_ids`."""
    for run_id in list_run_ids(outputs_dir):
        manifest = read_manifest(outputs_dir, run_id)
        if manifest is not None:
            yield manifest


def update_manifest(
    outputs_dir: Path,
    run_id: str,
    mutate: Callable[[LoraRunManifest], None],
) -> LoraRunManifest | None:
    """Apply an in-place mutation to a manifest under the per-run lock.

    Centralises the read-modify-write pattern so callers don't reach for
    the lock map directly. ``mutate`` is called on a freshly loaded
    manifest and is expected to mutate it in place; the new manifest is
    persisted atomically before the function returns.
    """
    lock = _get_run_lock(run_id)
    with lock:
        manifest = read_manifest(outputs_dir, run_id)
        if manifest is None:
            return None
        mutate(manifest)
        write_manifest(outputs_dir, manifest)
        return manifest


def _reconcile_slots_against_disk(
    outputs_dir: Path,
    manifest: LoraRunManifest,
) -> bool:
    """Reset slot fields whose recorded files no longer exist on disk.

    Returns ``True`` if the manifest was modified and should be written
    back. The reconciliation rule is intentionally conservative: only
    file references are validated; the slot's ``excluded`` flag and
    ``error`` text remain untouched.
    """
    changed = False
    run_dir = run_dir_for(outputs_dir, manifest.run_id)
    for slot in manifest.slots.values():
        if slot.image_filename:
            png_path = run_dir / slot.image_filename
            if not png_path.exists():
                slot.image_filename = None
                slot.status = "pending"
                slot.seed = None
                changed = True
        if slot.caption_filename:
            txt_path = run_dir / slot.caption_filename
            if not txt_path.exists():
                slot.caption_filename = None
                changed = True
    return changed


def _now_unix() -> float:
    """Current UNIX timestamp in seconds, monotonic-friendly across writes."""
    return datetime.now(tz=UTC).timestamp()

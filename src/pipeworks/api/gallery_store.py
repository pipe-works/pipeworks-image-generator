"""Gallery metadata storage helpers for the Pipe-Works API.

This module isolates the gallery JSON persistence logic from
``pipeworks.api.main`` so route handlers can focus on HTTP concerns while the
file-backed gallery store remains testable as a small unit.

The gallery is intentionally simple:

- metadata lives in a single ``gallery.json`` file
- image files live in the gallery directory
- list order is reverse-chronological (newest first)

Because users can also manipulate the gallery directory manually outside the
application, the helpers in this module reconcile the metadata file against the
image directory whenever the gallery is loaded.  Missing image files are
treated as deleted items and pruned from ``gallery.json`` automatically.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def load_gallery_entries(gallery_db: Path, gallery_dir: Path) -> list[dict]:
    """Load the gallery metadata and reconcile it against files on disk.

    The gallery metadata is authoritative for image ordering and generation
    metadata, but the image file itself must still exist for the entry to be
    usable.  Users may remove files directly from the gallery directory without
    going through the API, so every load reconciles the JSON metadata against
    the file system and removes stale entries automatically.

    The reconciliation rule is intentionally conservative:

    - if the JSON file is missing or invalid, return an empty gallery
    - if an entry has no filename, drop it
    - if the referenced image file does not exist, drop the entry

    When stale entries are removed, the cleaned list is persisted immediately
    so future reads observe the corrected counts and pagination.

    Args:
        gallery_db: Path to ``gallery.json``.
        gallery_dir: Directory that should contain the image files.

    Returns:
        List of surviving gallery entry dictionaries in persisted order.
    """
    if gallery_db.exists():
        try:
            with open(gallery_db, encoding="utf-8") as handle:
                raw_entries = json.load(handle)
        except Exception:
            raw_entries = []
    else:
        raw_entries = []

    if not isinstance(raw_entries, list):
        raw_entries = []

    cleaned_entries: list[dict] = []

    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue

        filename = entry.get("filename")

        # Entries without a filename cannot be mapped back to an on-disk asset,
        # so they are treated as irrecoverably stale metadata.
        if not filename:
            continue

        filepath = gallery_dir / filename

        # Reconcile metadata with the real gallery directory.  If the file was
        # removed manually, the metadata should no longer contribute to counts,
        # filters, pagination, or stats.
        if not filepath.exists():
            continue

        cleaned_entries.append(entry)

    if cleaned_entries != raw_entries:
        save_gallery_entries(gallery_db, cleaned_entries)

    return cleaned_entries


def save_gallery_entries(gallery_db: Path, entries: list[dict]) -> None:
    """Persist the gallery metadata list to disk.

    Args:
        gallery_db: Path to ``gallery.json``.
        entries: Gallery entry dictionaries to persist.
    """
    with open(gallery_db, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def filter_gallery_entries(
    entries: list[dict],
    *,
    favourites_only: bool = False,
    model_id: str | None = None,
) -> list[dict]:
    """Apply favourite and model filters to gallery entries.

    Args:
        entries: Source gallery entries.
        favourites_only: Whether to keep only favourited entries.
        model_id: Optional model identifier to filter by.

    Returns:
        Filtered gallery entries in their original order.
    """
    filtered_entries = entries

    if favourites_only:
        filtered_entries = [entry for entry in filtered_entries if entry.get("is_favourite")]

    if model_id:
        filtered_entries = [
            entry for entry in filtered_entries if entry.get("model_id") == model_id
        ]

    return filtered_entries


def paginate_gallery_entries(entries: list[dict], page: int, per_page: int) -> dict:
    """Paginate gallery entries and clamp the requested page to valid bounds.

    Clamping matters after deletes.  If a user is viewing the last gallery page
    and removes the final image on that page, the previous page becomes the new
    last page.  Returning the clamped page keeps the frontend and backend in
    agreement about the correct image count and page range.

    Args:
        entries: Filtered gallery entries.
        page: Requested one-based page number.
        per_page: Requested items per page.

    Returns:
        Dictionary containing ``total``, ``page``, ``per_page``, ``pages``, and
        ``images`` for the resolved page.
    """
    total = len(entries)
    pages = (total + per_page - 1) // per_page if total > 0 else 1
    resolved_page = min(max(page, 1), pages)

    start = (resolved_page - 1) * per_page
    end = start + per_page

    return {
        "total": total,
        "page": resolved_page,
        "per_page": per_page,
        "pages": pages,
        "images": entries[start:end],
    }


def group_entries_into_runs(entries: list[dict]) -> list[dict]:
    """Group gallery entries by generation run (``batch_seed``).

    Each run object aggregates all surviving images that share the same
    ``batch_seed``.  Legacy entries that lack a ``batch_seed`` are assigned a
    synthetic key so they appear as standalone single-image runs.

    Runs are sorted by ``created_at`` descending (newest first).  Within each
    run, images are sorted by ``batch_index`` ascending.

    Args:
        entries: Flat gallery entries (already filtered/reconciled).

    Returns:
        List of run dictionaries, each containing:

        - ``batch_seed`` — the shared seed (int or synthetic string).
        - ``model_id`` / ``model_label`` — from the first entry in the run.
        - ``created_at`` — latest timestamp among the run's images.
        - ``date`` — ISO date string derived from ``created_at``.
        - ``total_images`` — surviving image count.
        - ``has_favourites`` — whether any image in the run is favourited.
        - ``images`` — all entries in the run, sorted by ``batch_index``.
    """
    runs_map: dict[int | str, list[dict]] = {}

    for entry in entries:
        key = entry.get("batch_seed")
        if key is None:
            key = f"_orphan_{entry['id']}"
        runs_map.setdefault(key, []).append(entry)

    runs: list[dict] = []
    for batch_seed, images in runs_map.items():
        images.sort(key=lambda e: e.get("batch_index", 0))
        latest_ts = max(e.get("created_at", 0) for e in images)
        date_str = (
            datetime.fromtimestamp(latest_ts, tz=UTC).strftime("%Y-%m-%d")
            if latest_ts
            else "unknown"
        )
        first = images[0]
        runs.append(
            {
                "batch_seed": batch_seed,
                "model_id": first.get("model_id", "unknown"),
                "model_label": first.get("model_label", "Unknown"),
                "created_at": latest_ts,
                "date": date_str,
                "total_images": len(images),
                "has_favourites": any(e.get("is_favourite") for e in images),
                "images": images,
            }
        )

    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs


def paginate_runs(runs: list[dict], page: int, per_page: int) -> dict:
    """Paginate a list of runs and clamp the page to valid bounds.

    Args:
        runs: Grouped run dictionaries.
        page: Requested one-based page number.
        per_page: Runs per page.

    Returns:
        Dictionary with ``total_runs``, ``total_images``, ``page``,
        ``per_page``, ``pages``, and ``runs``.
    """
    total_runs = len(runs)
    total_images = sum(r["total_images"] for r in runs)
    pages = (total_runs + per_page - 1) // per_page if total_runs > 0 else 1
    resolved_page = min(max(page, 1), pages)

    start = (resolved_page - 1) * per_page
    end = start + per_page

    return {
        "total_runs": total_runs,
        "total_images": total_images,
        "page": resolved_page,
        "per_page": per_page,
        "pages": pages,
        "runs": runs[start:end],
    }


def get_run_entries(entries: list[dict], batch_seed: int) -> list[dict]:
    """Return all gallery entries for a specific run, sorted by batch_index.

    Args:
        entries: Full gallery entry list.
        batch_seed: The ``batch_seed`` identifying the run.

    Returns:
        Matching entries sorted by ``batch_index``, or an empty list.
    """
    matching = [e for e in entries if e.get("batch_seed") == batch_seed]
    matching.sort(key=lambda e: e.get("batch_index", 0))
    return matching

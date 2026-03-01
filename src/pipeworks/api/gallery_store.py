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

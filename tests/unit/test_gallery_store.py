"""Unit tests for gallery_store grouping and pagination helpers."""

from __future__ import annotations

import time

from pipeworks.api.gallery_store import (
    get_run_entries,
    group_entries_into_runs,
    paginate_runs,
)


def _make_entry(
    img_id: str,
    batch_seed: int | None = None,
    batch_index: int = 0,
    batch_size: int = 1,
    model_id: str = "z-image-turbo",
    model_label: str = "Z-Image Turbo",
    created_at: float | None = None,
    is_favourite: bool = False,
) -> dict:
    """Create a minimal gallery entry for testing."""
    return {
        "id": img_id,
        "filename": f"{img_id}.png",
        "batch_seed": batch_seed,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "model_id": model_id,
        "model_label": model_label,
        "created_at": created_at or time.time(),
        "is_favourite": is_favourite,
    }


class TestGroupEntriesIntoRuns:
    """Tests for group_entries_into_runs()."""

    def test_empty_list(self):
        assert group_entries_into_runs([]) == []

    def test_single_image_runs(self):
        entries = [
            _make_entry("a", batch_seed=100, created_at=1000.0),
            _make_entry("b", batch_seed=200, created_at=2000.0),
        ]
        runs = group_entries_into_runs(entries)

        assert len(runs) == 2
        # Newest first.
        assert runs[0]["batch_seed"] == 200
        assert runs[1]["batch_seed"] == 100

    def test_multi_image_run_grouped(self):
        entries = [
            _make_entry("a", batch_seed=100, batch_index=0, created_at=1000.0),
            _make_entry("b", batch_seed=100, batch_index=1, created_at=1001.0),
            _make_entry("c", batch_seed=100, batch_index=2, created_at=1002.0),
        ]
        runs = group_entries_into_runs(entries)

        assert len(runs) == 1
        run = runs[0]
        assert run["batch_seed"] == 100
        assert run["total_images"] == 3
        assert [img["id"] for img in run["images"]] == ["a", "b", "c"]

    def test_images_sorted_by_batch_index(self):
        entries = [
            _make_entry("c", batch_seed=100, batch_index=2, created_at=1000.0),
            _make_entry("a", batch_seed=100, batch_index=0, created_at=1000.0),
            _make_entry("b", batch_seed=100, batch_index=1, created_at=1000.0),
        ]
        runs = group_entries_into_runs(entries)

        assert [img["id"] for img in runs[0]["images"]] == ["a", "b", "c"]

    def test_runs_sorted_by_date_descending(self):
        entries = [
            _make_entry("a", batch_seed=100, created_at=1000.0),
            _make_entry("b", batch_seed=200, created_at=3000.0),
            _make_entry("c", batch_seed=300, created_at=2000.0),
        ]
        runs = group_entries_into_runs(entries)

        assert [r["batch_seed"] for r in runs] == [200, 300, 100]

    def test_legacy_entries_without_batch_seed(self):
        entries = [
            _make_entry("a", batch_seed=None, created_at=1000.0),
            _make_entry("b", batch_seed=None, created_at=2000.0),
        ]
        runs = group_entries_into_runs(entries)

        # Each orphan becomes its own run.
        assert len(runs) == 2
        assert runs[0]["total_images"] == 1
        assert runs[1]["total_images"] == 1

    def test_has_favourites_flag(self):
        entries = [
            _make_entry("a", batch_seed=100, is_favourite=False, created_at=1000.0),
            _make_entry("b", batch_seed=100, is_favourite=True, created_at=1000.0),
        ]
        runs = group_entries_into_runs(entries)

        assert runs[0]["has_favourites"] is True

    def test_no_favourites_flag(self):
        entries = [
            _make_entry("a", batch_seed=100, is_favourite=False, created_at=1000.0),
        ]
        runs = group_entries_into_runs(entries)

        assert runs[0]["has_favourites"] is False

    def test_partial_run_after_deletes(self):
        """A 4-image run with 2 deleted shows total_images=2."""
        entries = [
            _make_entry("a", batch_seed=100, batch_index=0, batch_size=4, created_at=1000.0),
            _make_entry("c", batch_seed=100, batch_index=2, batch_size=4, created_at=1000.0),
        ]
        runs = group_entries_into_runs(entries)

        assert runs[0]["total_images"] == 2

    def test_date_field_derived_from_created_at(self):
        # 2026-03-05 00:00:00 UTC
        entries = [_make_entry("a", batch_seed=100, created_at=1772668800.0)]
        runs = group_entries_into_runs(entries)

        assert runs[0]["date"] == "2026-03-05"

    def test_model_metadata_from_first_entry(self):
        entries = [
            _make_entry(
                "a",
                batch_seed=100,
                model_id="sdxl",
                model_label="SDXL",
                created_at=1000.0,
            ),
            _make_entry(
                "b",
                batch_seed=100,
                model_id="sdxl",
                model_label="SDXL",
                created_at=1000.0,
            ),
        ]
        runs = group_entries_into_runs(entries)

        assert runs[0]["model_id"] == "sdxl"
        assert runs[0]["model_label"] == "SDXL"


class TestPaginateRuns:
    """Tests for paginate_runs()."""

    def test_single_page(self):
        runs = [{"total_images": 3}, {"total_images": 5}]
        result = paginate_runs(runs, page=1, per_page=10)

        assert result["total_runs"] == 2
        assert result["total_images"] == 8
        assert result["page"] == 1
        assert result["pages"] == 1
        assert len(result["runs"]) == 2

    def test_multiple_pages(self):
        runs = [{"total_images": 1} for _ in range(5)]
        result = paginate_runs(runs, page=2, per_page=2)

        assert result["total_runs"] == 5
        assert result["page"] == 2
        assert result["pages"] == 3
        assert len(result["runs"]) == 2

    def test_page_clamped_to_max(self):
        runs = [{"total_images": 1}]
        result = paginate_runs(runs, page=99, per_page=10)

        assert result["page"] == 1

    def test_page_clamped_to_min(self):
        runs = [{"total_images": 1}]
        result = paginate_runs(runs, page=0, per_page=10)

        assert result["page"] == 1

    def test_empty_runs(self):
        result = paginate_runs([], page=1, per_page=10)

        assert result["total_runs"] == 0
        assert result["total_images"] == 0
        assert result["pages"] == 1


class TestGetRunEntries:
    """Tests for get_run_entries()."""

    def test_returns_matching_entries(self):
        entries = [
            _make_entry("a", batch_seed=100, batch_index=0),
            _make_entry("b", batch_seed=200, batch_index=0),
            _make_entry("c", batch_seed=100, batch_index=1),
        ]
        result = get_run_entries(entries, 100)

        assert len(result) == 2
        assert [e["id"] for e in result] == ["a", "c"]

    def test_sorted_by_batch_index(self):
        entries = [
            _make_entry("c", batch_seed=100, batch_index=2),
            _make_entry("a", batch_seed=100, batch_index=0),
            _make_entry("b", batch_seed=100, batch_index=1),
        ]
        result = get_run_entries(entries, 100)

        assert [e["id"] for e in result] == ["a", "b", "c"]

    def test_returns_empty_for_unknown_seed(self):
        entries = [_make_entry("a", batch_seed=100)]
        result = get_run_entries(entries, 999)

        assert result == []

"""Gallery and archive API routes."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from pipeworks.api.gallery_store import (
    filter_gallery_entries,
    get_run_entries,
    group_entries_into_runs,
    load_gallery_entries,
    paginate_gallery_entries,
    paginate_runs,
    save_gallery_entries,
)
from pipeworks.api.models import BulkDeleteRequest, BulkZipRequest, FavouriteRequest
from pipeworks.api.services.runtime_policy import RuntimePolicyService
from pipeworks.api.services.zip_metadata import build_zip_metadata_for_entry


@dataclass(frozen=True, slots=True)
class GalleryRouterDependencies:
    """Dependencies required by gallery and zip routes."""

    data_dir: Callable[[], Path]
    gallery_dir: Callable[[], Path]
    gallery_db: Callable[[], Path]
    runtime_policy_service: RuntimePolicyService
    normalize_base_url: Callable[[str | None], str]


def create_gallery_router(deps: GalleryRouterDependencies) -> APIRouter:
    """Build APIRouter for gallery/read/write/archive endpoints."""
    router = APIRouter()

    @router.get("/api/gallery")
    async def get_gallery(
        page: int = 1,
        per_page: int = 20,
        favourites_only: bool = False,
        model_id: str | None = None,
    ) -> dict:
        gallery = load_gallery_entries(deps.gallery_db(), deps.gallery_dir())
        filtered_gallery = filter_gallery_entries(
            gallery,
            favourites_only=favourites_only,
            model_id=model_id,
        )
        return paginate_gallery_entries(filtered_gallery, page, per_page)

    @router.get("/api/gallery/runs")
    async def get_gallery_runs(
        page: int = 1,
        per_page: int = 20,
        model_id: str | None = None,
        thumbnail_limit: int = 6,
    ) -> dict:
        gallery = load_gallery_entries(deps.gallery_db(), deps.gallery_dir())
        filtered = filter_gallery_entries(gallery, model_id=model_id)
        runs = group_entries_into_runs(filtered)

        for run in runs:
            run["thumbnail_count"] = min(len(run["images"]), thumbnail_limit)
            run["images"] = run["images"][:thumbnail_limit]

        return paginate_runs(runs, page, per_page)

    @router.get("/api/gallery/runs/{batch_seed}")
    async def get_gallery_run(batch_seed: int) -> dict:
        gallery = load_gallery_entries(deps.gallery_db(), deps.gallery_dir())
        run_entries = get_run_entries(gallery, batch_seed)
        if not run_entries:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "batch_seed": batch_seed,
            "total_images": len(run_entries),
            "images": run_entries,
        }

    @router.get("/api/gallery/runs/{batch_seed}/zip")
    async def download_run_zip(batch_seed: int) -> Response:
        data_dir = deps.data_dir()
        gallery_dir = deps.gallery_dir()
        gallery = load_gallery_entries(deps.gallery_db(), gallery_dir)
        run_entries = get_run_entries(gallery, batch_seed)

        if not run_entries:
            raise HTTPException(status_code=404, detail="Run not found")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for entry in run_entries:
                image_path = gallery_dir / entry["filename"]
                if not image_path.exists():
                    continue

                id_short = entry["id"][:8]
                metadata = build_zip_metadata_for_entry(
                    entry=entry,
                    data_dir=data_dir,
                    runtime_policy_service=deps.runtime_policy_service,
                    normalize_base_url=deps.normalize_base_url,
                )

                archive.write(image_path, f"pipeworks_{id_short}.png")
                archive.writestr(
                    f"pipeworks_{id_short}_metadata.json",
                    json.dumps(metadata, indent=2),
                )
        buffer.seek(0)

        zip_filename = f"pipeworks_run_{batch_seed}.zip"
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        )

    @router.get("/api/gallery/{image_id}")
    async def get_image(image_id: str) -> dict:
        gallery = load_gallery_entries(deps.gallery_db(), deps.gallery_dir())
        entry = next((g for g in gallery if g["id"] == image_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Image not found")
        return entry

    @router.post("/api/gallery/favourite")
    async def toggle_favourite(req: FavouriteRequest) -> dict:
        gallery_db = deps.gallery_db()
        gallery_dir = deps.gallery_dir()
        gallery = load_gallery_entries(gallery_db, gallery_dir)
        entry = next((g for g in gallery if g["id"] == req.image_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Image not found")

        entry["is_favourite"] = req.is_favourite
        save_gallery_entries(gallery_db, gallery)

        return {"success": True, "id": req.image_id, "is_favourite": req.is_favourite}

    @router.post("/api/gallery/bulk-delete")
    async def bulk_delete_images(req: BulkDeleteRequest) -> dict:
        gallery_db = deps.gallery_db()
        gallery_dir = deps.gallery_dir()
        gallery = load_gallery_entries(gallery_db, gallery_dir)

        gallery_by_id: dict[str, dict] = {g["id"]: g for g in gallery}

        deleted: list[str] = []
        not_found: list[str] = []

        for image_id in req.image_ids:
            entry = gallery_by_id.get(image_id)
            if not entry:
                not_found.append(image_id)
                continue

            filepath = gallery_dir / entry["filename"]
            if filepath.exists():
                filepath.unlink()

            deleted.append(image_id)

        deleted_set = set(deleted)
        gallery = [g for g in gallery if g["id"] not in deleted_set]
        save_gallery_entries(gallery_db, gallery)

        return {"success": True, "deleted": deleted, "not_found": not_found}

    @router.post("/api/gallery/bulk-zip")
    async def bulk_zip_images(req: BulkZipRequest) -> Response:
        data_dir = deps.data_dir()
        gallery_dir = deps.gallery_dir()
        gallery = load_gallery_entries(deps.gallery_db(), gallery_dir)
        gallery_by_id: dict[str, dict] = {g["id"]: g for g in gallery}

        buffer = io.BytesIO()
        included = 0
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for image_id in req.image_ids:
                entry = gallery_by_id.get(image_id)
                if not entry:
                    continue

                image_path = gallery_dir / entry["filename"]
                if not image_path.exists():
                    continue

                id_short = entry["id"][:8]
                metadata = build_zip_metadata_for_entry(
                    entry=entry,
                    data_dir=data_dir,
                    runtime_policy_service=deps.runtime_policy_service,
                    normalize_base_url=deps.normalize_base_url,
                )

                archive.write(image_path, f"pipeworks_{id_short}.png")
                archive.writestr(
                    f"pipeworks_{id_short}_metadata.json",
                    json.dumps(metadata, indent=2),
                )
                included += 1

        if included == 0:
            raise HTTPException(status_code=404, detail="No matching images found")

        buffer.seek(0)
        zip_filename = f"pipeworks_selected_{included}.zip"
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        )

    @router.delete("/api/gallery/{image_id}")
    async def delete_image(image_id: str) -> dict:
        gallery_db = deps.gallery_db()
        gallery_dir = deps.gallery_dir()
        gallery = load_gallery_entries(gallery_db, gallery_dir)
        entry = next((g for g in gallery if g["id"] == image_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Image not found")

        filepath = gallery_dir / entry["filename"]
        if filepath.exists():
            filepath.unlink()

        gallery = [g for g in gallery if g["id"] != image_id]
        save_gallery_entries(gallery_db, gallery)

        return {"success": True, "deleted": image_id}

    @router.get("/api/gallery/{image_id}/prompt")
    async def get_image_prompt(image_id: str) -> dict:
        gallery = load_gallery_entries(deps.gallery_db(), deps.gallery_dir())
        entry = next((g for g in gallery if g["id"] == image_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Image not found")
        return {
            "id": image_id,
            "compiled_prompt": entry.get("compiled_prompt", ""),
            "model_id": entry.get("model_id"),
            "seed": entry.get("seed"),
            "steps": entry.get("steps"),
            "guidance": entry.get("guidance"),
            "width": entry.get("width"),
            "height": entry.get("height"),
        }

    @router.get("/api/gallery/{image_id}/zip")
    async def download_image_zip(image_id: str) -> Response:
        data_dir = deps.data_dir()
        gallery_dir = deps.gallery_dir()
        gallery = load_gallery_entries(deps.gallery_db(), gallery_dir)
        entry = next((g for g in gallery if g["id"] == image_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Image not found")

        image_path = gallery_dir / entry["filename"]
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found on disk")

        id_short = image_id[:8]
        metadata = build_zip_metadata_for_entry(
            entry=entry,
            data_dir=data_dir,
            runtime_policy_service=deps.runtime_policy_service,
            normalize_base_url=deps.normalize_base_url,
        )

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(image_path, f"pipeworks_{id_short}.png")
            archive.writestr(
                f"pipeworks_{id_short}_metadata.json",
                json.dumps(metadata, indent=2),
            )
        buffer.seek(0)

        zip_filename = f"pipeworks_{id_short}.zip"
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        )

    @router.get("/api/stats")
    async def get_stats() -> dict:
        gallery = load_gallery_entries(deps.gallery_db(), deps.gallery_dir())
        total = len(gallery)
        favourites = sum(1 for g in gallery if g.get("is_favourite"))

        model_counts: dict[str, int] = {}
        for entry in gallery:
            model_id = entry.get("model_id", "unknown")
            model_counts[model_id] = model_counts.get(model_id, 0) + 1

        return {
            "total_images": total,
            "total_favourites": favourites,
            "model_counts": model_counts,
        }

    return router

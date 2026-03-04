"""Unit tests for the GET /api/gallery/{image_id}/zip endpoint.

These tests verify that the zip download endpoint returns a valid zip
archive containing the expected PNG image and structured metadata JSON.
"""

from __future__ import annotations

import io
import json
import zipfile


class TestZipEndpoint:
    """Test GET /api/gallery/{image_id}/zip — zip archive download."""

    def test_zip_returns_valid_archive(self, test_client, sample_gallery):
        """The endpoint should return a valid zip with PNG and metadata JSON."""
        image = sample_gallery[0]
        resp = test_client.get(f"/api/gallery/{image['id']}/zip")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert "attachment" in resp.headers.get("content-disposition", "")

        # Verify the zip contains exactly two files.
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            assert len(names) == 2

            id_short = image["id"][:8]
            assert f"pipeworks_{id_short}.png" in names
            assert f"pipeworks_{id_short}_metadata.json" in names

    def test_zip_metadata_structure(self, test_client, sample_gallery):
        """The metadata JSON should contain all expected top-level sections."""
        image = sample_gallery[0]
        resp = test_client.get(f"/api/gallery/{image['id']}/zip")

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            id_short = image["id"][:8]
            metadata_bytes = zf.read(f"pipeworks_{id_short}_metadata.json")
            metadata = json.loads(metadata_bytes)

        # Top-level keys.
        assert metadata["id"] == image["id"]
        assert "model" in metadata
        assert "prompt" in metadata
        assert "generation" in metadata
        assert "batch" in metadata
        assert "created_at" in metadata
        assert "is_favourite" in metadata

        # Model section.
        assert metadata["model"]["id"] == image["model_id"]

        # Prompt section — top-level compiled string.
        assert metadata["prompt"]["compiled"] == image["compiled_prompt"]

        # Prompt section — prepend sub-object.
        prepend = metadata["prompt"]["prepend"]
        assert prepend["mode"] == image.get("prepend_mode", "template")
        assert prepend["preset_id"] == image["prepend_prompt_id"]
        assert "preset_label" in prepend
        assert "text" in prepend

        # Prompt section — main sub-object (same shape as prepend/append).
        main = metadata["prompt"]["main"]
        assert main["mode"] == image["prompt_mode"]
        assert main["preset_id"] == image.get("automated_prompt_id")
        assert "preset_label" in main
        assert "text" in main

        # Prompt section — append sub-object.
        append = metadata["prompt"]["append"]
        assert append["mode"] == image.get("append_mode", "template")
        assert append["preset_id"] == image["append_prompt_id"]
        assert "preset_label" in append
        assert "text" in append

        # Generation section.
        assert metadata["generation"]["width"] == image["width"]
        assert metadata["generation"]["height"] == image["height"]
        assert metadata["generation"]["steps"] == image["steps"]
        assert metadata["generation"]["guidance"] == image["guidance"]
        assert metadata["generation"]["seed"] == image["seed"]
        assert metadata["generation"]["negative_prompt"] == image.get("negative_prompt")

        # Batch section.
        assert metadata["batch"]["index"] == image["batch_index"]
        assert metadata["batch"]["size"] == image["batch_size"]
        assert metadata["batch"]["seed"] == image["batch_seed"]

    def test_zip_png_content_matches_gallery_file(self, test_client, sample_gallery):
        """The PNG inside the zip should match the original gallery file."""
        image = sample_gallery[0]
        resp = test_client.get(f"/api/gallery/{image['id']}/zip")

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            id_short = image["id"][:8]
            png_bytes = zf.read(f"pipeworks_{id_short}.png")

        # The PNG should be non-empty and start with the PNG magic bytes.
        assert len(png_bytes) > 0
        assert png_bytes[:4] == b"\x89PNG"

    def test_zip_404_for_missing_image(self, test_client, sample_gallery):
        """A missing image ID should return 404."""
        resp = test_client.get("/api/gallery/nonexistent-id/zip")
        assert resp.status_code == 404

    def test_zip_content_disposition_filename(self, test_client, sample_gallery):
        """The Content-Disposition header should contain the expected filename."""
        image = sample_gallery[0]
        resp = test_client.get(f"/api/gallery/{image['id']}/zip")

        id_short = image["id"][:8]
        expected_filename = f"pipeworks_{id_short}.zip"
        assert expected_filename in resp.headers["content-disposition"]

    def test_zip_favourite_status_reflected(self, test_client, sample_gallery):
        """The metadata should reflect the current favourite status."""
        # sample_gallery[0] is favourited (is_favourite=True).
        image = sample_gallery[0]
        resp = test_client.get(f"/api/gallery/{image['id']}/zip")

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            id_short = image["id"][:8]
            metadata = json.loads(zf.read(f"pipeworks_{id_short}_metadata.json"))

        assert metadata["is_favourite"] is True

        # sample_gallery[1] is not favourited.
        image2 = sample_gallery[1]
        resp2 = test_client.get(f"/api/gallery/{image2['id']}/zip")

        with zipfile.ZipFile(io.BytesIO(resp2.content)) as zf2:
            id_short2 = image2["id"][:8]
            metadata2 = json.loads(zf2.read(f"pipeworks_{id_short2}_metadata.json"))

        assert metadata2["is_favourite"] is False

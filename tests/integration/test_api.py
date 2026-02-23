"""Integration tests for pipeworks.api.main — FastAPI REST API endpoints.

All tests use the FastAPI TestClient with a mocked ModelManager so that no
real model loading or GPU access occurs.  Tests cover every endpoint:

- ``GET /`` — HTML page serving.
- ``GET /api/config`` — Configuration delivery.
- ``POST /api/generate`` — Batch image generation.
- ``POST /api/prompt/compile`` — Prompt preview.
- ``GET /api/gallery`` — Paginated gallery listing with filters.
- ``GET /api/gallery/{id}`` — Single gallery entry.
- ``GET /api/gallery/{id}/prompt`` — Prompt metadata.
- ``POST /api/gallery/favourite`` — Favourite toggling.
- ``DELETE /api/gallery/{id}`` — Image deletion.
- ``GET /api/stats`` — Gallery statistics.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Index page tests.
# ---------------------------------------------------------------------------


class TestIndexPage:
    """Test GET / — main HTML page."""

    def test_index_returns_html(self, test_client):
        """GET / should return 200 with HTML content."""
        resp = test_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Pipe-Works Image Generator" in resp.text


# ---------------------------------------------------------------------------
# Configuration endpoint tests.
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Test GET /api/config — application configuration."""

    def test_config_returns_version(self, test_client):
        """Response should include a version string."""
        resp = test_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data

    def test_config_returns_models(self, test_client):
        """Response should include a models list."""
        resp = test_client.get("/api/config")
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) >= 1

    def test_config_returns_prompts(self, test_client):
        """Response should include all three prompt categories."""
        resp = test_client.get("/api/config")
        data = resp.json()
        assert "prepend_prompts" in data
        assert "automated_prompts" in data
        assert "append_prompts" in data


# ---------------------------------------------------------------------------
# Generation endpoint tests.
# ---------------------------------------------------------------------------


class TestGenerate:
    """Test POST /api/generate — image generation."""

    def _make_generate_payload(self, **overrides) -> dict:
        """Build a valid generate request payload with optional overrides.

        Args:
            **overrides: Fields to override in the default payload.

        Returns:
            Dictionary suitable for ``POST /api/generate``.
        """
        payload = {
            "model_id": "z-image-turbo",
            "prepend_prompt_id": "none",
            "prompt_mode": "manual",
            "manual_prompt": "A goblin workshop.",
            "aspect_ratio_id": "1:1",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance": 0.0,
            "batch_size": 1,
        }
        payload.update(overrides)
        return payload

    def test_generate_success(self, test_client):
        """A valid generate request should return 200 with image data."""
        resp = test_client.post("/api/generate", json=self._make_generate_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["images"]) == 1
        assert "compiled_prompt" in data

    def test_generate_batch(self, test_client):
        """Batch generation should return the correct number of images."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(batch_size=3),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["images"]) == 3

    def test_generate_seeds_increment(self, test_client):
        """Each image in a batch should have an incrementing seed."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(batch_size=3, seed=100),
        )
        data = resp.json()
        seeds = [img["seed"] for img in data["images"]]
        assert seeds == [100, 101, 102]

    def test_generate_unknown_model(self, test_client):
        """An unknown model_id should return 400."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(model_id="nonexistent-model"),
        )
        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["detail"]

    def test_generate_manual_missing_prompt(self, test_client):
        """Manual mode without a prompt should return 400."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                prompt_mode="manual",
                manual_prompt="",
            ),
        )
        assert resp.status_code == 400
        assert "manual_prompt" in resp.json()["detail"]

    def test_generate_automated_mode(self, test_client):
        """Automated mode with a valid preset should succeed."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                prompt_mode="automated",
                automated_prompt_id="goblin-workshop",
                manual_prompt=None,
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_generate_automated_missing_id(self, test_client):
        """Automated mode without automated_prompt_id should return 400."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                prompt_mode="automated",
                automated_prompt_id=None,
            ),
        )
        assert resp.status_code == 400

    def test_generate_invalid_prompt_mode(self, test_client):
        """An invalid prompt_mode should return 400."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(prompt_mode="invalid"),
        )
        assert resp.status_code == 400
        assert "prompt_mode" in resp.json()["detail"]

    def test_generate_invalid_batch_size_zero(self, test_client):
        """batch_size of 0 should return 400."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(batch_size=0),
        )
        assert resp.status_code == 400

    def test_generate_invalid_batch_size_too_large(self, test_client):
        """batch_size above 16 should return 400."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(batch_size=17),
        )
        assert resp.status_code == 400

    def test_generate_with_prepend_prompt(self, test_client):
        """A valid prepend_prompt_id should be resolved into the compiled prompt."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(prepend_prompt_id="oil-painting"),
        )
        assert resp.status_code == 200
        data = resp.json()
        # The compiled prompt should contain the oil painting style text.
        assert "oil painting" in data["compiled_prompt"].lower()

    def test_generate_with_append_prompt(self, test_client):
        """A valid append_prompt_id should be resolved into the compiled prompt."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(append_prompt_id="high-detail"),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "8K" in data["compiled_prompt"]

    def test_generate_manual_prepend(self, test_client):
        """Manual prepend mode should use the free-text prepend value."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                prepend_mode="manual",
                manual_prepend="Watercolour painting style.",
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Watercolour painting style." in data["compiled_prompt"]

    def test_generate_manual_append(self, test_client):
        """Manual append mode should use the free-text append value."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                append_mode="manual",
                manual_append="Cinematic lighting, dramatic shadows.",
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Cinematic lighting, dramatic shadows." in data["compiled_prompt"]

    def test_generate_manual_prepend_and_append(self, test_client):
        """Both manual prepend and append should appear in the compiled prompt."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                prepend_mode="manual",
                manual_prepend="Ink sketch style.",
                append_mode="manual",
                manual_append="High contrast.",
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Ink sketch style." in data["compiled_prompt"]
        assert "High contrast." in data["compiled_prompt"]

    def test_generate_with_scheduler(self, test_client):
        """Scheduler should be passed through and stored in gallery metadata."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(scheduler="dpmpp-2m-karras"),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["images"][0]["scheduler"] == "dpmpp-2m-karras"

    def test_generate_without_scheduler(self, test_client):
        """Omitting scheduler should store None in gallery metadata."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["images"][0]["scheduler"] is None

    def test_generate_template_mode_backward_compat(self, test_client):
        """Default template mode (no prepend_mode/append_mode) should still work."""
        resp = test_client.post(
            "/api/generate",
            json=self._make_generate_payload(
                prepend_prompt_id="oil-painting",
                append_prompt_id="high-detail",
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "oil painting" in data["compiled_prompt"].lower()
        assert "8K" in data["compiled_prompt"]


# ---------------------------------------------------------------------------
# Prompt compilation endpoint tests.
# ---------------------------------------------------------------------------


class TestPromptCompile:
    """Test POST /api/prompt/compile — prompt preview."""

    def test_compile_manual_prompt(self, test_client):
        """Manual prompt should compile with boilerplate sections."""
        resp = test_client.post(
            "/api/prompt/compile",
            json={
                "model_id": "z-image-turbo",
                "prepend_prompt_id": "none",
                "prompt_mode": "manual",
                "manual_prompt": "A test scene.",
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "compiled_prompt" in data
        assert "A test scene." in data["compiled_prompt"]
        assert "Main Scene:" in data["compiled_prompt"]

    def test_compile_automated_prompt(self, test_client):
        """Automated prompt should resolve the preset value."""
        resp = test_client.post(
            "/api/prompt/compile",
            json={
                "model_id": "z-image-turbo",
                "prepend_prompt_id": "none",
                "prompt_mode": "automated",
                "automated_prompt_id": "goblin-workshop",
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "goblin" in data["compiled_prompt"].lower()

    def test_compile_manual_prepend(self, test_client):
        """Manual prepend mode should include free-text prepend in compiled prompt."""
        resp = test_client.post(
            "/api/prompt/compile",
            json={
                "model_id": "z-image-turbo",
                "prepend_mode": "manual",
                "manual_prepend": "Pencil sketch style.",
                "prompt_mode": "manual",
                "manual_prompt": "A castle.",
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Pencil sketch style." in data["compiled_prompt"]
        assert "A castle." in data["compiled_prompt"]

    def test_compile_manual_append(self, test_client):
        """Manual append mode should include free-text append in compiled prompt."""
        resp = test_client.post(
            "/api/prompt/compile",
            json={
                "model_id": "z-image-turbo",
                "prompt_mode": "manual",
                "manual_prompt": "A castle.",
                "append_mode": "manual",
                "manual_append": "Vibrant colours.",
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Vibrant colours." in data["compiled_prompt"]

    def test_compile_manual_prepend_and_append(self, test_client):
        """Both manual prepend and append should appear in compiled prompt."""
        resp = test_client.post(
            "/api/prompt/compile",
            json={
                "model_id": "z-image-turbo",
                "prepend_mode": "manual",
                "manual_prepend": "Woodcut engraving.",
                "prompt_mode": "manual",
                "manual_prompt": "A dragon.",
                "append_mode": "manual",
                "manual_append": "Dramatic shadows.",
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Woodcut engraving." in data["compiled_prompt"]
        assert "A dragon." in data["compiled_prompt"]
        assert "Dramatic shadows." in data["compiled_prompt"]


# ---------------------------------------------------------------------------
# Gallery endpoint tests.
# ---------------------------------------------------------------------------


class TestGallery:
    """Test gallery listing, filtering, and pagination."""

    def test_empty_gallery(self, test_client):
        """An empty gallery should return zero images."""
        resp = test_client.get("/api/gallery")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["images"] == []

    def test_gallery_after_generation(self, test_client):
        """After generating an image, the gallery should contain it."""
        # Generate one image first.
        test_client.post(
            "/api/generate",
            json={
                "model_id": "z-image-turbo",
                "prepend_prompt_id": "none",
                "prompt_mode": "manual",
                "manual_prompt": "test",
                "aspect_ratio_id": "1:1",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 0.0,
                "batch_size": 1,
            },
        )

        resp = test_client.get("/api/gallery")
        data = resp.json()
        assert data["total"] == 1

    def test_gallery_pagination(self, test_client, sample_gallery):
        """Pagination should return the correct slice of images."""
        resp = test_client.get("/api/gallery?page=1&per_page=2")
        data = resp.json()
        assert data["total"] == 5
        assert len(data["images"]) == 2
        assert data["page"] == 1
        assert data["pages"] == 3  # ceil(5 / 2) = 3

    def test_gallery_favourites_filter(self, test_client, sample_gallery):
        """Favourites filter should return only favourited images."""
        resp = test_client.get("/api/gallery?favourites_only=true")
        data = resp.json()
        # Only the first entry is favourited in sample_gallery.
        assert data["total"] == 1
        assert data["images"][0]["is_favourite"] is True

    def test_gallery_model_filter(self, test_client, sample_gallery):
        """Model filter should return only images from the specified model."""
        resp = test_client.get("/api/gallery?model_id=z-image-turbo")
        data = resp.json()
        assert data["total"] == 5  # All sample images use z-image-turbo.

    def test_gallery_model_filter_no_match(self, test_client, sample_gallery):
        """Model filter with no matching images should return empty list."""
        resp = test_client.get("/api/gallery?model_id=nonexistent")
        data = resp.json()
        assert data["total"] == 0


class TestGalleryEntry:
    """Test single gallery entry retrieval."""

    def test_get_existing_entry(self, test_client, sample_gallery):
        """GET /api/gallery/{id} should return the correct entry."""
        entry_id = sample_gallery[0]["id"]
        resp = test_client.get(f"/api/gallery/{entry_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == entry_id

    def test_get_nonexistent_entry(self, test_client, sample_gallery):
        """GET /api/gallery/{id} with an unknown ID should return 404."""
        resp = test_client.get("/api/gallery/nonexistent-id")
        assert resp.status_code == 404


class TestGalleryPrompt:
    """Test prompt metadata retrieval."""

    def test_get_prompt_metadata(self, test_client, sample_gallery):
        """GET /api/gallery/{id}/prompt should return prompt details."""
        entry_id = sample_gallery[0]["id"]
        resp = test_client.get(f"/api/gallery/{entry_id}/prompt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == entry_id
        assert "compiled_prompt" in data
        assert "seed" in data

    def test_get_prompt_nonexistent(self, test_client, sample_gallery):
        """GET /api/gallery/{id}/prompt with unknown ID should return 404."""
        resp = test_client.get("/api/gallery/nonexistent-id/prompt")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Favourite endpoint tests.
# ---------------------------------------------------------------------------


class TestFavourite:
    """Test POST /api/gallery/favourite — favourite toggling."""

    def test_set_favourite(self, test_client, sample_gallery):
        """Should mark an image as favourite."""
        entry_id = sample_gallery[1]["id"]  # Not favourited initially.
        resp = test_client.post(
            "/api/gallery/favourite",
            json={"image_id": entry_id, "is_favourite": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["is_favourite"] is True

        # Verify the change persisted in the gallery.
        check = test_client.get(f"/api/gallery/{entry_id}")
        assert check.json()["is_favourite"] is True

    def test_unset_favourite(self, test_client, sample_gallery):
        """Should unmark a favourited image."""
        entry_id = sample_gallery[0]["id"]  # Favourited initially.
        resp = test_client.post(
            "/api/gallery/favourite",
            json={"image_id": entry_id, "is_favourite": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_favourite"] is False

    def test_favourite_nonexistent(self, test_client, sample_gallery):
        """Favouriting a nonexistent image should return 404."""
        resp = test_client.post(
            "/api/gallery/favourite",
            json={"image_id": "nonexistent-id", "is_favourite": True},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete endpoint tests.
# ---------------------------------------------------------------------------


class TestDelete:
    """Test DELETE /api/gallery/{id} — image deletion."""

    def test_delete_existing(self, test_client, sample_gallery):
        """Deleting an existing image should remove it from the gallery."""
        entry_id = sample_gallery[0]["id"]
        resp = test_client.delete(f"/api/gallery/{entry_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["deleted"] == entry_id

        # Verify it's gone from the gallery.
        check = test_client.get(f"/api/gallery/{entry_id}")
        assert check.status_code == 404

    def test_delete_reduces_count(self, test_client, sample_gallery):
        """Gallery total should decrease after deletion."""
        entry_id = sample_gallery[0]["id"]
        test_client.delete(f"/api/gallery/{entry_id}")

        resp = test_client.get("/api/gallery")
        assert resp.json()["total"] == 4

    def test_delete_nonexistent(self, test_client, sample_gallery):
        """Deleting a nonexistent image should return 404."""
        resp = test_client.delete("/api/gallery/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Stats endpoint tests.
# ---------------------------------------------------------------------------


class TestBulkDelete:
    """Test POST /api/gallery/bulk-delete — bulk image deletion."""

    def test_bulk_delete_success(self, test_client, sample_gallery):
        """Deleting 2 of 5 images should leave 3 remaining."""
        ids_to_delete = [sample_gallery[0]["id"], sample_gallery[1]["id"]]

        resp = test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": ids_to_delete},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert set(data["deleted"]) == set(ids_to_delete)
        assert data["not_found"] == []

        # Verify gallery count decreased.
        gallery_resp = test_client.get("/api/gallery")
        assert gallery_resp.json()["total"] == 3

    def test_bulk_delete_partial_not_found(self, test_client, sample_gallery):
        """Some IDs missing should delete what exists and report not_found."""
        valid_id = sample_gallery[0]["id"]
        fake_id = "nonexistent-id-12345"

        resp = test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": [valid_id, fake_id]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert valid_id in data["deleted"]
        assert fake_id in data["not_found"]

        # Gallery should have 4 remaining.
        gallery_resp = test_client.get("/api/gallery")
        assert gallery_resp.json()["total"] == 4

    def test_bulk_delete_empty_list_rejected(self, test_client, sample_gallery):
        """An empty image_ids list should be rejected by Pydantic validation."""
        resp = test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": []},
        )
        assert resp.status_code == 422

    def test_bulk_delete_removes_files(self, test_client, sample_gallery, tmp_gallery_dir):
        """PNG files should be removed from disk after bulk delete."""
        entry = sample_gallery[0]
        filepath = tmp_gallery_dir / entry["filename"]
        assert filepath.exists()

        test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": [entry["id"]]},
        )

        assert not filepath.exists()

    def test_bulk_delete_all(self, test_client, sample_gallery):
        """Deleting all images should result in an empty gallery."""
        all_ids = [e["id"] for e in sample_gallery]

        resp = test_client.post(
            "/api/gallery/bulk-delete",
            json={"image_ids": all_ids},
        )
        assert resp.status_code == 200
        assert len(resp.json()["deleted"]) == 5

        gallery_resp = test_client.get("/api/gallery")
        assert gallery_resp.json()["total"] == 0


class TestStats:
    """Test GET /api/stats — gallery statistics."""

    def test_stats_empty_gallery(self, test_client):
        """Stats on an empty gallery should show all zeros."""
        resp = test_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 0
        assert data["total_favourites"] == 0
        assert data["model_counts"] == {}

    def test_stats_with_gallery(self, test_client, sample_gallery):
        """Stats should reflect the sample gallery contents."""
        resp = test_client.get("/api/stats")
        data = resp.json()
        assert data["total_images"] == 5
        assert data["total_favourites"] == 1  # Only first entry is favourited.
        assert data["model_counts"]["z-image-turbo"] == 5

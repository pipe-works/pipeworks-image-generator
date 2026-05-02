"""Integration tests for the LoRA Dataset tab API.

Cover the full per-run lifecycle end-to-end against a mocked mud-server
and a mocked ModelManager:

- create a run from snapshotted prompt-v3 sections plus a location set
- run generation, asserting per-tile PNG/TXT files land on disk
- regenerate a single tile and confirm the seed bump
- toggle the excluded flag via PATCH
- export the dataset and confirm only non-excluded pairs are copied
- list runs and read the run back through the public surface
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def _fake_login_fetch(*, base_url, method, path, body):
    """Stand-in for the anonymous mud-server login transport."""
    return {
        "session_id": "session-admin-1",
        "role": "admin",
        "available_worlds": [{"id": "pipeworks_web", "name": "Pipeworks Web"}],
    }


def _make_authenticated_fetch(items):
    """Build an authenticated mud-server fetch stub with a fixed policy listing."""

    def _fetch(*, runtime, method, path, query_params, json_payload=None):
        if path == "/api/policy-capabilities":
            return {
                "allowed_policy_types": ["prompt", "location"],
                "allowed_statuses": ["draft", "active"],
            }
        if path == "/api/policies":
            return {"items": items}
        raise AssertionError(f"Unexpected path: {path}")

    return _fetch


_LOCATIONS = [
    {
        "policy_id": "location:image.locations.environment:cozy_inn",
        "policy_type": "location",
        "namespace": "image.locations.environment",
        "policy_key": "cozy_inn",
        "variant": "v1",
        "content": {"text": "A warm timber-beamed inn lit by hearth firelight."},
    },
    {
        "policy_id": "location:image.locations.environment:foggy_moor",
        "policy_type": "location",
        "namespace": "image.locations.environment",
        "policy_key": "foggy_moor",
        "variant": "v1",
        "content": {
            "text": (
                "A wide stretch of damp open land covered in low growth, "
                "visibility reduced by drifting ground-level fog."
            )
        },
    },
]


def _login(test_client) -> None:
    resp = test_client.post(
        "/api/runtime-login",
        json={"username": "admin", "password": "pw"},
    )
    assert resp.status_code == 200


def _create_run_payload(*, location_ids: list[str]) -> dict:
    return {
        "model_id": "z-image-turbo",
        "aspect_ratio_id": "1:1",
        "width": 64,
        "height": 64,
        "steps": 4,
        "guidance": 0.0,
        "scheduler": None,
        "seed": 12345,
        "negative_prompt": None,
        "pinned_sections": [
            {
                "label": "Goblin",
                "mode": "automated",
                "manual_text": "A goblin. Small humanoid proportions, upright posture.",
                "automated_prompt_id": "species_block:image.blocks.species:goblin:v1",
            },
            {
                "label": "Tone",
                "mode": "automated",
                "manual_text": "Linocut style with muted sepia palette.",
                "automated_prompt_id": "tone_profile:image.tone_profiles:ledger_engraving:v1",
            },
        ],
        "location_section_label": "Location",
        "location_policy_ids": location_ids,
    }


class TestLoraDatasetRunLifecycle:
    """Full create → generate → curate → export flow."""

    def test_create_run_snapshots_locations_and_pinned_sections(self, test_client, test_config):
        """Run creation pins canonical text and sections to the manifest."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=[
                    "location:image.locations.environment:cozy_inn:v1",
                    "location:image.locations.environment:foggy_moor:v1",
                ]
            )
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert manifest["status"] == "pending"
            assert manifest["params"]["base_seed"] == 12345
            assert manifest["slot_order"] == ["cozy_inn", "foggy_moor"]
            cozy = manifest["slots"]["cozy_inn"]
            assert cozy["location_text"].startswith("A warm timber-beamed inn")
            assert cozy["location_label"]
            assert cozy["seed"] == 12345
            assert "Goblin:" in cozy["compiled_prompt"]
            assert "Linocut style" in cozy["compiled_prompt"]
            assert "Location:" in cozy["compiled_prompt"]
            # Manifest persisted on disk under outputs_dir/lora_runs/<run_id>/.
            run_dir = test_config.outputs_dir / "lora_runs" / manifest["run_id"]
            assert (run_dir / "manifest.json").exists()

    def test_create_run_rejects_unknown_location(self, test_client):
        """An unknown location id must produce a 400, not crash later."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=["location:image.locations.environment:bogus:v1"]
            )
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 400
            assert "bogus" in resp.json()["detail"]

    def test_create_run_rejects_non_location_policy(self, test_client):
        """Selecting a non-`location` policy id must be rejected at create time."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(
                    [
                        *_LOCATIONS,
                        {
                            "policy_id": "prompt:image.prompts:test",
                            "policy_type": "prompt",
                            "namespace": "image.prompts",
                            "policy_key": "test",
                            "variant": "v1",
                            "content": {"text": "Test prompt content."},
                        },
                    ]
                ),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(location_ids=["prompt:image.prompts:test:v1"])
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 400
            assert "not a canonical location" in resp.json()["detail"]

    def test_generate_run_writes_per_tile_files_and_marks_done(
        self, test_client, test_config, mock_model_manager
    ):
        """Run generation writes one PNG and one TXT per slot into the run dir."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
            patch(
                "pipeworks.api.main.get_model_runtime_support",
                return_value=(True, None),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=[
                    "location:image.locations.environment:cozy_inn:v1",
                    "location:image.locations.environment:foggy_moor:v1",
                ]
            )
            create = test_client.post("/api/lora-dataset/runs", json=payload)
            run_id = create.json()["run_id"]

            generate = test_client.post(f"/api/lora-dataset/runs/{run_id}/generate")
            assert generate.status_code == 200, generate.text
            body = generate.json()
            assert body["status"] == "complete"
            assert body["completed"] == 2
            assert body["cancelled"] is False

            # ModelManager was loaded once and called once per tile.
            mock_model_manager.load_model.assert_called_once()
            assert mock_model_manager.generate.call_count == 2

            run_dir: Path = test_config.outputs_dir / "lora_runs" / run_id
            assert (run_dir / "00_cozy_inn.png").exists()
            assert (run_dir / "00_cozy_inn.txt").exists()
            assert (run_dir / "01_foggy_moor.png").exists()
            assert (run_dir / "01_foggy_moor.txt").exists()
            assert (
                (run_dir / "00_cozy_inn.txt")
                .read_text(encoding="utf-8")
                .startswith("A warm timber-beamed inn")
            )

            manifest = test_client.get(f"/api/lora-dataset/runs/{run_id}").json()
            assert manifest["status"] == "complete"
            assert manifest["slots"]["cozy_inn"]["status"] == "done"
            assert manifest["slots"]["cozy_inn"]["image_filename"] == "00_cozy_inn.png"
            assert manifest["slots"]["foggy_moor"]["seed"] == 12346

    def test_regenerate_slot_bumps_seed_and_clears_excluded(
        self, test_client, test_config, mock_model_manager
    ):
        """Per-slot regeneration must vary the seed and re-include excluded slots."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
            patch(
                "pipeworks.api.main.get_model_runtime_support",
                return_value=(True, None),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=[
                    "location:image.locations.environment:cozy_inn:v1",
                    "location:image.locations.environment:foggy_moor:v1",
                ]
            )
            run_id = test_client.post("/api/lora-dataset/runs", json=payload).json()["run_id"]
            test_client.post(f"/api/lora-dataset/runs/{run_id}/generate")
            first_seed = test_client.get(f"/api/lora-dataset/runs/{run_id}").json()["slots"][
                "cozy_inn"
            ]["seed"]

            # Mark the slot excluded, then regenerate — regen should clear excluded.
            patch_resp = test_client.patch(
                f"/api/lora-dataset/runs/{run_id}/slots/cozy_inn",
                json={"excluded": True},
            )
            assert patch_resp.status_code == 200
            assert patch_resp.json()["excluded"] is True

            mock_model_manager.generate.reset_mock()
            regen = test_client.post(f"/api/lora-dataset/runs/{run_id}/slots/cozy_inn/regenerate")
            assert regen.status_code == 200, regen.text

            after = test_client.get(f"/api/lora-dataset/runs/{run_id}").json()
            assert after["slots"]["cozy_inn"]["excluded"] is False
            assert after["slots"]["cozy_inn"]["seed"] != first_seed
            assert mock_model_manager.generate.call_count == 1

    def test_export_dataset_skips_excluded_slots(
        self, test_client, test_config, mock_model_manager
    ):
        """Dataset export must omit slots flagged excluded."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
            patch(
                "pipeworks.api.main.get_model_runtime_support",
                return_value=(True, None),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=[
                    "location:image.locations.environment:cozy_inn:v1",
                    "location:image.locations.environment:foggy_moor:v1",
                ]
            )
            run_id = test_client.post("/api/lora-dataset/runs", json=payload).json()["run_id"]
            test_client.post(f"/api/lora-dataset/runs/{run_id}/generate")

            test_client.patch(
                f"/api/lora-dataset/runs/{run_id}/slots/foggy_moor",
                json={"excluded": True},
            )

            export = test_client.post(f"/api/lora-dataset/runs/{run_id}/dataset")
            assert export.status_code == 200, export.text
            result = export.json()
            assert result["pairs_copied"] == 1
            assert result["excluded"] == 1
            assert result["skipped"] == 0

            dataset_dir = test_config.outputs_dir / "lora_runs" / run_id / "dataset"
            assert (dataset_dir / "00_cozy_inn.png").exists()
            assert (dataset_dir / "00_cozy_inn.txt").exists()
            # The excluded foggy_moor pair is intentionally absent.
            assert not (dataset_dir / "01_foggy_moor.png").exists()
            assert not (dataset_dir / "01_foggy_moor.txt").exists()

    def test_list_runs_returns_persisted_manifests(self, test_client):
        """GET /api/lora-dataset/runs must return all runs, newest first."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=["location:image.locations.environment:cozy_inn:v1"]
            )
            run_id = test_client.post("/api/lora-dataset/runs", json=payload).json()["run_id"]
            list_resp = test_client.get("/api/lora-dataset/runs")
            assert list_resp.status_code == 200
            payload = list_resp.json()
            assert any(run["run_id"] == run_id for run in payload["runs"])

    def test_run_file_route_serves_tile_pngs(self, test_client, test_config, mock_model_manager):
        """The run file route serves tile PNGs and rejects path traversal."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
            patch(
                "pipeworks.api.main.get_model_runtime_support",
                return_value=(True, None),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=["location:image.locations.environment:cozy_inn:v1"]
            )
            run_id = test_client.post("/api/lora-dataset/runs", json=payload).json()["run_id"]
            test_client.post(f"/api/lora-dataset/runs/{run_id}/generate")

            png_resp = test_client.get(f"/api/lora-dataset/runs/{run_id}/files/00_cozy_inn.png")
            assert png_resp.status_code == 200
            assert png_resp.headers["content-type"].startswith("image/")

            traversal = test_client.get(f"/api/lora-dataset/runs/{run_id}/files/..%2Fmanifest.json")
            # FastAPI rejects the literal "%2F" in the path before routing,
            # so any successful resolution must point inside the run dir.
            assert traversal.status_code in {400, 404}

    def test_cancel_sets_cancel_requested_flag(self, test_client):
        """POST /cancel must flag the manifest even when no run is in flight."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=["location:image.locations.environment:cozy_inn:v1"]
            )
            run_id = test_client.post("/api/lora-dataset/runs", json=payload).json()["run_id"]
            cancel = test_client.post(f"/api/lora-dataset/runs/{run_id}/cancel")
            assert cancel.status_code == 200
            assert cancel.json()["status"] == "cancelling"
            manifest = test_client.get(f"/api/lora-dataset/runs/{run_id}").json()
            assert manifest["cancel_requested"] is True


class TestLoraPlaceholderFreezing:
    """Placeholder draws must be frozen at run creation, not per-tile."""

    def test_placeholders_resolved_once_and_shared_across_all_slots(self, test_client):
        """Pinned-section placeholders are drawn once and reused by every tile.

        For LoRA datasets the consistency stack must be identical across
        every tile — re-rolling ``{a|b|c}`` per tile would teach the model
        "this character has all variants" rather than locking identity.
        ``create_run`` is the single point at which the draw happens; the
        per-tile loop reads the resolved text directly.
        """
        location_ids = [
            "location:image.locations.environment:cozy_inn:v1",
            "location:image.locations.environment:foggy_moor:v1",
        ]
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(location_ids=location_ids)
            payload["pinned_sections"] = [
                {
                    "label": "Goblin",
                    "mode": "manual",
                    "manual_text": (
                        "A goblin with {yellow|green|red|blue|crimson|silver|black} eyes."
                    ),
                    "automated_prompt_id": None,
                },
            ]
            payload["negative_prompt"] = "{blurry|low quality} background"
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()

            # The pinned section's placeholder is resolved on the manifest itself.
            resolved_text = manifest["pinned_sections"][0]["manual_text"]
            assert "{" not in resolved_text and "|" not in resolved_text
            assert any(
                eye in resolved_text
                for eye in ("yellow", "green", "red", "blue", "crimson", "silver", "black")
            )

            # Every slot's compiled prompt embeds that same resolved text —
            # not a fresh per-tile draw.
            slot_texts = {key: slot["compiled_prompt"] for key, slot in manifest["slots"].items()}
            for compiled in slot_texts.values():
                assert resolved_text in compiled

            # The negative prompt is also resolved exactly once.
            resolved_negative = manifest["params"]["negative_prompt"]
            assert "{" not in resolved_negative and "|" not in resolved_negative

    def test_placeholders_resolved_when_no_brace_syntax_present(self, test_client):
        """Sections without placeholders pass through unchanged."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=["location:image.locations.environment:cozy_inn:v1"]
            )
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert (
                manifest["pinned_sections"][0]["manual_text"]
                == payload["pinned_sections"][0]["manual_text"]
            )


class TestLoraRunStoreReconciliation:
    """Reconciliation prunes references to disappeared tile files on read."""

    def test_missing_png_resets_slot_to_pending(self, test_client, test_config, mock_model_manager):
        """Deleting a tile PNG mid-curation should reset that slot to pending."""
        with (
            patch(
                "pipeworks.api.main._fetch_mud_api_json_anonymous",
                side_effect=_fake_login_fetch,
            ),
            patch(
                "pipeworks.api.main._fetch_mud_api_json",
                side_effect=_make_authenticated_fetch(_LOCATIONS),
            ),
            patch(
                "pipeworks.api.main.get_model_runtime_support",
                return_value=(True, None),
            ),
        ):
            _login(test_client)
            payload = _create_run_payload(
                location_ids=[
                    "location:image.locations.environment:cozy_inn:v1",
                ]
            )
            run_id = test_client.post("/api/lora-dataset/runs", json=payload).json()["run_id"]
            test_client.post(f"/api/lora-dataset/runs/{run_id}/generate")

            run_dir = test_config.outputs_dir / "lora_runs" / run_id
            (run_dir / "00_cozy_inn.png").unlink()

            manifest = test_client.get(f"/api/lora-dataset/runs/{run_id}").json()
            slot = manifest["slots"]["cozy_inn"]
            assert slot["status"] == "pending"
            assert slot["image_filename"] is None

            # On-disk manifest reflects the reconciliation.
            on_disk = json.loads((run_dir / "manifest.json").read_text())
            assert on_disk["slots"]["cozy_inn"]["status"] == "pending"

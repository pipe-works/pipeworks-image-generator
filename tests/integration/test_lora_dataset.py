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
from importlib.resources import files
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

    combined_items = [
        *items,
        {
            "policy_id": "species_block:image.blocks.species:goblin",
            "policy_type": "species_block",
            "namespace": "image.blocks.species",
            "policy_key": "goblin",
            "variant": "v1",
            "content": {
                "text": (
                    "A goblin. Small humanoid proportions, upright posture.\n"
                    "Large pointed ears set high on the skull.\n"
                    "Wide-set forward-facing crimson eyes."
                )
            },
        },
        {
            "policy_id": "species_block:image.blocks.species:human",
            "policy_type": "species_block",
            "namespace": "image.blocks.species",
            "policy_key": "human",
            "variant": "v1",
            "content": {
                "text": (
                    "A human of pipe-works canon with grounded proportions,\n"
                    "practical bearing, and no exaggerated fantasy morphology."
                )
            },
        },
        {
            "policy_id": "tone_profile:image.tone_profiles:ledger_engraving",
            "policy_type": "tone_profile",
            "namespace": "image.tone_profiles",
            "policy_key": "ledger_engraving",
            "variant": "v1",
            "content": {
                "prompt_block": "Linocut style with muted sepia palette and archival paper texture."
            },
        },
    ]

    def _fetch(*, runtime, method, path, query_params, json_payload=None):
        if path == "/api/policy-capabilities":
            return {
                "allowed_policy_types": ["prompt", "location"],
                "allowed_statuses": ["draft", "active"],
            }
        if path == "/api/policies":
            return {"items": combined_items}
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
        "location_policy_ids": location_ids,
        "character_sheet_keys": [],
        "character_view_anatomy_profile": "auto",
        "facial_expression_keys": [],
        "body_action_keys": [],
    }


def _seed_tile_pack(filename: str) -> None:
    import shutil

    from pipeworks.api.main import DATA_DIR

    bundled = files("pipeworks") / "static" / "data" / filename
    target = DATA_DIR / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(bundled), target)


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
            assert cozy["tile_kind"] == "location"
            assert cozy["tile_text"].startswith("A warm timber-beamed inn")
            assert cozy["tile_label"]
            assert cozy["section_label"] == "Location"
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

    def test_create_run_resolves_automated_sections_from_prompt_lookup(self, test_client):
        """LoRA snapshots must ignore stale textarea text for automated sections."""
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
            payload["pinned_sections"][0]["manual_text"] = (
                "A human of pipe-works canon with grounded proportions.\n"
                "A goblin. Polluted stale textarea content."
            )
            payload["pinned_sections"][1]["manual_text"] = "Stale tone textarea content."
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert manifest["pinned_sections"][0]["manual_text"].startswith("A goblin.")
            assert (
                "polluted stale textarea content"
                not in manifest["pinned_sections"][0]["manual_text"].lower()
            )
            assert (
                manifest["pinned_sections"][1]["manual_text"]
                == "Linocut style with muted sepia palette and archival paper texture."
            )

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
            # Default seed strategy is shared: every slot uses base_seed.
            assert manifest["slots"]["foggy_moor"]["seed"] == 12345

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

    def test_export_dataset_renumbers_curated_subset_densely(
        self, test_client, test_config, mock_model_manager
    ):
        """Export numbering should not leave holes after excluded earlier slots."""
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
                f"/api/lora-dataset/runs/{run_id}/slots/cozy_inn",
                json={"excluded": True},
            )

            export = test_client.post(f"/api/lora-dataset/runs/{run_id}/dataset")
            assert export.status_code == 200, export.text

            dataset_dir = test_config.outputs_dir / "lora_runs" / run_id / "dataset"
            assert (dataset_dir / "00_foggy_moor.png").exists()
            assert (dataset_dir / "00_foggy_moor.txt").exists()
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


class TestLoraTilePacks:
    """Bundled tile-packs feed the non-location LoRA tab categories."""

    def test_tile_packs_endpoint_returns_directional_character_views(self, test_client):
        """The bundled character-sheet pack ships with four directional view entries."""
        # The endpoint reads from the test config's data_dir, which is a
        # tmp dir per test. Seed the bundled fixture there so the route
        # is exercised end-to-end rather than mocked out.
        _seed_tile_pack("lora_character_sheet.json")
        _seed_tile_pack("lora_character_view_profiles.json")
        _seed_tile_pack("lora_facial_expressions.json")
        _seed_tile_pack("lora_body_actions.json")

        resp = test_client.get("/api/lora-dataset/tile-packs")
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert len(payload["character_view_profiles"]) == 6
        assert len(payload["facial_expression"]) == 5
        assert len(payload["body_action"]) == 5
        human_profile = next(
            profile for profile in payload["character_view_profiles"] if profile["key"] == "human"
        )
        orc_profile = next(
            profile for profile in payload["character_view_profiles"] if profile["key"] == "orc"
        )
        assert human_profile["available"] is True
        assert "upright posture" in human_profile["prompt_suffix"].lower()
        assert orc_profile["available"] is False
        keys = [tile["key"] for tile in payload["character_sheet"]]
        assert keys == ["front_view", "left_profile", "back_view", "right_profile"]
        front_view = next(
            tile for tile in payload["character_sheet"] if tile["key"] == "front_view"
        )
        assert front_view["section_label"] == "Character View"
        assert front_view["aspect_ratio_hint"] == "3:4"
        assert "front view reference" in front_view["text"].lower()
        assert "slightly hunched posture" not in front_view["text"].lower()
        neutral = next(
            tile for tile in payload["facial_expression"] if tile["key"] == "neutral_closeup"
        )
        assert neutral["section_label"] == "Facial Expression"
        assert neutral["aspect_ratio_hint"] == "1:1"
        assert "neutral resting expression" in neutral["text"].lower()
        walking = next(tile for tile in payload["body_action"] if tile["key"] == "walking_stride")
        assert walking["section_label"] == "Body Action"
        assert walking["aspect_ratio_hint"] == "3:4"
        assert "walking stride" in walking["text"].lower()

    def test_create_run_with_character_sheet_tile_only(
        self, test_client, test_config, mock_model_manager
    ):
        """A run can be created from a character-sheet tile alone (no locations)."""
        _seed_tile_pack("lora_character_sheet.json")
        _seed_tile_pack("lora_character_view_profiles.json")

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
            payload = _create_run_payload(location_ids=[])
            payload["character_sheet_keys"] = [
                "front_view",
                "left_profile",
                "back_view",
                "right_profile",
            ]
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert manifest["slot_order"] == [
                "front_view",
                "left_profile",
                "back_view",
                "right_profile",
            ]
            slot = manifest["slots"]["front_view"]
            assert slot["tile_kind"] == "character_sheet"
            assert slot["section_label"] == "Character View"
            assert "front view reference" in slot["tile_text"].lower()
            assert "slightly hunched posture" in slot["tile_text"].lower()
            # The compiled prompt must use the tile's section_label as the
            # header, not the legacy "Location:" header.
            assert "Character View:" in slot["compiled_prompt"]
            assert "Location:" not in slot["compiled_prompt"]

            generate = test_client.post(f"/api/lora-dataset/runs/{manifest['run_id']}/generate")
            assert generate.status_code == 200, generate.text
            assert generate.json()["status"] == "complete"

            run_dir = test_config.outputs_dir / "lora_runs" / manifest["run_id"]
            assert (run_dir / "00_front_view.png").exists()
            assert (run_dir / "00_front_view.txt").exists()
            assert (run_dir / "01_left_profile.png").exists()
            assert (run_dir / "01_left_profile.txt").exists()
            assert (run_dir / "02_back_view.png").exists()
            assert (run_dir / "02_back_view.txt").exists()
            assert (run_dir / "03_right_profile.png").exists()
            assert (run_dir / "03_right_profile.txt").exists()

    def test_create_run_mixed_sources_locations_plus_character_sheet(self, test_client):
        """A run can mix locations and bundled tile-pack tiles in one batch."""
        _seed_tile_pack("lora_character_sheet.json")
        _seed_tile_pack("lora_character_view_profiles.json")

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
            payload["character_sheet_keys"] = ["front_view", "back_view"]
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert manifest["slot_order"] == ["cozy_inn", "front_view", "back_view"]
            assert manifest["slots"]["cozy_inn"]["tile_kind"] == "location"
            assert manifest["slots"]["front_view"]["tile_kind"] == "character_sheet"
            assert manifest["slots"]["cozy_inn"]["section_label"] == "Location"
            assert manifest["slots"]["front_view"]["section_label"] == "Character View"

    def test_create_run_character_view_human_override_uses_human_anatomy(self, test_client):
        """Manual Section-2 profile override beats auto-inferred goblin anatomy."""
        _seed_tile_pack("lora_character_sheet.json")
        _seed_tile_pack("lora_character_view_profiles.json")

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
            payload = _create_run_payload(location_ids=[])
            payload["character_sheet_keys"] = ["front_view"]
            payload["character_view_anatomy_profile"] = "human"
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            slot = manifest["slots"]["front_view"]
            assert "upright posture" in slot["tile_text"].lower()
            assert "slightly hunched posture" not in slot["tile_text"].lower()

    def test_create_run_character_view_auto_detects_human_species(self, test_client):
        """Auto profile inference picks the human anatomy suffix from species block."""
        _seed_tile_pack("lora_character_sheet.json")
        _seed_tile_pack("lora_character_view_profiles.json")

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
            payload = _create_run_payload(location_ids=[])
            payload["character_sheet_keys"] = ["front_view"]
            payload["pinned_sections"][0]["label"] = "Human"
            payload["pinned_sections"][0][
                "manual_text"
            ] = "A human. Grounded human proportions and upright posture."
            payload["pinned_sections"][0][
                "automated_prompt_id"
            ] = "species_block:image.blocks.species:human:v1"
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            slot = manifest["slots"]["front_view"]
            assert "upright posture" in slot["tile_text"].lower()
            assert "slightly hunched posture" not in slot["tile_text"].lower()

    def test_create_run_rejects_unknown_character_sheet_key(self, test_client):
        """Unknown character-sheet keys are rejected with a 400."""
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
            payload = _create_run_payload(location_ids=[])
            payload["character_sheet_keys"] = ["nonexistent_pose"]
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 400
            assert "character-sheet" in resp.json()["detail"].lower()

    def test_create_run_rejects_unknown_character_view_profile(self, test_client):
        """Unknown Section-2 anatomy profiles are rejected with a 400."""
        _seed_tile_pack("lora_character_sheet.json")
        _seed_tile_pack("lora_character_view_profiles.json")

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
            payload = _create_run_payload(location_ids=[])
            payload["character_sheet_keys"] = ["front_view"]
            payload["character_view_anatomy_profile"] = "dwarf"
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 400
            assert "anatomy profile" in resp.json()["detail"].lower()

    def test_create_run_rejects_placeholder_character_view_profile(self, test_client):
        """Placeholder Section-2 anatomy profiles are visible but not usable yet."""
        _seed_tile_pack("lora_character_sheet.json")
        _seed_tile_pack("lora_character_view_profiles.json")

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
            payload = _create_run_payload(location_ids=[])
            payload["character_sheet_keys"] = ["front_view"]
            payload["character_view_anatomy_profile"] = "orc"
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 400
            assert "placeholder" in resp.json()["detail"].lower()

    def test_create_run_with_facial_expression_tiles_only(
        self, test_client, test_config, mock_model_manager
    ):
        """A run can be created from facial-expression tiles alone."""
        _seed_tile_pack("lora_facial_expressions.json")

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
            payload = _create_run_payload(location_ids=[])
            payload["facial_expression_keys"] = [
                "neutral_closeup",
                "smiling_closeup",
                "angry_closeup",
            ]
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert manifest["slot_order"] == [
                "neutral_closeup",
                "smiling_closeup",
                "angry_closeup",
            ]
            slot = manifest["slots"]["neutral_closeup"]
            assert slot["tile_kind"] == "facial_expression"
            assert slot["section_label"] == "Facial Expression"
            assert "neutral resting expression" in slot["tile_text"].lower()
            assert "Facial Expression:" in slot["compiled_prompt"]

            generate = test_client.post(f"/api/lora-dataset/runs/{manifest['run_id']}/generate")
            assert generate.status_code == 200, generate.text
            assert generate.json()["status"] == "complete"

            run_dir = test_config.outputs_dir / "lora_runs" / manifest["run_id"]
            assert (run_dir / "00_neutral_closeup.png").exists()
            assert (run_dir / "00_neutral_closeup.txt").exists()
            assert (run_dir / "01_smiling_closeup.png").exists()
            assert (run_dir / "01_smiling_closeup.txt").exists()
            assert (run_dir / "02_angry_closeup.png").exists()
            assert (run_dir / "02_angry_closeup.txt").exists()

    def test_create_run_mixed_locations_character_sheet_and_facial_expression(self, test_client):
        """A run can mix canonical locations with multiple bundled tile-pack kinds."""
        _seed_tile_pack("lora_character_sheet.json")
        _seed_tile_pack("lora_character_view_profiles.json")
        _seed_tile_pack("lora_facial_expressions.json")

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
            payload["character_sheet_keys"] = ["front_view"]
            payload["facial_expression_keys"] = ["smiling_closeup", "surprised_closeup"]
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert manifest["slot_order"] == [
                "cozy_inn",
                "front_view",
                "smiling_closeup",
                "surprised_closeup",
            ]
            assert manifest["slots"]["cozy_inn"]["tile_kind"] == "location"
            assert manifest["slots"]["front_view"]["tile_kind"] == "character_sheet"
            assert manifest["slots"]["smiling_closeup"]["tile_kind"] == "facial_expression"
            assert manifest["slots"]["smiling_closeup"]["section_label"] == "Facial Expression"

    def test_create_run_rejects_unknown_facial_expression_key(self, test_client):
        """Unknown facial-expression keys are rejected with a 400."""
        _seed_tile_pack("lora_facial_expressions.json")

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
            payload = _create_run_payload(location_ids=[])
            payload["facial_expression_keys"] = ["grimacing_closeup"]
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 400
            assert "facial-expression" in resp.json()["detail"].lower()

    def test_create_run_with_body_action_tiles_only(
        self, test_client, test_config, mock_model_manager
    ):
        """A run can be created from body-action tiles alone."""
        _seed_tile_pack("lora_body_actions.json")

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
            payload = _create_run_payload(location_ids=[])
            payload["body_action_keys"] = [
                "walking_stride",
                "running_stride",
                "crouched_balance",
            ]
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert manifest["slot_order"] == [
                "walking_stride",
                "running_stride",
                "crouched_balance",
            ]
            slot = manifest["slots"]["walking_stride"]
            assert slot["tile_kind"] == "body_action"
            assert slot["section_label"] == "Body Action"
            assert "walking stride" in slot["tile_text"].lower()
            assert "Body Action:" in slot["compiled_prompt"]

            generate = test_client.post(f"/api/lora-dataset/runs/{manifest['run_id']}/generate")
            assert generate.status_code == 200, generate.text
            assert generate.json()["status"] == "complete"

            run_dir = test_config.outputs_dir / "lora_runs" / manifest["run_id"]
            assert (run_dir / "00_walking_stride.png").exists()
            assert (run_dir / "00_walking_stride.txt").exists()
            assert (run_dir / "01_running_stride.png").exists()
            assert (run_dir / "01_running_stride.txt").exists()
            assert (run_dir / "02_crouched_balance.png").exists()
            assert (run_dir / "02_crouched_balance.txt").exists()

    def test_create_run_mixed_locations_views_expressions_and_actions(self, test_client):
        """A run can mix canonical locations with all bundled tile-pack kinds."""
        _seed_tile_pack("lora_character_sheet.json")
        _seed_tile_pack("lora_character_view_profiles.json")
        _seed_tile_pack("lora_facial_expressions.json")
        _seed_tile_pack("lora_body_actions.json")

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
            payload["character_sheet_keys"] = ["front_view"]
            payload["facial_expression_keys"] = ["smiling_closeup"]
            payload["body_action_keys"] = ["walking_stride", "leaning_stance"]
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert manifest["slot_order"] == [
                "cozy_inn",
                "front_view",
                "smiling_closeup",
                "walking_stride",
                "leaning_stance",
            ]
            assert manifest["slots"]["cozy_inn"]["tile_kind"] == "location"
            assert manifest["slots"]["front_view"]["tile_kind"] == "character_sheet"
            assert manifest["slots"]["smiling_closeup"]["tile_kind"] == "facial_expression"
            assert manifest["slots"]["walking_stride"]["tile_kind"] == "body_action"
            assert manifest["slots"]["walking_stride"]["section_label"] == "Body Action"

    def test_create_run_rejects_unknown_body_action_key(self, test_client):
        """Unknown body-action keys are rejected with a 400."""
        _seed_tile_pack("lora_body_actions.json")

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
            payload = _create_run_payload(location_ids=[])
            payload["body_action_keys"] = ["vaulting_pose"]
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 400
            assert "body-action" in resp.json()["detail"].lower()

    def test_create_run_rejects_no_tiles_at_all(self, test_client):
        """A run with neither locations nor character-sheet keys is rejected."""
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
            payload = _create_run_payload(location_ids=[])
            payload["character_sheet_keys"] = []
            payload["facial_expression_keys"] = []
            payload["body_action_keys"] = []
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 400
            assert "at least one tile" in resp.json()["detail"].lower()


class TestLoraSeedStrategy:
    """Seed strategy controls character lock vs pose variance."""

    def test_default_seed_strategy_is_shared_across_all_tiles(self, test_client):
        """Run creation defaults to identical seed across every slot.

        Shared seed is the strongest character-lock available without i2i:
        every tile starts from the same noise, so only the location section
        varies. This is the default for LoRA runs.
        """
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
            assert manifest["params"]["share_seed_across_tiles"] is True
            assert manifest["params"]["base_seed"] == 12345
            assert all(slot["seed"] == 12345 for slot in manifest["slots"].values())

    def test_per_tile_offset_seed_strategy_when_share_disabled(self, test_client):
        """Disabling shared seed at create time produces per-tile-offset seeds."""
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
            payload["share_seed_across_tiles"] = False
            resp = test_client.post("/api/lora-dataset/runs", json=payload)
            assert resp.status_code == 200, resp.text
            manifest = resp.json()
            assert manifest["params"]["share_seed_across_tiles"] is False
            assert manifest["slots"]["cozy_inn"]["seed"] == 12345
            assert manifest["slots"]["foggy_moor"]["seed"] == 12346

    def test_patch_run_toggles_strategy_and_recomputes_pending_seeds(self, test_client):
        """Toggling the run-level flag must update pending slot seeds."""
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
            run_id = test_client.post("/api/lora-dataset/runs", json=payload).json()["run_id"]
            # Default state: shared seed, both slots at 12345.
            manifest = test_client.get(f"/api/lora-dataset/runs/{run_id}").json()
            assert manifest["slots"]["cozy_inn"]["seed"] == 12345
            assert manifest["slots"]["foggy_moor"]["seed"] == 12345

            patch_resp = test_client.patch(
                f"/api/lora-dataset/runs/{run_id}",
                json={"share_seed_across_tiles": False},
            )
            assert patch_resp.status_code == 200, patch_resp.text
            patched = patch_resp.json()
            assert patched["params"]["share_seed_across_tiles"] is False
            # Pending slots get the per-tile-offset seeds.
            assert patched["slots"]["cozy_inn"]["seed"] == 12345
            assert patched["slots"]["foggy_moor"]["seed"] == 12346

            # Flip back — both slots return to base_seed.
            back = test_client.patch(
                f"/api/lora-dataset/runs/{run_id}",
                json={"share_seed_across_tiles": True},
            ).json()
            assert back["params"]["share_seed_across_tiles"] is True
            assert all(slot["seed"] == 12345 for slot in back["slots"].values())

    def test_patch_run_preserves_seed_on_done_slots(self, test_client, mock_model_manager):
        """Toggling after generation must not rewrite historical seeds."""
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
            done_seed = test_client.get(f"/api/lora-dataset/runs/{run_id}").json()["slots"][
                "cozy_inn"
            ]["seed"]
            assert done_seed == 12345

            # Toggle to per-tile-offset; cozy_inn is done so its seed must
            # stay at 12345 (historical fact). foggy_moor is also done, so
            # it stays at 12345 too — neither slot is recomputed.
            patched = test_client.patch(
                f"/api/lora-dataset/runs/{run_id}",
                json={"share_seed_across_tiles": False},
            ).json()
            assert patched["params"]["share_seed_across_tiles"] is False
            assert patched["slots"]["cozy_inn"]["seed"] == 12345
            assert patched["slots"]["foggy_moor"]["seed"] == 12345

    def test_patch_run_rejects_strategy_change_while_running(self, test_client):
        """The endpoint refuses to act on a run currently generating."""
        # Construct the run, then forge the running status directly via the
        # underlying store so we don't have to actually start a thread that
        # would race the test. The router consults `status == "running"`.
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
        from pipeworks.api.main import OUTPUTS_DIR
        from pipeworks.api.services.lora_run_store import update_manifest

        def _set_running(manifest):
            manifest.status = "running"

        update_manifest(OUTPUTS_DIR, run_id, _set_running)
        resp = test_client.patch(
            f"/api/lora-dataset/runs/{run_id}",
            json={"share_seed_across_tiles": False},
        )
        assert resp.status_code == 409
        assert "currently generating" in resp.json()["detail"]


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

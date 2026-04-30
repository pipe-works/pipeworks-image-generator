"""Tests for pipeworks.api.models — Pydantic request/response models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeworks.api.models import (
    BulkDeleteRequest,
    FavouriteRequest,
    GenerateRequest,
    GpuSettingsTestRequest,
    GpuSettingsUpdateRequest,
    PromptSection,
)


def _minimal_kwargs(**overrides) -> dict:
    payload = {
        "model_id": "z-image-turbo",
        "prompt_schema_version": 3,
        "sections": [
            {
                "label": "Subject",
                "mode": "manual",
                "manual_text": "A goblin workshop.",
            }
        ],
        "aspect_ratio_id": "1:1",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "guidance": 0.0,
    }
    payload.update(overrides)
    return payload


class TestGenerateRequest:
    """Test GenerateRequest Pydantic model under the dynamic-section schema."""

    def test_valid_minimal_request(self):
        """A request with all required fields should validate successfully."""
        req = GenerateRequest(**_minimal_kwargs())
        assert req.model_id == "z-image-turbo"
        assert req.batch_size == 1
        assert req.prompt_schema_version == 3
        assert req.sections[0].label == "Subject"
        assert req.sections[0].manual_text == "A goblin workshop."

    def test_missing_required_field_raises(self):
        """Omitting a required field should raise ValidationError."""
        kwargs = _minimal_kwargs()
        kwargs.pop("model_id")
        with pytest.raises(ValidationError):
            GenerateRequest(**kwargs)

    def test_default_batch_size(self):
        """Default batch_size should be 1."""
        req = GenerateRequest(**_minimal_kwargs())
        assert req.batch_size == 1

    def test_default_optional_fields(self):
        """Optional fields should default to None / empty list."""
        req = GenerateRequest(**_minimal_kwargs(sections=[]))
        assert req.seed is None
        assert req.negative_prompt is None
        assert req.generation_id is None
        assert req.gpu_worker_id is None
        assert req.scheduler is None
        assert req.sections == []

    def test_all_fields_populated(self):
        """All fields should be settable and retrievable."""
        req = GenerateRequest(
            **_minimal_kwargs(
                aspect_ratio_id="16:9",
                width=1280,
                height=720,
                steps=9,
                guidance=7.5,
                seed=42,
                batch_size=4,
                negative_prompt="blurry",
                scheduler="dpmpp-2m-karras",
                generation_id="gen-123",
                gpu_worker_id="remote-gpu-1",
            )
        )
        assert req.batch_size == 4
        assert req.negative_prompt == "blurry"
        assert req.scheduler == "dpmpp-2m-karras"
        assert req.generation_id == "gen-123"
        assert req.gpu_worker_id == "remote-gpu-1"

    def test_rejects_unsupported_schema_version(self):
        """Schema versions other than 3 should be rejected by Pydantic."""
        with pytest.raises(ValidationError):
            GenerateRequest(**_minimal_kwargs(prompt_schema_version=2))

    def test_serialisation_round_trip(self):
        """Model should serialise to dict and back without data loss."""
        req = GenerateRequest(**_minimal_kwargs(seed=42))
        data = req.model_dump()
        restored = GenerateRequest(**data)
        assert restored.model_id == req.model_id
        assert restored.seed == 42
        assert [s.label for s in restored.sections] == ["Subject"]


class TestPromptSection:
    """Test PromptSection Pydantic model."""

    def test_default_label_is_policy(self):
        section = PromptSection()
        assert section.label == "Policy"
        assert section.mode == "manual"
        assert section.manual_text is None
        assert section.automated_prompt_id is None

    def test_rejects_invalid_mode(self):
        with pytest.raises(ValidationError):
            PromptSection(mode="invalid")


class TestFavouriteRequest:
    """Test FavouriteRequest Pydantic model."""

    def test_valid_favourite_request(self):
        req = FavouriteRequest(image_id="abc-123", is_favourite=True)
        assert req.image_id == "abc-123"
        assert req.is_favourite is True

    def test_missing_image_id_raises(self):
        with pytest.raises(ValidationError):
            FavouriteRequest(is_favourite=True)

    def test_missing_is_favourite_raises(self):
        with pytest.raises(ValidationError):
            FavouriteRequest(image_id="abc-123")

    def test_unfavourite(self):
        req = FavouriteRequest(image_id="abc-123", is_favourite=False)
        assert req.is_favourite is False


class TestBulkDeleteRequest:
    """Test BulkDeleteRequest Pydantic model."""

    def test_valid_request(self):
        req = BulkDeleteRequest(image_ids=["id-1", "id-2"])
        assert req.image_ids == ["id-1", "id-2"]

    def test_single_id(self):
        req = BulkDeleteRequest(image_ids=["id-1"])
        assert len(req.image_ids) == 1

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError):
            BulkDeleteRequest(image_ids=[])

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            BulkDeleteRequest()


class TestGpuSettingsRequests:
    """Test GPU settings request models."""

    def test_gpu_settings_update_defaults(self):
        req = GpuSettingsUpdateRequest()
        assert req.use_remote_gpu is False
        assert req.remote_base_url is None
        assert req.bearer_token is None
        assert req.default_to_remote is False
        assert req.timeout_seconds == 240.0

    def test_gpu_settings_test_request_allows_missing_token(self):
        req = GpuSettingsTestRequest(remote_base_url="https://gpu-worker.example")
        assert req.remote_base_url == "https://gpu-worker.example"
        assert req.bearer_token is None

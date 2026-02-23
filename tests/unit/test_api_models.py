"""Tests for pipeworks.api.models — Pydantic request/response models.

Tests cover:
- Required field validation on GenerateRequest.
- Default values for optional fields.
- FavouriteRequest field validation.
- Serialisation round-trips.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeworks.api.models import BulkDeleteRequest, FavouriteRequest, GenerateRequest


class TestGenerateRequest:
    """Test GenerateRequest Pydantic model."""

    def test_valid_minimal_request(self):
        """A request with all required fields should validate successfully."""
        req = GenerateRequest(
            model_id="z-image-turbo",
            prepend_prompt_id="none",
            prompt_mode="manual",
            manual_prompt="A goblin workshop.",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
        )
        assert req.model_id == "z-image-turbo"
        assert req.batch_size == 1  # Default value.

    def test_missing_required_field_raises(self):
        """Omitting a required field should raise ValidationError."""
        with pytest.raises(ValidationError):
            GenerateRequest(
                # Missing model_id.
                prepend_prompt_id="none",
                prompt_mode="manual",
                aspect_ratio_id="1:1",
                width=1024,
                height=1024,
                steps=4,
                guidance=0.0,
            )

    def test_default_batch_size(self):
        """Default batch_size should be 1."""
        req = GenerateRequest(
            model_id="test",
            prepend_prompt_id="none",
            prompt_mode="manual",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
        )
        assert req.batch_size == 1

    def test_default_optional_fields_are_none(self):
        """Optional fields should default to None."""
        req = GenerateRequest(
            model_id="test",
            prepend_prompt_id="none",
            prompt_mode="manual",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
        )
        assert req.manual_prompt is None
        assert req.automated_prompt_id is None
        assert req.seed is None
        assert req.negative_prompt is None

    def test_all_fields_populated(self):
        """All fields should be settable and retrievable."""
        req = GenerateRequest(
            model_id="z-image-turbo",
            prepend_prompt_id="oil-painting",
            prompt_mode="automated",
            manual_prompt=None,
            automated_prompt_id="goblin-workshop",
            append_prompt_id="high-detail",
            aspect_ratio_id="16:9",
            width=1280,
            height=720,
            steps=9,
            guidance=7.5,
            seed=42,
            batch_size=4,
            negative_prompt="blurry",
        )
        assert req.append_prompt_id == "high-detail"
        assert req.batch_size == 4
        assert req.negative_prompt == "blurry"

    def test_default_prepend_mode_is_template(self):
        """Default prepend_mode should be 'template'."""
        req = GenerateRequest(
            model_id="test",
            prompt_mode="manual",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
        )
        assert req.prepend_mode == "template"
        assert req.manual_prepend is None

    def test_default_append_mode_is_template(self):
        """Default append_mode should be 'template'."""
        req = GenerateRequest(
            model_id="test",
            prompt_mode="manual",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
        )
        assert req.append_mode == "template"
        assert req.manual_append is None

    def test_manual_prepend_mode(self):
        """Manual prepend mode should accept free-text input."""
        req = GenerateRequest(
            model_id="test",
            prepend_mode="manual",
            manual_prepend="Custom oil painting style.",
            prompt_mode="manual",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
        )
        assert req.prepend_mode == "manual"
        assert req.manual_prepend == "Custom oil painting style."

    def test_manual_append_mode(self):
        """Manual append mode should accept free-text input."""
        req = GenerateRequest(
            model_id="test",
            prompt_mode="manual",
            append_mode="manual",
            manual_append="Cinematic lighting.",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
        )
        assert req.append_mode == "manual"
        assert req.manual_append == "Cinematic lighting."

    def test_default_scheduler_is_none(self):
        """Default scheduler should be None."""
        req = GenerateRequest(
            model_id="test",
            prompt_mode="manual",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
        )
        assert req.scheduler is None

    def test_scheduler_can_be_set(self):
        """Scheduler field should accept a string value."""
        req = GenerateRequest(
            model_id="test",
            prompt_mode="manual",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
            scheduler="dpmpp-2m-karras",
        )
        assert req.scheduler == "dpmpp-2m-karras"

    def test_serialisation_round_trip(self):
        """Model should serialise to dict and back without data loss."""
        req = GenerateRequest(
            model_id="test",
            prepend_prompt_id="none",
            prompt_mode="manual",
            manual_prompt="test prompt",
            aspect_ratio_id="1:1",
            width=1024,
            height=1024,
            steps=4,
            guidance=0.0,
            seed=42,
        )
        data = req.model_dump()
        restored = GenerateRequest(**data)
        assert restored.model_id == req.model_id
        assert restored.seed == req.seed


class TestFavouriteRequest:
    """Test FavouriteRequest Pydantic model."""

    def test_valid_favourite_request(self):
        """A valid request should have image_id and is_favourite."""
        req = FavouriteRequest(
            image_id="abc-123",
            is_favourite=True,
        )
        assert req.image_id == "abc-123"
        assert req.is_favourite is True

    def test_missing_image_id_raises(self):
        """Omitting image_id should raise ValidationError."""
        with pytest.raises(ValidationError):
            FavouriteRequest(is_favourite=True)

    def test_missing_is_favourite_raises(self):
        """Omitting is_favourite should raise ValidationError."""
        with pytest.raises(ValidationError):
            FavouriteRequest(image_id="abc-123")

    def test_unfavourite(self):
        """is_favourite=False should be accepted."""
        req = FavouriteRequest(
            image_id="abc-123",
            is_favourite=False,
        )
        assert req.is_favourite is False


class TestBulkDeleteRequest:
    """Test BulkDeleteRequest Pydantic model."""

    def test_valid_request(self):
        """A request with one or more IDs should validate."""
        req = BulkDeleteRequest(image_ids=["id-1", "id-2"])
        assert req.image_ids == ["id-1", "id-2"]

    def test_single_id(self):
        """A request with exactly one ID should validate."""
        req = BulkDeleteRequest(image_ids=["id-1"])
        assert len(req.image_ids) == 1

    def test_empty_list_raises(self):
        """An empty image_ids list should raise ValidationError."""
        with pytest.raises(ValidationError):
            BulkDeleteRequest(image_ids=[])

    def test_missing_field_raises(self):
        """Omitting image_ids should raise ValidationError."""
        with pytest.raises(ValidationError):
            BulkDeleteRequest()

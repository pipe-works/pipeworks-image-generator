"""Pydantic request and response models for the Image Generator API.

These models define the JSON schema for every API endpoint.  FastAPI uses
them for automatic request validation, serialisation, and OpenAPI
documentation generation.

Models
------
GenerateRequest
    Payload for ``POST /api/generate`` — contains all parameters needed to
    run a batch image generation.
FavouriteRequest
    Payload for ``POST /api/gallery/favourite`` — toggles the favourite
    status of a single gallery image.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for the ``POST /api/generate`` endpoint.

    Contains every parameter needed to generate a batch of images:
    model selection, prompt composition parts, image dimensions, inference
    settings, seed, and batch size.

    Attributes:
        model_id: Identifier of the model to use (must match an ``id`` value
            in ``models.json``).
        prepend_mode: Either ``"template"`` (preset from ``prompts.json``) or
            ``"manual"`` (user-supplied text).  Defaults to ``"template"``.
        prepend_prompt_id: Identifier of the prepend style prompt to apply.
            Use ``"none"`` to skip.  Used when ``prepend_mode`` is
            ``"template"``.
        manual_prepend: Free-text prepend style.  Used when ``prepend_mode``
            is ``"manual"``.
        prompt_mode: Either ``"manual"`` (user-supplied prompt) or
            ``"automated"`` (preset scene selection).
        manual_prompt: Free-text prompt.  Required when ``prompt_mode`` is
            ``"manual"``.
        automated_prompt_id: Identifier of the automated scene preset.
            Required when ``prompt_mode`` is ``"automated"``.
        append_mode: Either ``"template"`` (preset from ``prompts.json``) or
            ``"manual"`` (user-supplied text).  Defaults to ``"template"``.
        append_prompt_id: Identifier of the append modifier.  Use
            ``"none"`` to skip.  Used when ``append_mode`` is ``"template"``.
        manual_append: Free-text append modifier.  Used when ``append_mode``
            is ``"manual"``.
        aspect_ratio_id: Identifier of the selected aspect ratio preset.
        width: Image width in pixels (should be a multiple of 64).
        height: Image height in pixels (should be a multiple of 64).
        steps: Number of diffusion inference steps.
        guidance: Classifier-free guidance scale.
        seed: Random seed.  ``None`` means the server picks a random seed.
        batch_size: Number of images to generate in a single request
            (1–16 inclusive).
        negative_prompt: Optional text describing what to avoid.
    """

    model_id: str = Field(
        ...,
        description="Model identifier from models.json (e.g. 'z-image-turbo').",
    )
    prepend_mode: str = Field(
        default="template",
        description="Prepend mode: 'template' (use preset) or 'manual' (free text).",
    )
    prepend_prompt_id: str = Field(
        default="none",
        description="Prepend style prompt ID, or 'none' to skip (template mode).",
    )
    manual_prepend: str | None = Field(
        default=None,
        description="Free-text prepend style (used when prepend_mode='manual').",
    )
    prompt_mode: str = Field(
        ...,
        description="Prompt mode: 'manual' or 'automated'.",
    )
    manual_prompt: str | None = Field(
        default=None,
        description="Free-text prompt (required when prompt_mode='manual').",
    )
    automated_prompt_id: str | None = Field(
        default=None,
        description="Automated scene preset ID (required when prompt_mode='automated').",
    )
    append_mode: str = Field(
        default="template",
        description="Append mode: 'template' (use preset) or 'manual' (free text).",
    )
    append_prompt_id: str | None = Field(
        default=None,
        description="Append modifier prompt ID, or 'none' to skip (template mode).",
    )
    manual_append: str | None = Field(
        default=None,
        description="Free-text append modifier (used when append_mode='manual').",
    )
    aspect_ratio_id: str = Field(
        ...,
        description="Aspect ratio preset identifier (e.g. '1:1', '16:9').",
    )
    width: int = Field(
        ...,
        description="Image width in pixels.",
    )
    height: int = Field(
        ...,
        description="Image height in pixels.",
    )
    steps: int = Field(
        ...,
        description="Number of diffusion inference steps.",
    )
    guidance: float = Field(
        ...,
        description="Classifier-free guidance scale.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed.  None = server picks a random seed.",
    )
    batch_size: int = Field(
        default=1,
        description="Number of images to generate (1–16).",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="Optional negative prompt (not supported by all models).",
    )


class FavouriteRequest(BaseModel):
    """Request body for the ``POST /api/gallery/favourite`` endpoint.

    Attributes:
        image_id: UUID of the gallery image to update.
        is_favourite: ``True`` to mark as favourite, ``False`` to unmark.
    """

    image_id: str = Field(
        ...,
        description="UUID of the gallery image to update.",
    )
    is_favourite: bool = Field(
        ...,
        description="True to mark as favourite, False to unmark.",
    )

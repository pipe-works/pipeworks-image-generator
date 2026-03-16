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
BulkDeleteRequest
    Payload for ``POST /api/gallery/bulk-delete`` — deletes multiple gallery
    images in a single request.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for the ``POST /api/generate`` endpoint.

    Contains every parameter needed to generate a batch of images:
    model selection, prompt composition parts, image dimensions, inference
    settings, seed, and batch size.

    Attributes:
        model_id: Identifier of the model to use (must match an ``id`` value
            in ``models.json``).
        prepend_mode: Either ``"template"`` (preset from the prompt library) or
            ``"manual"`` (user-supplied text).  Defaults to ``"template"``.
        prepend_prompt_id: Identifier of the prepend style prompt to apply.
            Use ``"none"`` to skip.  Used when ``prepend_mode`` is
            ``"template"``.
        manual_prepend: Free-text prepend style.  Used when ``prepend_mode``
            is ``"manual"``.
        prompt_mode: Either ``"manual"`` (user-supplied prompt) or
            ``"automated"`` (preset scene selection).
        manual_prompt: Free-text prompt. Optional when ``prompt_mode`` is
            ``"manual"``.
        automated_prompt_id: Identifier of the automated scene preset.
            Optional when ``prompt_mode`` is ``"automated"``.
        append_mode: Either ``"template"`` (preset from the prompt library) or
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
            (1–1000 inclusive).
        negative_prompt: Optional text describing what to avoid.
        scheduler: Optional scheduler/sampler identifier.  Must match an
            ``id`` in the model's ``schedulers`` list from ``models.json``.
            ``None`` means use the model's default scheduler.
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
        default="manual",
        description="Prompt mode: 'manual' or 'automated'.",
    )
    manual_prompt: str | None = Field(
        default=None,
        description="Free-text prompt (optional when prompt_mode='manual').",
    )
    automated_prompt_id: str | None = Field(
        default=None,
        description="Automated scene preset ID (optional when prompt_mode='automated').",
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
    prompt_schema_version: int | None = Field(
        default=None,
        description=(
            "Optional prompt schema version " "(2 = Subject/Setting/Details/Lighting/Atmosphere)."
        ),
    )
    subject_mode: str | None = Field(
        default=None,
        description="Subject mode: 'manual' or 'automated' (schema v2).",
    )
    manual_subject: str | None = Field(
        default=None,
        description="Free-text Subject section value (schema v2).",
    )
    automated_subject_prompt_id: str | None = Field(
        default=None,
        description="Automated Subject prompt snippet ID (schema v2).",
    )
    setting_mode: str | None = Field(
        default=None,
        description="Setting mode: 'manual' or 'automated' (schema v2).",
    )
    manual_setting: str | None = Field(
        default=None,
        description="Free-text Setting section value (schema v2).",
    )
    automated_setting_prompt_id: str | None = Field(
        default=None,
        description="Automated Setting prompt snippet ID (schema v2).",
    )
    details_mode: str | None = Field(
        default=None,
        description="Details mode: 'manual' or 'automated' (schema v2).",
    )
    manual_details: str | None = Field(
        default=None,
        description="Free-text Details section value (schema v2).",
    )
    automated_details_prompt_id: str | None = Field(
        default=None,
        description="Automated Details prompt snippet ID (schema v2).",
    )
    lighting_mode: str | None = Field(
        default=None,
        description="Lighting mode: 'manual' or 'automated' (schema v2).",
    )
    manual_lighting: str | None = Field(
        default=None,
        description="Free-text Lighting section value (schema v2).",
    )
    automated_lighting_prompt_id: str | None = Field(
        default=None,
        description="Automated Lighting prompt snippet ID (schema v2).",
    )
    atmosphere_mode: str | None = Field(
        default=None,
        description="Atmosphere mode: 'manual' or 'automated' (schema v2).",
    )
    manual_atmosphere: str | None = Field(
        default=None,
        description="Free-text Atmosphere section value (schema v2).",
    )
    automated_atmosphere_prompt_id: str | None = Field(
        default=None,
        description="Automated Atmosphere prompt snippet ID (schema v2).",
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
        description="Number of images to generate (1–1000).",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="Optional negative prompt (not supported by all models).",
    )
    scheduler: str | None = Field(
        default=None,
        description=(
            "Scheduler/sampler identifier (e.g. 'pndm', 'dpmpp-2m-karras'). "
            "None = use model's default scheduler."
        ),
    )
    generation_id: str | None = Field(
        default=None,
        description="Optional client-supplied batch identifier used for cancellation.",
    )
    gpu_worker_id: str | None = Field(
        default=None,
        description="Optional GPU worker identifier from /api/config.",
    )


class CancelGenerationRequest(BaseModel):
    """Request body for cancelling an in-flight generation batch."""

    generation_id: str = Field(
        ...,
        description="Client-supplied generation batch identifier to cancel.",
    )


class WorkerGenerateJob(BaseModel):
    """One compiled generation unit routed to a worker host."""

    index: int = Field(ge=0)
    seed: int = Field(ge=0)
    prompt: str
    negative_prompt: str | None = None


class WorkerGenerateBatchRequest(BaseModel):
    """Request body for ``POST /api/worker/generate-batch``."""

    generation_id: str
    hf_id: str
    width: int = Field(ge=64)
    height: int = Field(ge=64)
    steps: int = Field(ge=1)
    guidance: float
    scheduler: str | None = None
    jobs: list[WorkerGenerateJob] = Field(min_length=1, max_length=1000)


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


class BulkDeleteRequest(BaseModel):
    """Request body for the ``POST /api/gallery/bulk-delete`` endpoint.

    Attributes:
        image_ids: List of gallery image UUIDs to delete.  Must contain at
            least one ID.
    """

    image_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of gallery image UUIDs to delete (at least one required).",
    )


class BulkZipRequest(BaseModel):
    """Request body for the ``POST /api/gallery/bulk-zip`` endpoint.

    Attributes:
        image_ids: List of gallery image UUIDs to include in the zip.  Must
            contain at least one ID.
    """

    image_ids: list[str] = Field(
        ...,
        min_length=1,
        description="List of gallery image UUIDs to zip (at least one required).",
    )


class RuntimeModeOptionResponse(BaseModel):
    """One runtime source mode option available in the UI."""

    mode_key: str
    label: str
    source_kind: str
    default_server_url: str | None
    active_server_url: str | None
    url_editable: bool


class RuntimeModeResponse(BaseModel):
    """Current runtime source mode and all selectable mode options."""

    mode_key: str
    source_kind: str
    active_server_url: str | None
    options: list[RuntimeModeOptionResponse]


class RuntimeModeRequest(BaseModel):
    """Request payload to switch runtime source mode."""

    mode_key: str = Field(min_length=1)
    server_url: str | None = None


class RuntimeAuthResponse(BaseModel):
    """Runtime auth/capability probe result for server-backed snippet loading."""

    mode_key: str
    source_kind: str
    active_server_url: str | None
    session_present: bool
    access_granted: bool
    status: str
    detail: str
    available_worlds: list[dict[str, Any]] = Field(default_factory=list)


class RuntimeLoginRequest(BaseModel):
    """Request payload to authenticate against the active mud-server profile."""

    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class RuntimeLoginResponse(BaseModel):
    """Runtime login result used by image-generator snippet source controls."""

    success: bool
    session_id: str | None = None
    role: str | None = None
    available_worlds: list[dict[str, Any]] = Field(default_factory=list)
    detail: str


class RuntimeLogoutResponse(BaseModel):
    """Runtime logout result for browser-session teardown."""

    success: bool
    detail: str

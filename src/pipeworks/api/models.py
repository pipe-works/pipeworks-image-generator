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

from typing import Any, Literal

from pydantic import BaseModel, Field


class PromptSection(BaseModel):
    """One slot of the dynamic prompt composer (schema v3).

    Each slot carries a curator-supplied label, an authoring mode, and one
    of two text sources (manual free text or a referenced policy/prompt
    snippet). The compiled prompt emits sections in submitted order, using
    the label as a block header and skipping sections that resolve to an
    empty string.
    """

    label: str = Field(
        default="Policy",
        description="Display label rendered as the section block header.",
    )
    mode: Literal["manual", "automated"] = Field(
        default="manual",
        description="'manual' uses manual_text; 'automated' uses automated_prompt_id.",
    )
    manual_text: str | None = Field(
        default=None,
        description="Free-text content (used when mode='manual').",
    )
    automated_prompt_id: str | None = Field(
        default=None,
        description="Identifier of a prompt or policy snippet (used when mode='automated').",
    )


class GenerateRequest(BaseModel):
    """Request body for the ``POST /api/generate`` endpoint.

    Contains every parameter needed to generate a batch of images:
    model selection, prompt composition (curator-ordered policy slots),
    image dimensions, inference settings, seed, and batch size.

    Attributes:
        model_id: Identifier of the model to use (must match an ``id`` value
            in ``models.json``).
        prompt_schema_version: Always ``3`` — required to disambiguate
            from older schemas that may surface in archived metadata.
        sections: Ordered list of policy slots (see ``PromptSection``).
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
    prompt_schema_version: Literal[3] = Field(
        default=3,
        description="Prompt schema version. Always 3 — see `sections`.",
    )
    sections: list[PromptSection] = Field(
        default_factory=list,
        description=(
            "Ordered list of prompt sections. Each section is compiled in "
            "submitted order; empty sections are silently dropped."
        ),
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


class GpuSettingsUpdateRequest(BaseModel):
    """Request body for updating runtime GPU worker settings."""

    use_remote_gpu: bool = Field(
        default=False,
        description="Enable or disable configured remote GPU worker usage.",
    )
    remote_label: str | None = Field(
        default=None,
        description="Optional label shown for the configured remote worker.",
    )
    remote_base_url: str | None = Field(
        default=None,
        description="Remote worker base URL (required when use_remote_gpu is true).",
    )
    bearer_token: str | None = Field(
        default=None,
        description=(
            "Remote worker bearer token. If omitted and no token exists, one "
            "is generated automatically."
        ),
    )
    timeout_seconds: float = Field(
        default=240.0,
        ge=1.0,
        le=3600.0,
        description="Remote worker timeout in seconds.",
    )
    default_to_remote: bool = Field(
        default=False,
        description="Set remote worker as default selection after saving.",
    )


class GpuSettingsTestRequest(BaseModel):
    """Request body for testing remote worker connectivity."""

    remote_base_url: str = Field(min_length=1)
    bearer_token: str | None = None
    timeout_seconds: float = Field(default=8.0, ge=1.0, le=120.0)


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

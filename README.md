[![CI](https://github.com/pipe-works/pipeworks-image-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/pipe-works/pipeworks-image-generator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pipe-works/pipeworks-image-generator/branch/main/graph/badge.svg)](https://codecov.io/gh/pipe-works/pipeworks-image-generator)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# pipeworks-image-generator

`pipeworks-image-generator` is the PipeWorks browser-facing image generation
application. It combines a FastAPI API, a vanilla HTML/CSS/JS frontend, local
or remote GPU execution, and JSON-backed gallery persistence for prompt
composition, generation, and review workflows.

## PipeWorks Workspace

These repositories are designed to live inside a shared PipeWorks workspace
rooted at `/srv/work/pipeworks`.

- `repos/` contains source checkouts only.
- `venvs/` contains per-project virtual environments such as `pw-mud-server`.
- `runtime/` contains mutable runtime state such as databases, exports, session
  files, and caches.
- `logs/` contains service-owned log output when a project writes logs outside
  the process manager.
- `config/` contains workspace-level configuration files that should not be
  treated as source.
- `bin/` contains optional workspace helper scripts.
- `home/` is reserved for workspace-local user data when a project needs it.

Across the PipeWorks ecosphere, the rule is simple: keep source in `repos/`,
keep mutable state outside the repo checkout, and use explicit paths between
repos when one project depends on another.

## What This Repo Owns

This repository is the source of truth for:

- the `pipeworks` CLI entry point
- the FastAPI application under `src/pipeworks/api/`
- the browser UI and static assets under `src/pipeworks/templates/` and
  `src/pipeworks/static/`
- prompt compilation and runtime policy integration for image-generation
  requests
- local and remote GPU-worker orchestration
- gallery metadata and image-management behavior
- deployment templates under `deploy/`

This repository does not own:

- canonical PipeWorks runtime policy state
- mud-server itself
- HuggingFace model hosting
- broader workspace-level host operations outside this repo's own deploy
  templates

## Main App Surfaces

### Browser UI

The browser UI provides:

- model selection and generation controls
- structured prompt composition across subject, setting, details, lighting,
  and atmosphere
- runtime mode and policy-snippet source selection
- gallery browsing, filtering, favourites, and deletion

### FastAPI API

The API exposes:

- `/api/config` for model and runtime metadata
- `/api/generate` for image batch generation
- `/api/prompt/compile` for prompt preview
- `/api/runtime-*` and `/api/policy-prompts` for canonical mud-server-backed
  snippet and auth flows
- `/api/gallery*` and `/api/stats` for gallery persistence and reporting
- `/api/worker/*` for bounded internal remote-worker execution endpoints

### Execution Modes

The application can run in two broad modes:

- controller plus local GPU inference on the same machine
- controller on one machine with image generation forwarded to a remote GPU
  worker over HTTP with bearer-token protection

That split matters. The browser/controller surface and the heavy inference
surface can be colocated, but they do not have to be.

## Relationship To Other PipeWorks Repos

- `pipeworks_mud_server`
  canonical source for policy snippets and runtime login-backed policy APIs
- `pipeworks-image-generator`
  browser-facing image-generation UI, prompt compilation, gallery behavior, and
  local or remote inference orchestration
- `pipeworks-policy-workbench`
  operator and developer workbench for policy objects rather than image
  generation itself

The image generator may consume canonical mud-server policy APIs, but it does
not become the policy authority by doing so.

## Repository Layout

- `src/pipeworks/api/main.py` FastAPI bootstrap, router registration, and CLI
  startup
- `src/pipeworks/api/routers/` route groups for generation, gallery, runtime,
  prompt, and worker endpoints
- `src/pipeworks/api/services/` orchestration for runtime policy, generation,
  worker transport, and prompt resolution
- `src/pipeworks/api/gallery_store.py` JSON-backed gallery persistence helpers
- `src/pipeworks/core/config.py` Pydantic settings and runtime path handling
- `src/pipeworks/core/model_manager.py` Diffusers pipeline lifecycle and
  generation behavior
- `src/pipeworks/templates/index.html` browser UI shell
- `src/pipeworks/static/` CSS, fonts, prompt-library data, and frontend JS
- `tests/` unit and integration coverage
- `docs/` Sphinx documentation sources
- `deploy/` example env, `systemd`, and nginx files

## Quick Start

### Requirements

- Python `>=3.12`
- a PipeWorks workspace rooted at `/srv/work/pipeworks`
- a compatible GPU and ML runtime if you want local inference
- sufficient disk space for model cache and generated outputs
- a Hugging Face access token if you want higher API rate limits or need to
  pull gated/private models
- access to a running `pipeworks_mud_server` if you want canonical
  mud-server-backed snippet workflows rather than local-only prompt library use

### Install

Create a project venv and install from `pyproject.toml`:

```bash
python3.12 -m venv /srv/work/pipeworks/venvs/pw-image-generator
/srv/work/pipeworks/venvs/pw-image-generator/bin/pip install -e ".[dev]"
```

If you also want docs tooling:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pip install -e ".[dev,docs]"
```

For host-managed service use, the venv should be built from a system-level
Python `3.12` install rather than from a user-home interpreter.

### Prepare Runtime Paths

For a workspace-backed run, keep mutable state outside the repo checkout.
A typical layout is:

- model cache under `/srv/work/pipeworks/runtime/image-generator/models`
- generated outputs under `/srv/work/pipeworks/runtime/image-generator/outputs`
- gallery images under `/srv/work/pipeworks/runtime/image-generator/gallery`
- gallery metadata under
  `/srv/work/pipeworks/runtime/image-generator/gallery.json`
- optional read-only host share under `/srv/share/image-generator/gallery`
  bind-mounted from `/srv/work/pipeworks/runtime/image-generator/outputs`
- workspace-managed env/config under `/srv/work/pipeworks/config/image-generator/`

### Prepare Environment

For local development:

```bash
cp .env.example .env
```

Important variables in the current codebase include:

- `PIPEWORKS_SERVER_HOST`
- `PIPEWORKS_SERVER_PORT`
- `PIPEWORKS_MODELS_DIR`
- `PIPEWORKS_OUTPUTS_DIR`
- `PIPEWORKS_GALLERY_DIR`
- `PIPEWORKS_GALLERY_DB`
- `PIPEWORKS_DEVICE`
- `PIPEWORKS_TORCH_DTYPE`
- `PW_POLICY_SOURCE_MODE`
- `PW_POLICY_DEV_MUD_API_BASE_URL`
- `PW_POLICY_PROD_MUD_API_BASE_URL`

Hugging Face token setup:

```bash
export HF_TOKEN=your_token_here
```

Or place it in your local `.env`:

```bash
HF_TOKEN=your_token_here
```

This is optional for public model access, but recommended. Without `HF_TOKEN`,
the app may hit lower anonymous rate limits and model downloads can be slower
or unavailable for gated resources.

### Run Locally

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pipeworks
```

The repo-local default bind is:

- `0.0.0.0:7860`

That default is intended for direct local development. Hosted deployments
should normally override the bind address and port through external env/config
rather than treating the repo default as the service contract.

## Runtime Conventions

Prompt-library behavior follows the split JSON files shipped in
`src/pipeworks/static/data/`:

- `prepend.json`
- `main.json`
- `append.json`
- `models.json`

Prompt APIs currently require `prompt_schema_version=2`.

Gallery behavior is file-backed, not database-backed:

- image files live in the configured gallery directory
- metadata lives in `gallery.json`
- missing image files are reconciled and pruned from metadata on load

Remote GPU-worker support is optional. Worker execution requires explicit URL
and bearer-token configuration and should be treated as an explicit trust
boundary, not as an implicit local shortcut.

## Validation And Development

Run the main checks from the repo root:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pytest -q
/srv/work/pipeworks/venvs/pw-image-generator/bin/ruff check src tests
/srv/work/pipeworks/venvs/pw-image-generator/bin/black --check src tests
/srv/work/pipeworks/venvs/pw-image-generator/bin/mypy src
```

Useful targeted checks:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pytest tests/unit/test_config.py -q
/srv/work/pipeworks/venvs/pw-image-generator/bin/pytest tests/integration/test_api.py -q
```

## Deployment Templates

Host-neutral deployment examples are shipped in:

- `deploy/env/image-generator.env.example`
- `deploy/systemd/pipeworks-image-generator.service`
- `deploy/nginx/images.pipeworks.luminal.local`

These are deployment templates, not the runtime authority themselves. Keep
machine-specific rollout detail in runbooks, MOCs, or host-level docs rather
than in this README.

## Documentation

Additional documentation lives in:

- `docs/`
- `AGENTS.md`
- `CLAUDE.md`

## License

[GPL-3.0-or-later](LICENSE)

# AGENTS.md

## Foundation Must-Dos (Org-Wide)

Read and apply these before repo-specific instructions:

- Local workspace path: `../.github/.github/docs/AGENT_FOUNDATION.md`
- Local workspace path: `../.github/.github/docs/TEST_TAGGING_AND_GITHUB_CHECKLIST.md`
- Canonical URL: `https://github.com/pipe-works/.github/blob/main/.github/docs/AGENT_FOUNDATION.md`
- Canonical URL: `https://github.com/pipe-works/.github/blob/main/.github/docs/TEST_TAGGING_AND_GITHUB_CHECKLIST.md`

Mandatory requirements:

1. Run the GitHub preflight checklist before any `gh` interaction, CI edits, or
   test-tag changes.
2. Preserve required checks (`All Checks Passed`, `Secret Scan (Gitleaks)`).
3. Do not weaken test-tag semantics to reduce runtime.
4. Keep CI optimization changes evidence-based (run IDs, timings, check states).

This file gives coding agents the minimum repo-specific guidance needed to work
safely and efficiently in this project.

## Project Summary

Pipeworks Image Generator is a Python 3.12 FastAPI application for browser-
facing image generation with:

- a FastAPI backend in `src/pipeworks/api/`
- a service/orchestration layer in `src/pipeworks/api/services/`
- a core model/runtime layer in `src/pipeworks/core/`
- a vanilla HTML/CSS/JS frontend in `src/pipeworks/templates/` and
  `src/pipeworks/static/` (composition root + ES modules under
  `static/js/app/`)
- pytest coverage across unit and integration tests, plus Node-based frontend
  tests in `tests/`
- optional remote GPU-worker execution for image generation

The app serves a browser UI, compiles prompts, talks to canonical mud-server
policy APIs when configured, generates images through HuggingFace Diffusers,
and stores gallery metadata in JSON rather than a database.

## Environment

This repo expects Python `3.12`.

- For local development, use a Python `3.12` environment you control.
- For hosted service use, build the venv from a system-level Python `3.12`
  install rather than from any interpreter under a private home directory.
- The intended workspace venv path is
  `/srv/work/pipeworks/venvs/pw-image-generator`.

Typical setup:

```bash
python3.12 -m venv /srv/work/pipeworks/venvs/pw-image-generator
/srv/work/pipeworks/venvs/pw-image-generator/bin/pip install -e ".[dev]"
```

If docs tooling is needed:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pip install -e ".[dev,docs]"
```

For `FLUX.2-klein-4B`, the runtime may require a Diffusers build that includes
`Flux2KleinPipeline`. If PyPI is missing that support, use:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pip install --upgrade "git+https://github.com/huggingface/diffusers.git"
```

## Commands

Run these from the repository root:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pytest -q
/srv/work/pipeworks/venvs/pw-image-generator/bin/ruff check src tests
/srv/work/pipeworks/venvs/pw-image-generator/bin/black --check src tests
/srv/work/pipeworks/venvs/pw-image-generator/bin/mypy src
/srv/work/pipeworks/venvs/pw-image-generator/bin/pipeworks
```

Useful targeted commands:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pytest tests/unit/test_config.py -q --no-cov
/srv/work/pipeworks/venvs/pw-image-generator/bin/pytest tests/integration/test_api.py -q --no-cov
/srv/work/pipeworks/venvs/pw-image-generator/bin/ruff check src tests --fix
/srv/work/pipeworks/venvs/pw-image-generator/bin/black src tests
```

Frontend tests are Node ES modules under `tests/frontend/`:

```bash
node --test tests/frontend
```

Coverage configuration lives in `.coveragerc`, `codecov.yml`, and `pytest.ini`.
A `.pre-commit-config.yaml` is checked in; install hooks with
`/srv/work/pipeworks/venvs/pw-image-generator/bin/pre-commit install` if you
intend to commit.

## Release Automation

This repo uses `release-please` and expects conventional commit semantics for
releasable changes.

- Use commit and PR titles with the correct prefix when the change should be
  eligible for a release entry.
- Prefer `feat:` for user-facing features, `fix:` for bug fixes, `perf:` for
  performance work, `docs:` for documentation, and `chore:` only for
  non-releasable housekeeping.
- Do not use untagged titles like `Update X` or `Add Y` when the work should
  trigger release automation.
- If you are preparing a PR that should ship, make sure the final title keeps
  the conventional prefix.
- Do not merge shipping work with plain commit titles. `release-please` will
  run on `main`, but it will skip unparseable commits and no release PR will
  be created.

Release-please configuration: `release-please-config.json` and
`.release-please-manifest.json`.

## Verified Baseline

The current codebase and deploy guidance assume:

- package metadata lives in `pyproject.toml`
- the top-level README is public-facing and workspace-oriented
- gallery metadata can be moved outside packaged `static/data` via
  `PIPEWORKS_GALLERY_DB` (default falls back to `data_dir/gallery.json` for
  backward compatibility)
- `/static/gallery` can be mounted from a host-managed directory via
  `PIPEWORKS_GALLERY_DIR`
- deploy templates live under `deploy/`

When validating a hosted-service change, also check:

- the service venv resolves to a system-level Python `3.12` interpreter
- hosted service bind/config is driven by external env files
- Luminal host-managed browser service uses localhost bind rather than repo
  default `0.0.0.0:7860`

## Important Paths

Backend:

- `src/pipeworks/api/main.py`: FastAPI app bootstrap, static mounts, route
  registration, lifespan, and CLI entry point
- `src/pipeworks/api/models.py`: request/response validation models
- `src/pipeworks/api/gallery_store.py`: JSON-backed gallery persistence logic
- `src/pipeworks/api/mud_api_client.py`: MUD policy API HTTP transport with
  authenticated and anonymous helpers
- `src/pipeworks/api/prompt_builder.py`: prompt assembly from prepend/main/
  append fragments
- `src/pipeworks/api/runtime_mode.py`: runtime/policy mode resolution helpers
- `src/pipeworks/api/routers/`: route groups — `generation.py`, `prompt.py`,
  `runtime.py`, `gallery.py`, `gpu_worker.py`
- `src/pipeworks/api/services/`: orchestration layer —
  `generation_runtime.py`, `gpu_workers.py`, `http_transport.py`,
  `prompt_catalog.py`, `prompt_resolution.py`, `runtime_policy.py`,
  `zip_metadata.py`
- `src/pipeworks/core/config.py`: Pydantic settings, GPU-worker config, and
  runtime path handling
- `src/pipeworks/core/model_manager.py`: model load/unload/generate lifecycle
  and per-model runtime support detection
- `src/pipeworks/core/prompt_token_counter.py`: prompt-token counting per model

Frontend:

- `src/pipeworks/templates/index.html`: main UI shell
- `src/pipeworks/static/js/app.js`: frontend composition root
- `src/pipeworks/static/js/app/`: feature modules (`api-client.mjs`,
  `dom-utils.mjs`, `generation-flow.mjs`, `gallery-manager.mjs`,
  `prompt-composer.mjs`, `runtime-gpu-controller.mjs`, `state.mjs`)
- `src/pipeworks/static/js/{gallery-context,gallery-navigation,output-lightbox}.mjs`:
  shared frontend helpers also exercised by Node tests
- `src/pipeworks/static/data/`: prompt catalog (`prepend.json`, `main.json`,
  `append.json`) and `models.json`

Tests:

- `tests/conftest.py`: shared fixtures and mocked app setup
- `tests/unit/`: backend unit tests (config, model manager, gallery store,
  prompt builder/catalog/token counter, MUD API client, runtime mode, API
  models, ZIP packaging, CLI entry, frontend template/app smoke)
- `tests/integration/test_api.py`: HTTP-level FastAPI integration tests
- `tests/frontend/`: Node `node --test` ES-module tests for shared frontend
  helpers

Deploy:

- `deploy/env/image-generator.env.example`
- `deploy/systemd/pipeworks-image-generator.service`
- `deploy/nginx/images.pipeworks.luminal.local`

## Architecture Notes

- `PipeworksConfig` is loaded via Pydantic Settings using `PIPEWORKS_*`
  environment variables and `.env`. GPU workers are modelled by
  `GpuWorkerConfig` (local or remote, with bearer-token auth for remote).
- `pipeworks.api.main` resolves important config-derived paths at import time,
  including static roots, gallery location, gallery metadata path, and the GPU
  workers runtime-settings file (`outputs_dir/gpu_workers.runtime.json`).
- The app mounts `/static` from packaged static assets and `/static/gallery`
  from the configured gallery directory; gallery mount is registered before
  the generic `/static` mount so it wins on overlap.
- Route groups live under `src/pipeworks/api/routers/` and are wired with
  explicit dependency dataclasses (`*RouterDependencies`) constructed in
  `api/main.py`. Service singletons live at module scope so tests can patch
  them.
- Orchestration helpers live under `src/pipeworks/api/services/`:
  `RuntimePolicyService`, `GpuWorkerService`, `GenerationRuntimeService`,
  prompt catalog loader, and HTTP transport helpers.
- Prompt libraries are split across `static/data/prepend.json`,
  `static/data/main.json`, and `static/data/append.json`; legacy single-file
  fallback only emits a warning at startup.
- Gallery persistence is file-backed JSON, not SQLite.
- Generated images are stored in the configured gallery directory; ZIP export
  metadata is built via `services/zip_metadata.py`.
- The frontend is plain HTML/CSS/JS with no bundler or build step. `app.js` is
  a composition root that imports ES modules from `static/js/app/` and the
  shared `static/js/*.mjs` helpers.
- The model manager keeps a single Diffusers pipeline in memory at a time and
  exposes per-model runtime support detection
  (`get_model_runtime_support`).
- Remote GPU-worker mode is part of the supported architecture and should be
  treated as a real execution boundary, with caps on batch size
  (`remote_worker_max_batch_size`) and decoded payload bytes
  (`remote_worker_max_decoded_bytes`).

## Constraints That Matter

- Do not reintroduce `requirements.txt` as the dependency authority; use
  `pyproject.toml`.
- Avoid changes that silently move mutable runtime state back into the repo
  working tree.
- Do not treat a user-home interpreter as acceptable for the hosted-service
  venv.
- Turbo models require `guidance_scale=0.0`; do not remove that enforcement
  casually.
- Prompt APIs require v2 payloads (`prompt_schema_version=2`).
- Prompt catalog loading uses split files only (legacy fallback warns).
- Runtime URL resolution uses canonical env vars:
  `PW_POLICY_DEV_MUD_API_BASE_URL` and `PW_POLICY_PROD_MUD_API_BASE_URL`.
- Worker API auth tokens come from `PIPEWORKS_WORKER_API_BEARER_TOKENS` plus
  any remote-worker bearer tokens in `gpu_workers`.
- Be careful with import-time side effects in `config.py` and `api/main.py`;
  path handling and directory creation are part of the app contract.
- Keep changes consistent with the current stack: FastAPI, Pydantic, pytest,
  Ruff, Black, mypy, and vanilla frontend assets.
- Do not introduce a frontend build system unless explicitly requested.
- `black` is pinned org-wide in `pyproject.toml`; do not bump it casually.

## Working Style

- Prefer targeted edits over broad refactors.
- Add or update tests when behavior changes.
- For backend changes, run at least the most relevant pytest subset.
- For API changes, prefer validating with `tests/integration/test_api.py`.
- For config or path changes, inspect both `src/pipeworks/core/` and
  `src/pipeworks/api/main.py` before editing.
- For frontend changes, run the Node tests under `tests/frontend/` when the
  helpers they cover are touched, and check the unit-level template/app
  smoke tests.
- For deploy changes, keep repo templates, host-managed env guidance, and
  README wording aligned.

## Notes For Future Agents

- There is existing repo guidance in `CLAUDE.md`; keep this file aligned with
  it when updating instructions.
- If you change developer workflow, update both `AGENTS.md` and any overlapping
  command documentation in `README.md` or `CLAUDE.md`.
- If you change hosted-service assumptions, also check the relevant MOC or
  project-map entry so repo docs and host docs do not drift apart.

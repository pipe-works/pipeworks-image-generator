# CLAUDE.md

## Repo Overview

Pipeworks Image Generator is a FastAPI-based image generation application with:

- a browser-facing UI in `src/pipeworks/templates/` and `src/pipeworks/static/`
- API routes under `src/pipeworks/api/`
- generation/runtime config under `src/pipeworks/core/`
- file-backed gallery persistence
- optional remote GPU-worker execution

The package entry point is:

```bash
pipeworks
```

That command launches the FastAPI app defined in `pipeworks.api.main`.

## Workspace Assumptions

The intended shared workspace root is:

```bash
/srv/work/pipeworks
```

Important workspace surfaces:

- source repo: `/srv/work/pipeworks/repos/pipeworks-image-generator`
- project venv: `/srv/work/pipeworks/venvs/pw-image-generator`
- runtime state: `/srv/work/pipeworks/runtime/image-generator`
- logs: `/srv/work/pipeworks/logs/image-generator`
- workspace-managed config:
  `/srv/work/pipeworks/config/image-generator`

Keep source in the repo checkout and mutable state outside it.

## Python And Venv Expectations

- Python `3.12` is required.
- `pyproject.toml` is the dependency authority.
- For hosted service use, the venv must be built from a system-level Python
  `3.12` install, not from a private interpreter under a user home directory.

Typical install:

```bash
python3.12 -m venv /srv/work/pipeworks/venvs/pw-image-generator
/srv/work/pipeworks/venvs/pw-image-generator/bin/pip install -e ".[dev]"
```

If docs tooling is needed:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pip install -e ".[dev,docs]"
```

## Local Run Posture

Local repo-default launch:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pipeworks
```

The repo-local default bind is:

- host `0.0.0.0`
- port `7860`

That is a development default, not the hosted-service contract.

## Hosted Service Posture

For hosted service use, the expected bind is driven by external env/config.

Current Luminal-oriented host-managed example:

- bind host `127.0.0.1`
- bind port `8400`
- nginx front door in front of the backend
- workspace-managed writable paths for models, outputs, gallery images, and
  gallery metadata

Do not treat the repo-local `7860` default as the host-managed service truth.

## Important Config Variables

Core runtime settings:

- `PIPEWORKS_SERVER_HOST`
- `PIPEWORKS_SERVER_PORT`
- `PIPEWORKS_MODELS_DIR`
- `PIPEWORKS_OUTPUTS_DIR`
- `PIPEWORKS_GALLERY_DIR`
- `PIPEWORKS_GALLERY_DB`
- `PIPEWORKS_DEVICE`
- `PIPEWORKS_TORCH_DTYPE`

Runtime policy integration:

- `PW_POLICY_SOURCE_MODE`
- `PW_POLICY_DEV_MUD_API_BASE_URL`
- `PW_POLICY_PROD_MUD_API_BASE_URL`

Optional remote worker settings:

- `PIPEWORKS_GPU_WORKERS`
- `PIPEWORKS_DEFAULT_GPU_WORKER_ID`
- `PIPEWORKS_WORKER_API_BEARER_TOKENS`

## Code Areas That Matter

- `src/pipeworks/api/main.py`
  app bootstrap, static mounts, CLI startup, and module-level path resolution
- `src/pipeworks/api/routers/generation.py`
  generation and cancellation API flows
- `src/pipeworks/api/routers/gpu_worker.py`
  remote worker API behavior
- `src/pipeworks/api/gallery_store.py`
  file-backed gallery persistence
- `src/pipeworks/core/config.py`
  settings, path defaults, and directory creation
- `src/pipeworks/core/model_manager.py`
  model lifecycle and generation execution

## Development Commands

Main checks:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pytest -q
/srv/work/pipeworks/venvs/pw-image-generator/bin/ruff check src tests
/srv/work/pipeworks/venvs/pw-image-generator/bin/black --check src tests
/srv/work/pipeworks/venvs/pw-image-generator/bin/mypy src
```

Useful focused checks:

```bash
/srv/work/pipeworks/venvs/pw-image-generator/bin/pytest tests/unit/test_config.py -q --no-cov
/srv/work/pipeworks/venvs/pw-image-generator/bin/pytest tests/integration/test_api.py -q --no-cov
```

## Editing Guidance

- Keep `README.md`, `AGENTS.md`, and `CLAUDE.md` aligned when workflow or
  deploy assumptions change.
- Do not reintroduce stale `requirements.txt`-based guidance.
- When a change should ship, the final commit and PR title must use a
  conventional prefix such as `feat:` or `fix:`. This repo uses
  `release-please`, and plain titles like `Add X` or `Update Y` will be
  ignored by release automation.
- Be careful when editing `config.py` or `api/main.py`; they control real path
  and mount behavior for both local development and hosted service use.
- If a change affects hosted-service assumptions, also check deploy templates
  under `deploy/`.
- If a change affects runtime paths, update tests and fixture expectations.

## Deploy Templates

This repo ships example deploy surfaces under:

- `deploy/env/image-generator.env.example`
- `deploy/systemd/pipeworks-image-generator.service`
- `deploy/nginx/images.pipeworks.luminal.local`

Treat those as checked-in templates. Machine-specific rollout state should live
outside the repo.

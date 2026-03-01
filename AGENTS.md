# AGENTS.md

This file gives coding agents the minimum repo-specific guidance needed to work safely and efficiently in this project.

## Project Summary

Pipeworks Image Generator is a Python 3.12 FastAPI application for local image generation with:

- a FastAPI backend in `src/pipeworks/api/`
- a core model/runtime layer in `src/pipeworks/core/`
- a vanilla HTML/CSS/JS frontend in `src/pipeworks/templates/` and `src/pipeworks/static/`
- pytest coverage across unit and integration tests in `tests/`

The app serves a web UI, generates images through HuggingFace Diffusers,
and stores gallery metadata in JSON rather than a database.

## Environment

This repo uses `pyenv`.

- `.python-version` is set to `pig`
- verified interpreter: `Python 3.12.8`
- prefer `pyenv exec ...` for all Python, pip, pytest, ruff, black, and app commands

Typical setup:

```bash
pyenv local pig
pyenv exec pip install -e ".[dev]"
```

If the environment is missing dependencies, install them into the active
`pyenv` environment instead of falling back to system Python.

## Commands

Run these from the repository root:

```bash
pyenv exec pytest -q
pyenv exec ruff check src tests
pyenv exec black --check src tests
pyenv exec pipeworks
```

Useful targeted commands:

```bash
pyenv exec pytest tests/unit/test_config.py -q
pyenv exec pytest tests/integration/test_api.py -q
pyenv exec ruff check src tests --fix
pyenv exec black src tests
```

## Verified Baseline

These commands were run successfully in this repository:

- `pyenv exec pytest -q`
- `pyenv exec ruff check src tests`
- `pyenv exec black --check src tests`

At the time of writing:

- `128` tests passed
- total coverage was `93.88%`

## Important Paths

- `src/pipeworks/api/main.py`: FastAPI app, routes, lifespan, CLI entry point
- `src/pipeworks/api/models.py`: request/response validation models
- `src/pipeworks/api/prompt_builder.py`: prompt compilation logic
- `src/pipeworks/core/config.py`: Pydantic settings and directory setup
- `src/pipeworks/core/model_manager.py`: model load/unload/generate lifecycle
- `src/pipeworks/templates/index.html`: main UI shell
- `src/pipeworks/static/js/app.js`: frontend behavior
- `tests/conftest.py`: shared fixtures and mocked app setup

## Architecture Notes

- `PipeworksConfig` is loaded via Pydantic Settings using `PIPEWORKS_*` environment variables and `.env`.
- `pipeworks.api.main` resolves config paths at import time and mounts static files immediately.
- Gallery persistence is file-based JSON under `static/data/`, not SQLite.
- Generated images are stored under `static/gallery/`.
- The frontend is plain HTML/CSS/JS with no bundler or build step.
- The model manager keeps a single diffusers pipeline in memory at a time.

## Constraints That Matter

- Always use `pyenv exec` for repo commands unless there is a strong reason not to.
- Avoid tests or changes that download real models unless the task explicitly requires it; the normal test suite uses mocks.
- Turbo models require `guidance_scale=0.0`; do not remove that enforcement casually.
- Be careful with import-time side effects in `config.py` and `api/main.py`;
  path handling and directory creation are part of the app contract.
- Keep changes consistent with the current stack: FastAPI, Pydantic, pytest, Ruff, Black, and vanilla frontend assets.
- Do not introduce a frontend build system unless the user explicitly asks for one.

## Working Style

- Prefer targeted edits over broad refactors.
- Add or update tests when behavior changes.
- For backend changes, run at least the most relevant pytest subset through `pyenv exec`.
- For API changes, prefer validating with `tests/integration/test_api.py`.
- For config or model lifecycle changes, inspect both `src/pipeworks/core/` and the related tests before editing.

## Notes For Future Agents

- There is existing repo guidance in `CLAUDE.md`; keep this file aligned with it when updating instructions.
- If you change developer workflow, update both `AGENTS.md` and any overlapping command documentation in `README.md` or `CLAUDE.md`.

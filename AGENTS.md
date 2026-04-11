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
- a core model/runtime layer in `src/pipeworks/core/`
- a vanilla HTML/CSS/JS frontend in `src/pipeworks/templates/` and
  `src/pipeworks/static/`
- pytest coverage across unit and integration tests in `tests/`
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

## Verified Baseline

The current codebase and deploy guidance assume:

- package metadata lives in `pyproject.toml`
- the top-level README is public-facing and workspace-oriented
- gallery metadata can be moved outside packaged `static/data`
- `/static/gallery` can be mounted from a host-managed directory
- deploy templates live under `deploy/`

When validating a hosted-service change, also check:

- the service venv resolves to a system-level Python `3.12` interpreter
- hosted service bind/config is driven by external env files
- Luminal host-managed browser service uses localhost bind rather than repo
  default `0.0.0.0:7860`

## Important Paths

- `src/pipeworks/api/main.py`: FastAPI app bootstrap, static mounts, route
  registration, and CLI entry point
- `src/pipeworks/api/models.py`: request/response validation models
- `src/pipeworks/api/gallery_store.py`: JSON-backed gallery persistence logic
- `src/pipeworks/api/routers/`: route groups for generation, prompt, runtime,
  gallery, and worker behavior
- `src/pipeworks/core/config.py`: Pydantic settings and runtime path handling
- `src/pipeworks/core/model_manager.py`: model load/unload/generate lifecycle
- `src/pipeworks/templates/index.html`: main UI shell
- `src/pipeworks/static/js/app.js`: frontend composition root
- `tests/conftest.py`: shared fixtures and mocked app setup
- `deploy/`: checked-in env, `systemd`, and nginx examples

## Architecture Notes

- `PipeworksConfig` is loaded via Pydantic Settings using `PIPEWORKS_*`
  environment variables and `.env`.
- `pipeworks.api.main` resolves important config-derived paths at import time,
  including static roots, gallery location, and gallery metadata path.
- The app mounts `/static` from packaged static assets and `/static/gallery`
  from the configured gallery directory.
- Route groups live under `src/pipeworks/api/routers/`.
- Orchestration helpers live under `src/pipeworks/api/services/`.
- Prompt libraries are split across `static/data/prepend.json`,
  `static/data/main.json`, and `static/data/append.json`.
- Gallery persistence is file-backed JSON, not SQLite.
- Generated images are stored in the configured gallery directory.
- The frontend is plain HTML/CSS/JS with no bundler or build step.
- The model manager keeps a single Diffusers pipeline in memory at a time.
- Remote GPU-worker mode is part of the supported architecture and should be
  treated as a real execution boundary.

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
- Prompt catalog loading uses split files only.
- Runtime URL resolution uses canonical env vars:
  `PW_POLICY_DEV_MUD_API_BASE_URL` and `PW_POLICY_PROD_MUD_API_BASE_URL`.
- Be careful with import-time side effects in `config.py` and `api/main.py`;
  path handling and directory creation are part of the app contract.
- Keep changes consistent with the current stack: FastAPI, Pydantic, pytest,
  Ruff, Black, and vanilla frontend assets.
- Do not introduce a frontend build system unless explicitly requested.

## Working Style

- Prefer targeted edits over broad refactors.
- Add or update tests when behavior changes.
- For backend changes, run at least the most relevant pytest subset.
- For API changes, prefer validating with `tests/integration/test_api.py`.
- For config or path changes, inspect both `src/pipeworks/core/` and
  `src/pipeworks/api/main.py` before editing.
- For deploy changes, keep repo templates, host-managed env guidance, and
  README wording aligned.

## Notes For Future Agents

- There is existing repo guidance in `CLAUDE.md`; keep this file aligned with
  it when updating instructions.
- If you change developer workflow, update both `AGENTS.md` and any overlapping
  command documentation in `README.md` or `CLAUDE.md`.
- If you change hosted-service assumptions, also check the relevant MOC or
  project-map entry so repo docs and host docs do not drift apart.

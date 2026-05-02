# Repository Guidelines

## Project Structure & Module Organization
Core application code lives under `src/pipeworks/`. Use `src/pipeworks/api/` for FastAPI routes, request models, and runtime services, `src/pipeworks/core/` for shared config and model management, `src/pipeworks/templates/` for the HTML shell, and `src/pipeworks/static/` for CSS, fonts, JSON prompt data, and frontend JavaScript. Tests are grouped under `tests/unit/` and `tests/integration/`. Documentation sources live in `docs/source/`, deploy templates in `deploy/`, and sample prompt inputs in `examples/inputs/`.

## Build, Test, and Development Commands
Install into the workspace venv with `pip install -e ".[dev]"`; add `,docs` when working on Sphinx docs. Run the app locally with `pipeworks`. Main checks:

- `pytest` runs the full test suite with coverage.
- `pytest tests/unit/test_config.py -q --no-cov` runs a focused test file quickly.
- `ruff check src tests` runs linting and import-order checks.
- `black --check src tests` validates formatting.
- `mypy src` runs static type checks.
- `make -C docs html` builds the documentation site.

## Coding Style & Naming Conventions
Target Python `3.12`. Black and Ruff both use a `100` character line length; do not hand-format against a different width. Follow existing naming: `snake_case` for Python modules/functions, `PascalCase` for classes, and scoped conventional commit prefixes such as `feat(api): ...` or `fix(lora): ...`. Match surrounding indentation: Python uses 4 spaces, frontend JS/CSS uses 2 spaces. Keep mutable runtime state out of the repo checkout and use workspace paths under `/srv/work/pipeworks/runtime/`.

## Testing Guidelines
Pytest discovers `test_*.py` and `*_test.py` under `tests/`. Use markers intentionally: `unit`, `integration`, `slow`, and `requires_model`. The repository enforces `--cov-fail-under=70`, so new features should include coverage for API routes, config changes, and gallery/runtime behavior. Prefer small unit tests beside touched modules, then add integration coverage when a change crosses router or service boundaries.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commits and release-friendly prefixes, for example `fix(lora): ...` and `feat(lora): ...`. Keep subjects imperative and specific. Pull requests should summarize user-visible behavior, call out config or deploy impacts, link the relevant issue, and include screenshots for UI changes. If you change runtime paths, env vars, or hosted-service assumptions, update `README.md`, `CLAUDE.md`, and deploy templates in the same PR.

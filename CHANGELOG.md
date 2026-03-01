# Changelog

## [0.2.5](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.4...pipeworks-image-generator-v0.2.5) (2026-03-01)


### Bug Fixes

* reconcile gallery counts with filesystem state ([f4f268f](https://github.com/pipe-works/pipeworks-image-generator/commit/f4f268f989d8ccb3657c98f70a08986268c4ed2f))
* reconcile gallery counts with filesystem state ([c190a49](https://github.com/pipe-works/pipeworks-image-generator/commit/c190a49b6b804de869e1487538b1dab289603287))

## [0.2.4](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.3...pipeworks-image-generator-v0.2.4) (2026-03-01)


### Bug Fixes

* correct gallery numbering and lightbox transport ([dfd6337](https://github.com/pipe-works/pipeworks-image-generator/commit/dfd6337b3b6c389efd37a7c613b21943a5f2dc82))
* correct gallery numbering and lightbox transport ([57fde85](https://github.com/pipe-works/pipeworks-image-generator/commit/57fde859244f0d58f306b9ce6633560e8224af4a))

## [0.2.3](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.2...pipeworks-image-generator-v0.2.3) (2026-02-23)


### Features

* add per-section token counters with model-specific limits ([7ee46e7](https://github.com/pipe-works/pipeworks-image-generator/commit/7ee46e7db27cc41040296918d2ec2b7a73794a4a))
* add per-section token counters with model-specific limits ([0d9fff7](https://github.com/pipe-works/pipeworks-image-generator/commit/0d9fff721f9aa08ca8425cbbbca7b7f5c7de0fc8))

## [0.2.2](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.1...pipeworks-image-generator-v0.2.2) (2026-02-23)


### Bug Fixes

* skip boilerplate in manual prepend/append mode ([#16](https://github.com/pipe-works/pipeworks-image-generator/issues/16)) ([fcef29b](https://github.com/pipe-works/pipeworks-image-generator/commit/fcef29b5dcb6cf39bd47210f1d74c210d7d946a1))

## [0.2.1](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.0...pipeworks-image-generator-v0.2.1) (2026-02-23)


### Features

* add manual override for prepend and append prompt sections ([#14](https://github.com/pipe-works/pipeworks-image-generator/issues/14)) ([6ba6e9a](https://github.com/pipe-works/pipeworks-image-generator/commit/6ba6e9adbbfe1809255675fa1ee805aeee4a7a43))

## [0.2.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.1.2...pipeworks-image-generator-v0.2.0) (2026-02-23)


### ⚠ BREAKING CHANGES

* Gradio UI replaced with FastAPI REST API. Entry point unchanged (`pipeworks`) but now serves a web UI at :7860 via FastAPI instead of Gradio.

### Features

* replace Gradio architecture with FastAPI + vanilla JS frontend ([#11](https://github.com/pipe-works/pipeworks-image-generator/issues/11)) ([1919fd5](https://github.com/pipe-works/pipeworks-image-generator/commit/1919fd55c3088797cf2a654136c31aa704c05547))


### Bug Fixes

* automate version display and remove demo references ([#13](https://github.com/pipe-works/pipeworks-image-generator/issues/13)) ([6cf6f31](https://github.com/pipe-works/pipeworks-image-generator/commit/6cf6f31a19fbb3d7b910614cbb095be2cdbed3b4))

## [0.1.2](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.1.1...pipeworks-image-generator-v0.1.2) (2026-01-27)


### Bug Fixes

* **packaging:** exclude __pycache__ from distribution ([#9](https://github.com/pipe-works/pipeworks-image-generator/issues/9)) ([dd15cac](https://github.com/pipe-works/pipeworks-image-generator/commit/dd15cac1f3feef17b0ead4c5050e9896d32de2bf))

## [0.1.1](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.1.0...pipeworks-image-generator-v0.1.1) (2026-01-26)


### Features

* add example prompt files for user onboarding ([9c7dd31](https://github.com/pipe-works/pipeworks-image-generator/commit/9c7dd31c73f0c0dd5793721a419bf2392cf9191c))
* add pytest to pre-commit hooks + relax test linting ([23c3d3e](https://github.com/pipe-works/pipeworks-image-generator/commit/23c3d3e81e9cf3c6700f618cb049a6513f0727dd))
* Adopt organization-wide standards and reusable CI workflow ([0236beb](https://github.com/pipe-works/pipeworks-image-generator/commit/0236beb2de0fef8ee1ffdf274b55c10af3aa3c1e))
* **ci:** add release automation workflows ([#6](https://github.com/pipe-works/pipeworks-image-generator/issues/6)) ([5e47fe9](https://github.com/pipe-works/pipeworks-image-generator/commit/5e47fe96e67e68db394218e2403360bb01520285))
* **ci:** add workflow_dispatch trigger to release-please ([6ba3aaa](https://github.com/pipe-works/pipeworks-image-generator/commit/6ba3aaa06236069733f6ca6f871cc68b32378825))
* Enable documentation builds in CI ([3c29318](https://github.com/pipe-works/pipeworks-image-generator/commit/3c293188e9986cc0506733661d7901da6a7681d2))
* Upgrade to enhanced organization pre-commit standards ([9e76e9b](https://github.com/pipe-works/pipeworks-image-generator/commit/9e76e9b54937336a0937381baf4d0b96e8caa632))


### Bug Fixes

* CI now runs integration tests to capture full coverage ([af70ffd](https://github.com/pipe-works/pipeworks-image-generator/commit/af70ffd33262014ac57bbfbcd8d1d1178ebdd822))
* **ci:** add required permissions to release-please workflow ([189b84a](https://github.com/pipe-works/pipeworks-image-generator/commit/189b84a1173b9be2bd974d56a8d0290e4aff3b38))
* **gallery:** always rescan on tab switch to show latest images ([6ddfa28](https://github.com/pipe-works/pipeworks-image-generator/commit/6ddfa2831c39636b851786cf41fe0ecb79ccaa31))
* **gallery:** resolve prompt indexing and improve UI layout ([2cc7771](https://github.com/pipe-works/pipeworks-image-generator/commit/2cc7771563a9efd0f7120fba04c0ab73441ba736))
* Update CI workflow to use renamed python-versions parameter ([e75ef11](https://github.com/pipe-works/pipeworks-image-generator/commit/e75ef113b1b15202211b7f3271c4b8581e54c40a))


### Documentation

* Fix README markdown formatting ([9b2510f](https://github.com/pipe-works/pipeworks-image-generator/commit/9b2510fcf62bed7f128a6874379411b045ecb474))

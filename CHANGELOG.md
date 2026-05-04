# Changelog

## [0.15.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.14.3...pipeworks-image-generator-v0.15.0) (2026-05-04)


### Features

* **composer:** dynamic ordered policy slots replace 5 hardcoded sections ([05afab7](https://github.com/pipe-works/pipeworks-image-generator/commit/05afab71644a1a2398d68f6de99a16a8ca159935))
* **composer:** dynamic ordered policy slots replace 5 hardcoded sections ([9e066ab](https://github.com/pipe-works/pipeworks-image-generator/commit/9e066ab912dcb1a18acc2e1eb51bffcd87dcf0a2))
* **lora:** add body action dataset tiles ([b87921c](https://github.com/pipe-works/pipeworks-image-generator/commit/b87921ca6585c7e7beef54d4abf89f25feb1fc44))
* **lora:** add facial expression dataset tiles ([922bed9](https://github.com/pipe-works/pipeworks-image-generator/commit/922bed93a15e130500593c22885947f4a9c07647))
* **lora:** add LoRA Dataset tab driven by canonical `location` policies ([c8324ef](https://github.com/pipe-works/pipeworks-image-generator/commit/c8324ef736213c19b88e8d2c96b311e81939dfd2))
* **lora:** add LoRA Dataset tab driven by canonical `location` policies ([61253d3](https://github.com/pipe-works/pipeworks-image-generator/commit/61253d3e45e369982d7b265e67763a03a5d9890a))
* **lora:** default-on shared seed across tiles, with toggle next to Generate all ([99ac69e](https://github.com/pipe-works/pipeworks-image-generator/commit/99ac69eb70dd242d3e1330b2efd945f5dc2f0738))
* **lora:** default-on shared seed across tiles, with toggle next to Generate all ([4aedd2a](https://github.com/pipe-works/pipeworks-image-generator/commit/4aedd2ab46a25bbe44bfa1b67bc4d1caaa16e731))
* **lora:** freeze prompt placeholders at run creation, surface seed in tile UI ([c4fa3b8](https://github.com/pipe-works/pipeworks-image-generator/commit/c4fa3b82ffd586bf9df328808aeb201f0ee7bd4f))
* **lora:** freeze prompt placeholders at run creation, surface seed in tile UI ([dd08920](https://github.com/pipe-works/pipeworks-image-generator/commit/dd08920769cdfabd3b0c2e448cca87f0eebfe043))
* **lora:** generalise slots to tile-kind, add Character Sheet turnaround ([39153b6](https://github.com/pipe-works/pipeworks-image-generator/commit/39153b64a9e25e82cbc68f683c7d4f739a84a77b))
* **lora:** generalise slots to tile-kind, add Character Sheet turnaround ([3a975c1](https://github.com/pipe-works/pipeworks-image-generator/commit/3a975c1f28cbec128a50a08c9765c2efdac0d0d0))
* **lora:** split character sheet into directional views ([ffbeed9](https://github.com/pipe-works/pipeworks-image-generator/commit/ffbeed9fea9c69c0248387f07535f070e943a665))
* **lora:** split character sheet into directional views ([0cfaf35](https://github.com/pipe-works/pipeworks-image-generator/commit/0cfaf3585a512d290961f053a57a21e1614d5489))
* **prompt:** add v3 dynamic-section schema alongside v2 ([9c7ceb7](https://github.com/pipe-works/pipeworks-image-generator/commit/9c7ceb703c9fd3c18492074c869f372a5b108603))
* **prompt:** add v3 dynamic-section schema alongside v2 ([6031c12](https://github.com/pipe-works/pipeworks-image-generator/commit/6031c12b7515168990e77ee35f7d11cff11d386a))
* **snippets:** expose canonical `location` policy as prompt snippet ([707f7e4](https://github.com/pipe-works/pipeworks-image-generator/commit/707f7e475de8700907c73e4e7a521ec6df6be4ed))
* **snippets:** expose canonical `location` policy as prompt snippet ([8943644](https://github.com/pipe-works/pipeworks-image-generator/commit/894364460fdcf2ea7f1b36caa85668cc5ee75c4e))
* **snippets:** include tone_profile policies in prompt-composer dropdown ([#92](https://github.com/pipe-works/pipeworks-image-generator/issues/92)) ([571d921](https://github.com/pipe-works/pipeworks-image-generator/commit/571d9210a3636d37c9c541420c600007389ee528))
* **ui:** keep selected snippet visible in dropdown + AGENTS.md refresh ([#90](https://github.com/pipe-works/pipeworks-image-generator/issues/90)) ([d58f3eb](https://github.com/pipe-works/pipeworks-image-generator/commit/d58f3eb3be72505f2afbbfb7828cc76c7d756409))


### Fixes

* **lightbox:** drop vestigial onImageChange callback throwing on render ([#96](https://github.com/pipe-works/pipeworks-image-generator/issues/96)) ([0b86485](https://github.com/pipe-works/pipeworks-image-generator/commit/0b86485fe18c6014ac8ac07828d4f2b7e405cc67))
* **lora:** undefined tile labels, truncated picker preview, add tile lightbox ([d8c48d1](https://github.com/pipe-works/pipeworks-image-generator/commit/d8c48d1f5bcc4ad75328f3e8cfdd421dbb296078))
* **lora:** undefined tile labels, truncated picker preview, add tile lightbox ([0c95a34](https://github.com/pipe-works/pipeworks-image-generator/commit/0c95a3437214d05e820c072f8296721ef65ecb94))
* **lora:** use single-column layout for Character Sheet picker ([49dc138](https://github.com/pipe-works/pipeworks-image-generator/commit/49dc1384dd17674aa1f66a0fd387cd7a92b9f7ea))
* **runtime:** correct image-generator share mounts ([bc10f8e](https://github.com/pipe-works/pipeworks-image-generator/commit/bc10f8e2fb10d586533d7a35d937b30258dcb8c8))


### Documentation

* add repository contributor guide ([4c2759b](https://github.com/pipe-works/pipeworks-image-generator/commit/4c2759bcf5e646b7e4de232f648c581130eadb77))
* **deploy:** document read-only outputs share ([#108](https://github.com/pipe-works/pipeworks-image-generator/issues/108)) ([4f0642c](https://github.com/pipe-works/pipeworks-image-generator/commit/4f0642cbc8c46f29528bd44652c431a1ff78c798))

## [0.14.3](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.14.2...pipeworks-image-generator-v0.14.3) (2026-04-11)


### Fixes

* **release:** clarify release-please commit requirements ([988f130](https://github.com/pipe-works/pipeworks-image-generator/commit/988f130c73c90471950085e449db5729f9bf8fa1))

## [0.14.2](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.14.1...pipeworks-image-generator-v0.14.2) (2026-03-16)


### Internal Changes

* **api:** remove deprecated prompt/runtime compatibility paths ([#86](https://github.com/pipe-works/pipeworks-image-generator/issues/86)) ([29a7f4d](https://github.com/pipe-works/pipeworks-image-generator/commit/29a7f4d91a5b9f545c51c79301deec71f80dcf58))
* **app:** split API/frontend monoliths and add deprecation signals ([#84](https://github.com/pipe-works/pipeworks-image-generator/issues/84)) ([741fd0f](https://github.com/pipe-works/pipeworks-image-generator/commit/741fd0fa74b7c819a33aa9407b9ded4368b068f9))

## [0.14.1](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.14.0...pipeworks-image-generator-v0.14.1) (2026-03-16)


### Fixes

* **gpu:** secure runtime worker settings and docs ([a057769](https://github.com/pipe-works/pipeworks-image-generator/commit/a057769dee9c133da94cf96be33b712595c07e48))
* **gpu:** secure runtime worker settings and docs ([d19dfd6](https://github.com/pipe-works/pipeworks-image-generator/commit/d19dfd6b0e40b99c687e889b226a9e984afea2fa))

## [0.14.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.13.1...pipeworks-image-generator-v0.14.0) (2026-03-16)


### Features

* **remote-worker:** add config-driven GPU worker routing ([#79](https://github.com/pipe-works/pipeworks-image-generator/issues/79)) ([ba2345c](https://github.com/pipe-works/pipeworks-image-generator/commit/ba2345c180a099bc5faef7b20c5e315fc78f7e1a))
* **remote-worker:** add runtime GPU settings UI and API ([#81](https://github.com/pipe-works/pipeworks-image-generator/issues/81)) ([6ab8ac4](https://github.com/pipe-works/pipeworks-image-generator/commit/6ab8ac4aad73069013aaca993bf2f2b4c56ac37d))

## [0.13.1](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.13.0...pipeworks-image-generator-v0.13.1) (2026-03-16)


### Fixes

* **config:** ignore non-PIPEWORKS dotenv keys ([#77](https://github.com/pipe-works/pipeworks-image-generator/issues/77)) ([c603f8f](https://github.com/pipe-works/pipeworks-image-generator/commit/c603f8f53c755b12012ca52fdc6d2ad303e76760))

## [0.13.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.12.0...pipeworks-image-generator-v0.13.0) (2026-03-16)


### Features

* **runtime:** load snippets from canonical policy APIs ([#75](https://github.com/pipe-works/pipeworks-image-generator/issues/75)) ([8c8543f](https://github.com/pipe-works/pipeworks-image-generator/commit/8c8543fccbd0364be609c42fab1bd7e9ef43f1cd))

## [0.12.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.11.0...pipeworks-image-generator-v0.12.0) (2026-03-09)


### Features

* **prompt-snippets:** include policy YAML text blocks in dropdowns ([6098bae](https://github.com/pipe-works/pipeworks-image-generator/commit/6098bae44eeee4e0760d6a001821a485bfa7a322))
* **prompt-snippets:** include policy YAML text blocks in dropdowns ([2a0d1bd](https://github.com/pipe-works/pipeworks-image-generator/commit/2a0d1bd39b48c82c19defd44f4aef5df15cc757d))

## [0.11.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.10.0...pipeworks-image-generator-v0.11.0) (2026-03-09)


### ⚠ BREAKING CHANGES

* **prompt-composer:** frontend generation payload now uses section-schema fields and no longer sends legacy prompt payload keys.

### Features

* **policies:** add goblin canon clothing block expansions ([b8b4512](https://github.com/pipe-works/pipeworks-image-generator/commit/b8b45125bd7e3d63124f5ed182f666c26587b27c))
* **policies:** add goblin canon clothing block expansions ([#70](https://github.com/pipe-works/pipeworks-image-generator/issues/70)) ([7f0e5ce](https://github.com/pipe-works/pipeworks-image-generator/commit/7f0e5ceb16e0e4dbd7e166a994ca4b480e028195))
* **prompt-composer:** migrate to five-section policy schema ([502fb85](https://github.com/pipe-works/pipeworks-image-generator/commit/502fb850e44577a133f364af2ed58c0afe5b833b))
* **prompt-composer:** migrate to five-section policy schema ([e41c106](https://github.com/pipe-works/pipeworks-image-generator/commit/e41c1062cdbdb0b7861d6ea771d8f70417464ef0))
* **runtime:** default to Klein model and prefix server logs ([6fc4006](https://github.com/pipe-works/pipeworks-image-generator/commit/6fc40066148b7e4c5529ec8a357746c5eb3d55af))
* **runtime:** default to Klein model and prefix server logs ([869ee00](https://github.com/pipe-works/pipeworks-image-generator/commit/869ee00be23b7cfed0469d2211f30b4a97b32715))


### Fixes

* **policies:** mirror pipeworks_web policy tree ([#68](https://github.com/pipe-works/pipeworks-image-generator/issues/68)) ([b2972f5](https://github.com/pipe-works/pipeworks-image-generator/commit/b2972f5dc30e3a28ed2f9657daf130ea241fa715))

## [0.10.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.9.0...pipeworks-image-generator-v0.10.0) (2026-03-06)


### Features

* allow adjustable CFG guidance for FLUX.2-klein model ([7d9a44b](https://github.com/pipe-works/pipeworks-image-generator/commit/7d9a44b9ce1ec674ca0fb60aeef4831d92e1760e))
* allow adjustable CFG guidance for FLUX.2-klein model ([def59d8](https://github.com/pipe-works/pipeworks-image-generator/commit/def59d8f15c9af4d644659ed0dab842d57c7ef51))

## [0.9.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.8.0...pipeworks-image-generator-v0.9.0) (2026-03-05)


### Features

* add generation output bulk select and zip controls ([e4248a2](https://github.com/pipe-works/pipeworks-image-generator/commit/e4248a213a40c465f019808f534a1542cee8c725))
* add output bulk select and zip controls ([b7bb303](https://github.com/pipe-works/pipeworks-image-generator/commit/b7bb303505809dbc3887d2489d4a62099977709c))

## [0.8.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.7.0...pipeworks-image-generator-v0.8.0) (2026-03-05)


### Features

* simplify gallery to flat grid with save-selected zip ([a019d72](https://github.com/pipe-works/pipeworks-image-generator/commit/a019d7256b48d2d7f01b547d915fcc5d853f0826))
* simplify gallery to flat grid with save-selected zip ([c30a92c](https://github.com/pipe-works/pipeworks-image-generator/commit/c30a92ca27db13a7ce4e74af2338a95fe690d042))

## [0.7.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.6.0...pipeworks-image-generator-v0.7.0) (2026-03-05)


### Features

* show all images on run expand with paginated navigation ([#59](https://github.com/pipe-works/pipeworks-image-generator/issues/59)) ([c9e78ff](https://github.com/pipe-works/pipeworks-image-generator/commit/c9e78ff014e381ebd234c598e9d499b53f058962))

## [0.6.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.5.1...pipeworks-image-generator-v0.6.0) (2026-03-05)


### Features

* group gallery by generation runs with collapsible view and bulk zip ([b342287](https://github.com/pipe-works/pipeworks-image-generator/commit/b3422876adf200e0f7fc7d1b6bbbf1e5e5ab47f1))
* group gallery by generation runs with collapsible view and bulk zip ([58e2c7f](https://github.com/pipe-works/pipeworks-image-generator/commit/58e2c7fbe98b6d2d6d22cad6b1625c9f0e37865c))

## [0.5.1](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.5.0...pipeworks-image-generator-v0.5.1) (2026-03-04)


### Fixes

* unify zip metadata main section to match prepend/append shape ([#55](https://github.com/pipe-works/pipeworks-image-generator/issues/55)) ([719b4b4](https://github.com/pipe-works/pipeworks-image-generator/commit/719b4b4b693dd2ec23cd4194d93e9e965820f493))

## [0.5.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.4.0...pipeworks-image-generator-v0.5.0) (2026-03-04)


### Features

* enrich zip metadata with full prompt section details ([#53](https://github.com/pipe-works/pipeworks-image-generator/issues/53)) ([0759efb](https://github.com/pipe-works/pipeworks-image-generator/commit/0759efb8ac5c3a7c3784f0826119b8af5ab13d27))

## [0.4.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.3.1...pipeworks-image-generator-v0.4.0) (2026-03-04)


### Features

* add Save Zip download from lightbox modal ([#51](https://github.com/pipe-works/pipeworks-image-generator/issues/51)) ([4576dda](https://github.com/pipe-works/pipeworks-image-generator/commit/4576dda125e8b43d71b3b5d2e0383df1b4c9a0e9))

## [0.3.1](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.3.0...pipeworks-image-generator-v0.3.1) (2026-03-03)


### Fixes

* force zero guidance for flux2 klein ([#47](https://github.com/pipe-works/pipeworks-image-generator/issues/47)) ([0038441](https://github.com/pipe-works/pipeworks-image-generator/commit/0038441ff2b36e4c99b7318d2bd67e234a1a2740))
* hide unused controls for flux2 klein ([#49](https://github.com/pipe-works/pipeworks-image-generator/issues/49)) ([79f48c6](https://github.com/pipe-works/pipeworks-image-generator/commit/79f48c6a2d72c255f066b76accbb3a56b52b2916))
* restore negative prompts and move prompt copy to lightbox ([#50](https://github.com/pipe-works/pipeworks-image-generator/issues/50)) ([77301b8](https://github.com/pipe-works/pipeworks-image-generator/commit/77301b81d75356f5fbede98b69d661d43f0dcb2a))

## [0.3.0](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.11...pipeworks-image-generator-v0.3.0) (2026-03-03)


### Features

* add prompt variants and cooperative batch stop ([#40](https://github.com/pipe-works/pipeworks-image-generator/issues/40)) ([e6a6c7a](https://github.com/pipe-works/pipeworks-image-generator/commit/e6a6c7a6ed7b4357cb5b6f1d99273344d034db9d))


### Fixes

* add release-please manifest config ([#45](https://github.com/pipe-works/pipeworks-image-generator/issues/45)) ([53f8df6](https://github.com/pipe-works/pipeworks-image-generator/commit/53f8df6690a105ad911919b21b093908f4506608))
* bump pre-1.0 features as minor releases ([#42](https://github.com/pipe-works/pipeworks-image-generator/issues/42)) ([147521b](https://github.com/pipe-works/pipeworks-image-generator/commit/147521b58805b336e5d98c74db453a55f7c13078))

## [0.2.11](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.10...pipeworks-image-generator-v0.2.11) (2026-03-03)


### Features

* add flux2 klein model support ([942a96d](https://github.com/pipe-works/pipeworks-image-generator/commit/942a96d768124625a9dd37247ec55f202c02f342))
* add flux2 klein model support ([aaffb82](https://github.com/pipe-works/pipeworks-image-generator/commit/aaffb823b12bcf003ef906fc9a4ffd4fc065d292))


### Bug Fixes

* keep flux runtime dependency installable ([93dab72](https://github.com/pipe-works/pipeworks-image-generator/commit/93dab72ba212c473f0d9bca09eb709ec5224269d))

## [0.2.10](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.9...pipeworks-image-generator-v0.2.10) (2026-03-02)


### Bug Fixes

* remove prompt composer paste buttons ([898cacf](https://github.com/pipe-works/pipeworks-image-generator/commit/898cacf4bf16927ed58b648bc254c67a7b822b9f))
* remove prompt composer paste buttons ([8170ebe](https://github.com/pipe-works/pipeworks-image-generator/commit/8170ebe341c51f4e0058ab8d2d269d2ca53c24e3))

## [0.2.9](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.8...pipeworks-image-generator-v0.2.9) (2026-03-02)


### Bug Fixes

* add fallback for prompt paste action ([2dfc7f0](https://github.com/pipe-works/pipeworks-image-generator/commit/2dfc7f0a7282313dcf0fcec86502bf8208958a21))
* add fallback for prompt paste action ([b18d5e8](https://github.com/pipe-works/pipeworks-image-generator/commit/b18d5e857a4d11dbe02de3a6acf47ace0bdf1a94))

## [0.2.8](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.7...pipeworks-image-generator-v0.2.8) (2026-03-02)


### Features

* add prompt composer clipboard actions ([054911f](https://github.com/pipe-works/pipeworks-image-generator/commit/054911fdd046b1c83dea04f7572cc24c81e2f980))
* add prompt composer clipboard actions ([392f408](https://github.com/pipe-works/pipeworks-image-generator/commit/392f408349b35573e6d028115ac8b61f0b7a8066))
* expand batch size and add local no-cache mode ([#29](https://github.com/pipe-works/pipeworks-image-generator/issues/29)) ([907f0c6](https://github.com/pipe-works/pipeworks-image-generator/commit/907f0c6267358e1e90e55eb23923cecd70c3e073))
* make generation sections collapsible ([faebdab](https://github.com/pipe-works/pipeworks-image-generator/commit/faebdabec600ed43b90a377266575b56e8dda4a9))
* make generation sections collapsible ([6ea3bd6](https://github.com/pipe-works/pipeworks-image-generator/commit/6ea3bd69b647aebccb23dca2febc8f8309f69d1d))
* split prompt libraries by section ([#31](https://github.com/pipe-works/pipeworks-image-generator/issues/31)) ([9121c96](https://github.com/pipe-works/pipeworks-image-generator/commit/9121c96c6e8ae607749821abd2be7937f839c79a))

## [0.2.7](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.6...pipeworks-image-generator-v0.2.7) (2026-03-02)


### Bug Fixes

* make prompt composer sections optional ([#27](https://github.com/pipe-works/pipeworks-image-generator/issues/27)) ([8920d90](https://github.com/pipe-works/pipeworks-image-generator/commit/8920d90b8becc530825eba7c3f481a3ad957a04d))

## [0.2.6](https://github.com/pipe-works/pipeworks-image-generator/compare/pipeworks-image-generator-v0.2.5...pipeworks-image-generator-v0.2.6) (2026-03-02)


### Bug Fixes

* align z-image prompt token limit with pipeline ([#25](https://github.com/pipe-works/pipeworks-image-generator/issues/25)) ([497eab1](https://github.com/pipe-works/pipeworks-image-generator/commit/497eab109a95b396ad5dc4056b8c96c87d99a09b))

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

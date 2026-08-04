# v0.1 release checklist

## Automated gates

- [x] Ruff lint and formatting checks pass.
- [x] mypy strict type checking passes.
- [x] Tests pass on CPython 3.11, 3.12, 3.13, and 3.14 in CI.
- [x] Performance suite covers scalar/batch trajectories, Ulam construction,
  FFT evolution, and dense eigenanalysis with explicit thresholds.
- [x] All five tutorial programs and every standalone example execute in CI.
- [x] Wheel and sdist build, metadata validation, and clean-environment installs
  are defined in CI.

## Release artifacts

- [x] Public API reference and numerical standards are present.
- [x] Quickstart, classical, Ulam, quantum, and spectral tutorials are executable.
- [x] `CHANGELOG.md`, BSD-3-Clause `LICENSE`, and `CITATION.cff` are present.
- [x] Package metadata declares supported Python versions and runtime dependencies.
- [x] TestPyPI trusted-publishing workflow is configured with post-upload install smoke test.

## Maintainer actions for a release candidate

- [ ] Replace the development version with the intended unique release version.
- [ ] Confirm the `testpypi` GitHub environment trusts this repository and run
  the **Publish to TestPyPI** workflow.
- [ ] Inspect the rendered TestPyPI project page and its wheel/sdist files.
- [ ] Tag the reviewed commit and publish to production PyPI through a separately
  protected environment.

The unchecked items require repository credentials or a maintainer's release
decision; they are intentionally not performed by ordinary CI or local tests.

## Known v0.1 constraints

- Built-in experiments use a fixed name-to-callable adapter set; there is no
  public plugin registry.
- NPZ/JSON is the only persistence backend; Zarr and concurrent writers are future work.
- Dense quantum eigensystems are limited to deliberately small dimensions.
- Plotting is optional and outside the stable numerical API.
- Benchmarks are CPU wall-clock smoke thresholds, not hardware-normalized scores.

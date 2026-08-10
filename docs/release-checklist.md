# v0.1 release checklist

## Automated gates

- [x] Ruff lint and formatting checks pass, including notebook cells.
- [x] mypy strict type checking passes over `src`, `tests`, and `benchmarks`.
- [x] Tests pass on CPython 3.11, 3.12, 3.13, and 3.14, and on macOS and Windows
  for the newest interpreter.
- [x] Tests pass against the declared dependency floor (oldest supported NumPy
  and SciPy), so the lower bounds in `pyproject.toml` are verified rather than
  assumed.
- [x] Warnings are errors in the test suite, so a diagnostic nobody asserts on
  fails the build.
- [x] Coverage is measured with branch coverage and gated.
- [x] Performance suite covers all six documented benchmark cases with
  peak-allocation and scaling gates that block, and timing reported per runner
  class.
- [x] All tutorial programs and every standalone example execute in CI.
- [x] Every Python code block in `README.md` and `docs/` is executed in CI.
- [x] `notebooks/known_results.ipynb` executes end to end on the default branch,
  and **checks** rather than merely prints its results: each reproduced value
  carries a tolerance and the final cell raises `AssertionError` when one drifts
  outside it. Without that the job would fail on exceptions alone and stay green
  with every number wrong.
- [x] Wheel and sdist build, metadata validation, and clean-environment installs
  are defined in CI.
- [x] The package version has a single source (`src/chaos_numerics/_version.py`,
  read by Hatchling); a test pins `CITATION.cff` to it.

## Release artifacts

- [x] Public API reference and numerical standards are present, and the
  per-module public-surface listings in `docs/design/public-api.md` are pinned to
  each package's `__all__` by a test, so a new export cannot ship undocumented.
- [x] Quickstart, classical, Ulam, quantum, and spectral tutorials are executable.
- [x] `CHANGELOG.md`, BSD-3-Clause `LICENSE`, and `CITATION.cff` are present.
- [x] Package metadata declares supported Python versions and runtime dependencies.
- [x] TestPyPI trusted-publishing workflow is configured with post-upload install smoke test.

## Maintainer actions for a release candidate

- [ ] Replace the development version with the intended unique release version in
  `src/chaos_numerics/_version.py` and `CITATION.cff`, and update the
  `date-released` field. The version test fails if the two disagree.
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

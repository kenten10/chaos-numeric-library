# Changelog

All notable changes to this project will be documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `loschmidt_echo` and `otoc` are the two standard finite-dimensional stand-ins for
  the classical Lyapunov instability, and the library shipped neither. Both take
  whole models rather than a perturbation size, because which two operators are
  being compared is what a reader of the result needs to know.
  `loschmidt_echo(reference, perturbed, state, steps=...)` returns
  `M(t) = |<psi| U_b^(-t) U_a^t |psi>|**2` for `t = 0 .. steps`, batched over
  initial states. Measured on `KickedRotor` at `N = 256` with the two models
  differing only in `kick_strength` by `1e-2`, `M(40)` falls from `0.99` at
  `K = 1` to `1.3e-3` at `K = 5`, nearly three orders of magnitude across the KAM
  threshold on an unchanged perturbation. `otoc(model, steps=...)` returns
  `C(t) = <|[A(t), B]|**2>` with `A(t) = U^(-t) A U^t`, defaulting to the Weyl
  translation pair that the new `weyl_translations` builds; because both operators
  are bounded unitaries, `C` saturates near `2` (measured time-average `1.9946`
  over `100 <= t <= 200` at `N = 128`, `K = 10`) rather than growing without
  limit. Neither result is clipped into a nominal range: a clipped diagnostic
  hides exactly the numerical drift it should be reporting.
- `docs/design/quantum-instability-diagnostics.md`, the design document for the new
  echo and correlator: the Weyl operator convention, the `C(0) = 4 sin^2(pi/N)`
  closed form that pins it, why the fitted rate is not `2 lambda`, and why no
  `lyapunov_from_otoc` helper is offered. The README design-document index now
  lists all nine design documents instead of three, and the module tree in
  `public-api.md` names every module in the package.
- `rmt_reference` gains the `"gse"` ensemble (`beta = 4`) with `"cse"` as its
  alias, completing the three Wigner-Dyson classes on every statistic it
  supports. It is deliberately reference-only: `beta = 4` needs an antiunitary
  symmetry squaring to `-1`, so half-integer spin, and every model in this library
  is spinless, which is stated at each entry point rather than left for a user to
  discover from a curve that will not fit. The curves describe the *distinct*
  levels of the Kramers-degenerate spectrum. `mean_gap_ratio_reference("gse")`
  returns `0.6744`, measured here as `0.674408 +- 0.000064` over `9.8e6` ratios,
  and the surmise value is `0.6762`. `K_GSE(tau)` has a real logarithmic pole at
  the Heisenberg time, so a grid landing exactly on `tau = 1` raises rather than
  returning an infinity, and `dimension` is rejected for `"gse"` on every
  statistic because the only finite-`N` kernel implemented is the circular unitary
  one.
- `run_sweep` reaches quantum models. `kicked_rotor`, `cylinder_kicked_rotor`,
  `quantum_cat_map`, and `quantum_baker_map` join the registry alongside the
  analyses `eigenphases`, `eigenstates`, `spectral_form_factor`, and
  `number_variance`, and `SpectralCurve` gained a persistence branch. This closes a
  gap between the library's headline feature and its headline figure: the README said
  a ramp-and-plateau plot needs an ensemble average over Bloch phases, and that
  ensemble is a parameter sweep, yet the sweep could not run it. Measured over twelve
  phases at `N = 128`, the sweep's ensemble mean gives a ramp slope of 1.08 and a
  plateau of 0.98, four times closer to the CUE reference than any single member.
  `boundary_phases` accepts a scalar, a two-element list, or the mapping that
  `BoundaryPhases.to_dict()` produces, so a persisted parameter set feeds straight
  back into the next sweep.
- `spectral_rigidity` computes the Dyson-Mehta `Delta_3(L)`, the standard companion
  to the number variance in level-statistics papers. The in-window least squares is
  solved in closed form and agrees with a naive quadrature to `1.5e-15`; the
  reference curve follows from the number variance through Mehta's exact relation, so
  both statistics share one cluster-function implementation. Verified against the
  exact Poisson `L/15` (bit-identical), against slopes of `1/pi**2` and
  `1/(2*pi**2)` for GOE and GUE (ratios 0.9994 and 1.0000), and against a
  self-derived closed form for an equally spaced spectrum.
- `rmt_reference` gained `"spectral_rigidity"` and `"gap_ratio_distribution"`. The
  ratio density is folded onto `[0, 1]` by default, matching what
  `adjacent_gap_ratios` returns; `folded=False` gives the density on `(0, inf)`. Its
  first moment reproduces `mean_gap_ratio_reference(..., surmise=True)`, which pins
  the normalization and the shape together.
- `wigner_distribution` and `WignerResult`. Marginals reproduce the position and
  momentum probabilities to `3e-16`, the distribution sums to one, and unlike the
  Husimi distribution it goes negative, which is the point: a two-lobe cat state
  reaches `-3.5e-2` against a peak of `+3.8e-2` while its Husimi minimum stays at
  `+3e-12`.
- `momentum_numbers`, `to_momentum_basis`, and `to_position_basis` on both rotors.
  The twisted transform was private, and its own docstring warned the convention is
  easy to get wrong, so a momentum-space figure was not reachable through the public
  API. Substituting a bare `numpy.fft` call is wrong by a measurable amount: omitting
  `norm="ortho"` scales every bin by 64 at `N = 64`, and omitting the momentum twist
  moves the distribution by a total variation distance of 0.5 at `beta = 0.5`.
- The reproduction notebook checks itself. Every reproduced value carries a
  tolerance and the summary cell raises when one drifts outside it; previously the
  notebook only printed its numbers, so the CI job that executes it would have
  stayed green with every value wrong. `Sigma^2(L=10)` carries a lower bound as
  well as an upper one, because RMT failing at long range is the physics the
  section demonstrates.
- `CylinderKickedRotor.apply_fft`, so `evolve(..., method="fft")` works on the
  model that dynamical localization needs.
- Public type aliases `CatMatrix`, `EvolutionMethod`, `SymmetrySector` (`quantum`),
  `AnyArray`, `ArrayPayload` (`core`), and `Side`, `OperatorLike` (`operators`),
  completing the rule that a type appearing in a public signature is public.
- `notebooks/known_results.ipynb` reproduces fourteen standard results of
  classical and quantum chaos against their analytic values, and is executed in
  CI on the default branch. It checks rather than merely prints them: every
  reproduced value carries a tolerance and the final cell raises `AssertionError`
  when one drifts outside it, because a notebook job fails on exceptions alone and
  would otherwise stay green with every number wrong.
- `CylinderKickedRotor`: the momentum-basis kicked rotor with `effective_hbar`
  independent of the basis size. The torus-quantized `KickedRotor` fixes
  `hbar_eff = 2*pi/N`, which puts the localization length above `N` and makes
  dynamical localization unreachable; this model is what that result needs.
- `LogisticMap`, and `poincare_section` with its `PoincareSection` container.
- `desymmetrize` projects a symmetry-resolved eigensystem onto one sector.
  Comparing an undesymmetrized spectrum with a random-matrix ensemble is a
  standard error: on the Saraceno baker map the raw mean gap ratio is 0.42
  against a COE reference of 0.5307, while at a generic even dimension the two
  parity sectors land on the reference within one or two standard errors
  (`N = 700`: 0.5211 and 0.5324, standard error 0.0134). Desymmetrizing is
  necessary but **not** sufficient: at `N = 2**k` the classical binary shift
  resonates with the dimension and the sectors stay 6 to 8 standard errors low
  (`N = 1024`: 0.4389 and 0.4559), barely moving off the raw 0.42.
- `rmt_reference("spacing_distribution", ...)` returns the Wigner surmise, and
  `mean_gap_ratio_reference` returns the mean adjacent gap ratio, so that the most
  common comparison in the field no longer requires writing the formulas by hand.
- `core.SerializableResult`: every public result container now splits into arrays
  and JSON metadata the same way, so the storage layer no longer works for only
  the four `core` types.
- Batch axes for `mean_square_displacement` and `autocorrelation`, with
  ensemble-averaged curves and a batch-derived uncertainty.
- Public type aliases: `Observable` and `Method` from `classical`, `Ensemble`,
  `Statistic`, and `Window` from `spectral`, and `DEFAULT_DENSE_LIMIT` and
  `KickedRotor.symmetry_operators` from `quantum`.
- Coverage measurement with a threshold, macOS and Windows test jobs, a
  dependency-floor job, and a notebook execution job in CI.
- `KickedRotor.symmetry_operators` publishes the parity operator for the boundary
  phases that admit it, so the default spectrum can be desymmetrized before it is
  compared with a random-matrix ensemble.
- `rmt_reference` accepts `"coe"` as an alias for `"goe"`, and unsupported
  statistic or ensemble names now list the valid values.
- `scipy` version, BLAS/LAPACK identity, thread limits, and a Git dirty flag are
  recorded in experiment environment metadata, as the numerical standards require.
- Coverage measurement with a threshold, `macos-latest` and `windows-latest` test
  jobs, and a notebook execution job in CI.
- Reproducible experiment execution, parameter sweeps, resume, and NPZ/JSON storage.
- Classical standard, cat, and baker maps with trajectory, Lyapunov, transport,
  and periodic-orbit analyses.
- Ulam transfer operators, partitions, sparse eigensolvers, stationary density,
  and spectral-gap analysis.
- Kicked rotor, quantum cat and baker maps, evolution, eigensystems, Husimi data,
  and localization diagnostics.
- Spectral unfolding, spacing, adjacent-gap, form-factor, number-variance, and
  Poisson/RMT reference statistics.
- Typed public APIs, numerical standards, executable tutorials, and performance
  regression benchmarks.

### Changed (breaking)

- The reported `uncertainty` of `local_diffusion_exponent` and `number_variance`
  changes, because both were far too small. A least-squares standard error assumes
  independent residuals, and neither a time-origin-averaged MSD nor a set of
  overlapping spectral windows provides them.
  - `local_diffusion_exponent` reported about `1/30` of the true estimator spread
    (measured `0.005` against a true `0.142` on a unit random walk, with `93%` of
    realizations further than two reported errors from the truth). It now accepts a
    two-dimensional ensemble of MSD curves and reports the standard error of the
    mean over per-curve fits, calibrated to `0.95`-`1.19` of the true spread with
    `3.0%`-`11.5%` beyond two errors. **With a single curve it returns `None` and
    warns**, because one curve carries no information about the spread of its own
    slope. `mean_square_displacement` grows a `per_trajectory` keyword to supply
    the ensemble.
  - `number_variance` reported `2.2`-`3.7` times too little, and its error shrank
    without bound as `samples` grew even though the value did not, so a fixed
    spectrum could be made to look many sigma away from RMT. The error is now based
    on the effective number of independent windows, `min(samples, N / L)`, via a
    batch-mean estimator, calibrated to `0.70`-`0.87` of the true spread. `bootstrap`
    became a block bootstrap, so its numbers change for the same seed. A window
    longer than half the spectrum warns.
  - Both record what their error means in `metadata.parameters["error_semantics"]`.
    `spectral_form_factor` records the same key: its `bootstrap` is a resampling
    diagnostic and overestimates the realization spread by `1.5`-`3.7` times, which
    is now stated rather than implied.
- `mean_square_displacement` and `autocorrelation` return `uncertainty=None` and
  emit `NumericalWarning` for a single trajectory. The MSD previously reported the
  scatter over the time origins of one trajectory, which is the same failure as the
  one above one level down: the origins of a time-origin-averaged curve are not
  independent draws. Measured against the exact `<dx**2> = t` of a unit random
  walk, it understated the true spread by factors of `1.0`, `2.6`, `8.8` and `20.9`
  at lags `1`, `10`, `100` and `500`, leaving `4.8%`, `46%`, `82%` and `90%` of
  realizations more than two of its sigma from the truth where `5%` was expected.
  Pass an ensemble of shape `(*batch, time[, dim])` for the calibrated batch
  standard error, which measured `0.98`-`1.13` with `4.5%`-`9.5%` beyond two
  errors over the same lags. The retired estimator is pinned by a regression test
  that recomputes it and asserts that it still fails, so it cannot come back
  quietly. Both functions now agree with `local_diffusion_exponent` on one
  convention, and `metadata.parameters["uncertainty_estimator"]` is removed in
  favour of the `error_semantics` key the rest of the library already used.
- `KickedRotor.symmetry_operators` publishes the reflection for all four twists in
  `{0, 1/2}`, not two of them. `(1/2, 0)` needs the reflection about the other
  centre and `(0, 1/2)` needs a sign-dressed one; both commute exactly (defect
  `7e-14`) and both were previously reported as symmetry-free, so a user comparing
  such a spectrum with a random-matrix ensemble was silently comparing a
  two-sector mixture. Desymmetrizing recovers COE: `(1/2, 0)` goes from `0.42` to
  `0.541 / 0.543` and `(0, 1/2)` from `0.42` to `0.509 / 0.532`.
- `evolve` returns a `QuantumEvolution` container instead of a bare array. The
  array's *shape* used to depend on `return_history`: `(*batch, N)` for one flag
  value and `(*batch, T, N)` for the other, so generic code had to branch on an
  argument to know what it was holding, and none of the reproducibility metadata
  survived the return at all. `final_state` is now always `(*batch, N)`, `history`
  is `None` unless it was asked for, and `metadata.parameters` records the route
  that actually ran (`method`, with `"auto"` already resolved, alongside
  `requested_method`) plus the norm audit as `norm_audits` and
  `maximum_norm_drift`. Callers append `.final_state`, or read `.history` where
  they used to index the time axis. The container follows the same contract as
  every other result type: read-only owned arrays, value-based `__eq__`, a
  `__reduce__` that survives `pickle` and `deepcopy`, and the `SerializableResult`
  split, whose `result_type` is `quantum_evolution`.
- The diagnostic code `stationary-density-not-converged` now carries
  `category="convergence"` instead of falling back to `"numerical"`, which is
  observable to anyone filtering `metadata.warnings` by category.

- `UlamMatrix` is no longer a `scipy.sparse.csr_matrix` subclass. It is an
  immutable container composed of a canonical CSR `matrix`, per-column
  `escape_probabilities`, and build `metadata`. Subclassing exported seventy-odd
  SciPy methods as v0.1 contract and left the values mutable, so escape
  diagnostics could drift out of step with the matrix they described. The
  container forwards `shape`, `dtype`, `nnz`, `matvec`, `toarray()`, and `@`,
  which is enough to satisfy `core.LinearOperatorLike`; every other SciPy
  operation goes through `.matrix`. The stored arrays are read-only, the
  constructor enforces column sums no greater than one, and diagnostics do not
  propagate through an operation whose escaped mass is not a function of its
  operands'.
- The `boundary_phase` constructor argument of `KickedRotor`, `QuantumCatMap`, and
  `QuantumBakerMap` is now `boundary_phases`, matching the attribute and every
  other function that takes the same concept. Positional use is unaffected.
- `dense_limit` defaults to `DEFAULT_DENSE_LIMIT`, raised from 256 to 512. At
  `N = 512` a dense reference costs about 0.5 s and 4 MiB, which is affordable
  interactively; the old default made every spectral-statistics user hit the guard
  once.
- `spacing_distribution(density=True)` normalizes by the total sample count rather
  than the in-range count, so a restricted `value_range` no longer rescales the
  histogram to integrate to one. See below.
- `mean_square_displacement` and `autocorrelation` select an `O(n log n)` FFT
  algorithm by default. `method="direct"` keeps the reference algorithm, which is
  what the numerical standards specify, and the two agree to `1e-13`.
- The declared SciPy floor is `1.11.4` instead of `1.17`. A CI job runs the suite
  against the floor, because `scipy>=1.17` left Python 3.11 users exactly one
  usable minor version (SciPy 1.18 requires Python 3.12) and nothing verified it.

### Changed

- `build_ulam(open_system=True)` is about 41 times faster (3.69 s to 0.089 s at
  32x32 cells with 256 samples each), which brings it level with the closed-system
  path. The escape test is vectorized for the built-in rectangular partitions and
  still falls back to one `locate` call per point for a user-defined `Partition`.
  Output is bit-identical, verified by SHA-256 over the CSR arrays and escape
  probabilities across eight configurations.
- `SCHEMA_VERSION` has one definition instead of three. The on-disk value is
  unchanged.
- `husimi_distribution` caches the coherent-state grid across calls within a bounded
  budget, and the batch path is documented: twenty states in one call cost what one
  costs, a 19x saving at `N = 1024` on a `64 x 64` grid, and a repeat call with the
  same grid returns in 2 ms instead of 547 ms. The notebook uses the batch path, which
  is most of why it now runs in three minutes instead of four.
- The performance suite implements the documented benchmark protocol: smoke and
  standard sizes, at least ten repetitions with short cases aggregated to 100 ms,
  median with MAD and IQR, the dual regression gate, peak-allocation gates that
  reject dense materialization in sparse and matrix-free paths, and scaling ratio
  gates. Timing blocks only on a pinned runner and only within one runner class;
  allocation and scaling gates block everywhere. The missing
  `sparse_leading_eigenpairs` case was added.
- `evolve` preallocates history instead of stacking a list, halving peak memory,
  and validates once before the loop rather than on every step. `KickedRotor`
  memoizes its dense reference, making `apply_dense` about 400 times faster at
  `N = 512`. Eigenpair residuals are computed in one batched application.

### Fixed

- The `desymmetrize` docstring and this changelog claimed the baker map's parity
  sectors reach `0.50` and `0.54` at `N = 1024`. They do not: measured `0.439` and
  `0.456`, six to eight standard errors below COE. The classical baker map is the
  binary shift, so `N = 2**k` resonates with it and an arithmetic structure survives
  the parity split; at a generic even dimension both sectors do land on COE
  (`0.521 / 0.532` at `N = 700`). Desymmetrization is necessary but not sufficient.
  The docstrings, the notebook's section 9, and both spectral examples said mutually
  contradictory things about this and now all state the dimension dependence.
- The `CylinderKickedRotor` docstring combined three statements that cannot all
  hold: an exponential saturated profile, `<n^2>_sat = 2 l^2`, and `l ~ D/2`.
  Following that recipe overestimates `D` by about two, because `l ~ D/2` describes
  the eigenfunction localization length while the saturated profile decays over
  roughly twice that. Measured at `K = 8`: `D = 39.2`, eigenfunction length `24.3`,
  profile length `30.4`, `<n^2>_sat = 1742 ~ D^2` rather than `D^2 / 2`. The
  notebook's section 10 attributed the resulting factor to finite `hbar`.
- `validate_protocol` promised in its docstring that every public entry point used
  it, while `iterate`, `lyapunov_spectrum`, `largest_lyapunov_exponent`,
  `find_periodic_orbits`, and `build_ulam` still leaked a bare `AttributeError`.
- `deepcopy` silently dropped the read-only flag on `PreparedEigenphases`,
  `UnfoldedSpectrum`, `SpacingDistributionResult`, `SpectralCurve`, `HusimiResult`,
  and `PeriodicOrbitResult`, and equality after a round trip failed for five of
  them.
- `README.md` opened with `pip install chaos-numerics`, which returns 404 because
  the project is not published yet, and described the notebook as plugging in models
  the library does not ship after both of them had been added to it. Its Lyapunov
  convergence figure was quoted without the kick strength it was measured at, and
  was three to thirty times optimistic for the sweep in the same section.
- Overflow in the transport accumulators raised a generic non-finite
  `ValidationError` after leaking a bare `RuntimeWarning`; it now raises
  `NumericalError` naming the magnitude, the limit, and the remedy, matching the
  Lyapunov policy.
- `_protocol_members` excluded CPython's protocol bookkeeping attributes by name,
  and the list had already fallen behind three interpreter versions, which would
  have degraded the error message it exists to produce.
- The timing regression gate had a 5 ms absolute floor that silenced the
  regressions it was meant to survive: at the recorded smoke medians it allowed
  `quantum_fft_evolve` to slow down 4.8 times and `classical_batch_iterate` 3.5
  times without failing. Scaling ratios are now measured by interleaving the two
  sizes, because timing them in sequence let one load spike land entirely on one
  half and report a ratio of 3.5 where the true value was 2.0.
- The `test-pypi` workflow installed `scipy>=1.17` while the package declares
  `1.11.4`, so it never exercised the floor.
- `spacing_distribution` values in the polynomial-unfolding test were compared with
  `== 1.0`; the last spacing is exactly one only 84% of the time and the final bit
  depends on the BLAS.
- Six result containers shipped `convergence_history` in `array_payload()` without
  describing it in `metadata_payload()`. The storage layer matches the two, so
  persisting such a container failed with a `KeyError` naming neither the container
  nor the array. Latent until this release added a persistence path for
  `SpectralCurve`, which a sweep now writes. A single test builds every container with
  a convergence history and asserts the two sides agree.
- Two tests asserted bit-exact equality on complex results, which does not hold at
  the declared dependency floor: on NumPy 1.26.4 a complex128 elementwise multiply
  is not reproducible call to call (eight invocations with the same input spread over
  `2e-17`). They now compare at rounding level, and the reproducibility section of
  the numerical standards records what the floor does and does not support. Seeded
  reproducibility itself is unaffected and is verified at the floor.
- `mean_square_displacement(unwrap=True)` reconstructed displacements with a
  minimum-image rule that silently aliased once a step moved half a period or
  more, under-reporting standard-map momentum diffusion by up to a factor of 34.
  Ambiguous input is now rejected with a message naming the coordinate, the
  observed displacement, and the remedy. For the unit-torus standard map the
  boundary is `K = pi`.
- `lyapunov_spectrum` returned a wrong trailing exponent with no warning when
  `reorthogonalization_interval` let the basis spread beyond the `float64`
  resolution between QR refreshes (on the cat map an interval of 40 reported
  `+0.81` for an exponent whose exact value is `-0.9624`, while the orthogonality
  defect stayed at `1e-16`). The largest QR condition number is now recorded,
  exceeding the resolution warns, and exceeding its square raises.
- A basis that overflowed between QR refreshes produced `inf`/`nan` diagonals that
  passed the singularity guard, because `nan <= tiny` is false, and surfaced far
  away as an unrelated `ValidationError`. Tangent norms are also computed with
  scaling so the usable interval range roughly doubles.
- Periodic coordinates could land on the excluded upper endpoint, because
  `np.mod(-1e-17, 1.0)` rounds to exactly `1.0`. `iterate`, `find_periodic_orbits`,
  the three classical maps, `Partition.locate`, `prepare_eigenphases`, and
  eigenphase wrapping now reduce through a shared half-open helper.
- `UlamMatrix` did not accept the `dtype` keyword that SciPy passes internally, so
  `copy`, `astype`, scalar multiplication, negation, `conj`, and `power` all raised
  `TypeError`, and the operations that did work reset the escape diagnostics to zero
  without saying so, reporting an open system as closed. Resolved by the move to
  composition above. Pickling and `deepcopy` work.
- Result containers and `ExperimentMetadata` could not be pickled or deep-copied
  because frozen mappings are not picklable, which blocked any future parallel
  sweep.
- `number_variance` miscounted by one level when `samples` was commensurate with
  the level count, because window origins coincided with level positions. Origins
  now sit on a half-shifted grid, and the estimator is verified against the closed
  form for an equally spaced spectrum.
- `spacing_distribution(density=True)` normalized by the in-range count, so a
  restricted `value_range` was rescaled to integrate to one and could not be
  compared with a Wigner surmise. It now records how many spacings fell outside
  the range.
- Tangent propagation emitted a bare `RuntimeWarning: overflow encountered in
  matmul` on NumPy 1.x before the actionable `NumericalError` that names the
  remedy, and stayed quiet on NumPy 2.x. The reported failure is now the same on
  every supported version.
- `ValidationError` for a protocol mismatch listed the missing members only on
  Python 3.12 and newer, because it read the private `__protocol_attrs__`. The
  members are now derived from the protocol itself.
- Public entry points raised bare `AttributeError` for objects that do not
  implement the expected protocol. `ValidationError` now names the protocol and
  the missing members, and additionally subclasses `ValueError` so that existing
  `except ValueError` handlers keep working (`NumericalError` subclasses
  `ArithmeticError`).
- `[tool.pytest]` is not a table pytest reads, so `-ra`, `--strict-config`,
  `--strict-markers`, and `testpaths` had no effect. Renamed to
  `[tool.pytest.ini_options]` and added `filterwarnings = ["error"]`.
- Markdown code examples in `README.md`, `docs/design/`, and `docs/product/` were
  never executed and had drifted: several referenced undefined names or methods
  that do not exist (`KickedRotor.eigensystem`, `AnalysisResult.warnings`,
  `evolve(...).norm_error`). Every block is now executed by
  `tests/test_documentation.py` and in CI.

## [0.1.0.dev0] - 2026-08-05

### Added

- First release-candidate baseline for the v0.1 public API.

[Unreleased]: https://pypi.org/project/chaos-numerics/
[0.1.0.dev0]: https://pypi.org/project/chaos-numerics/0.1.0.dev0/

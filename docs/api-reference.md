# API reference

This page lists the supported v0.1 public import surface. Signatures and detailed
validation rules live in the referenced objects' docstrings; names not exported
through a package `__all__` are implementation details.

## Top level: `chaos_numerics`

- Result types: `Trajectory`, `Spectrum`, `EigenstateResult`, `AnalysisResult`
- Classical models: `StandardMap`, `CatMap`, `BakerMap`, `LogisticMap`
- Quantum models: `KickedRotor`, `CylinderKickedRotor`, `QuantumCatMap`,
  `QuantumBakerMap`
- Experiments: `Experiment`, `run_experiment`, `run_sweep`
- Package version: `__version__`

## `chaos_numerics.core`

- Protocols: `ClassicalMap`, `Flow`, `QuantumMap`, `LinearOperatorLike`, `Partition`,
  `SerializableResult`
- Results and metadata: `Trajectory`, `Spectrum`, `EigenstateResult`,
  `AnalysisResult`, `ExperimentMetadata`, `ConvergenceInfo`, `Diagnostic`
- Error hierarchy: `ChaosNumericsError`, `ValidationError`, `NumericalError`,
  `ConvergenceError`
- Warning hierarchy: `ChaosNumericsWarning`, `NumericalWarning`,
  `ConvergenceWarning`, `ReproducibilityWarning`
- Array dtype aliases: `ArrayLike`, `FloatArray`, `ComplexArray`, `BoolArray`,
  `IndexArray`, `OperatorArray`
- Semantic state aliases: `ClassicalState`, `ClassicalBatch`, `QuantumState`,
  `QuantumBatch`
- Shape aliases: `Shape`, `StateShape`, `BatchShape`, `OperatorShape`
- Persistence aliases: `AnyArray`, `ArrayPayload`

### `SerializableResult`

A structural, `runtime_checkable` protocol for a result container that can be split
into numerical arrays and JSON metadata. The storage layer writes arrays to NPZ and
everything else to JSON, so a container that implements this protocol can be
persisted and reconstructed without the storage layer knowing which domain produced
it. Nothing is inherited; three members make a type conform.

| Member | Kind | Contract |
| --- | --- | --- |
| `metadata` | property | `ExperimentMetadata` for the computation that produced the result. |
| `array_payload()` | method | Returns `ArrayPayload` (`dict[str, AnyArray]`): independent, writable copies of every stored array, plus `"convergence_history"` when `metadata.convergence.history` exists. |
| `metadata_payload()` | method | Returns a JSON-compatible descriptor. Array **contents never appear**; each entry is reduced to `{"shape": [...], "dtype": "..."}`. Keys are `schema_version` (currently `1`), `result_type`, `metadata`, and `arrays`, plus per-type extras such as `name` on `AnalysisResult`. |

Both builders delegate to a shared implementation, so `schema_version` has a single
definition for every domain.

Fourteen public types implement the protocol in v0.1, and `result_type` is unique
per type. Implementing the protocol makes a container writable; reading one back
is separate. `experiment.load_result` reloads persisted *runs*, so it has a restore
branch only for the types a registered analysis can return -- `trajectory`,
`spectrum`, `eigenstate`, `analysis`, and `spectral_curve` -- and rejects any other
`result_type` rather than guessing. See `docs/design/experiment-persistence.md`.

| Type | Module | `result_type` |
| --- | --- | --- |
| `Trajectory` | `core` | `trajectory` |
| `Spectrum` | `core` | `spectrum` |
| `EigenstateResult` | `core` | `eigenstate` |
| `AnalysisResult` | `core` | `analysis` |
| `UlamMatrix` | `operators` | `ulam_matrix` |
| `PoincareSection` | `classical` | `poincare_section` |
| `PeriodicOrbitResult` | `classical` | `periodic_orbits` |
| `HusimiResult` | `quantum` | `husimi` |
| `WignerResult` | `quantum` | `wigner` |
| `QuantumEvolution` | `quantum` | `quantum_evolution` |
| `PreparedEigenphases` | `spectral` | `prepared_eigenphases` |
| `UnfoldedSpectrum` | `spectral` | `unfolded_spectrum` |
| `SpacingDistributionResult` | `spectral` | `spacing_distribution` |
| `SpectralCurve` | `spectral` | `spectral_curve` |

The aliases carry dtype and semantic intent only. NumPy's static typing cannot
express array shapes, so shapes are enforced by runtime validation at the public
boundary and documented per function; `ClassicalState` and `ClassicalBatch` are
the same runtime type and differ only in intent. See `core/types.py`.

## `chaos_numerics.classical`

- Models: `StandardMap`, `CatMap`, `BakerMap`, `LogisticMap`
- Evolution: `iterate`
- Phase-space sections: `poincare_section`, `PoincareSection`
- Stability: `largest_lyapunov_exponent`, `lyapunov_spectrum`
- Transport: `autocorrelation`, `mean_square_displacement`,
  `local_diffusion_exponent`
- Type aliases: `Observable`, `Method`
- Periodic orbits: `find_periodic_orbits`, `PeriodicOrbitResult`

All three transport functions share one uncertainty convention: the reported
`uncertainty` is the standard error of the mean across independent trajectories,
and a **single** trajectory or curve gets `uncertainty=None` together with a
`NumericalWarning` rather than a number derived from correlated samples. Each
records what its error means in `metadata.parameters["error_semantics"]`. Pass an
ensemble of shape `(*batch, time[, dim])`; `mean_square_displacement` also takes
`per_trajectory=True` to keep the curves separate, which is what
`local_diffusion_exponent` needs to report a spread over per-curve fits.

## `chaos_numerics.operators`

- Partitions: `RectangularPartition`, `UniformPartition`
- Ulam: `UlamMatrix`, `build_ulam`

`UlamMatrix` composes a canonical CSR `matrix`, per-column `escape_probabilities`,
and build `metadata`, and exposes `array_payload()` and `metadata_payload()` like
the `core` result types. It forwards `shape`, `dtype`, `nnz`, `matvec`,
`toarray()`, and `@` so that it satisfies `core.LinearOperatorLike`; every other
SciPy operation goes through `.matrix`, and diagnostics do not propagate through
one.
- Analysis: `leading_eigenpairs`, `stationary_density`, `spectral_gap`
- Type aliases: `Side` (`Literal["right", "left"]`), `OperatorLike`

## `chaos_numerics.quantum`

- State conventions: `BoundaryPhases`, `QuantumBasis`, `basis_state`,
  `quantum_state`, `normalize_state`
- Models and adapters: `KickedRotor`, `CylinderKickedRotor`, `QuantumCatMap`,
  `QuantumBakerMap`, `DenseUnitary`
- Evolution and spectra: `evolve`, `QuantumEvolution`, `eigenstates`,
  `eigenphases`, `unitarity_defect`, `desymmetrize`, `DEFAULT_DENSE_LIMIT`
- Instability diagnostics: `loschmidt_echo`, `otoc`, `weyl_translations`
- Phase space: `coherent_state`, `husimi_distribution`, `HusimiResult`,
  `wigner_distribution`, `WignerResult`, `inverse_participation_ratio`,
  `participation_ratio`, `shannon_entropy`
- Type aliases: `CatMatrix`, `EvolutionMethod`, `SymmetrySector`

`KickedRotor` and `CylinderKickedRotor` both expose `momentum_numbers`,
`to_momentum_basis`, and `to_position_basis`. The method names say which basis they
return, so the same call reads correctly on both classes even though the torus model
stores states in the position basis and the cylinder model in the momentum basis. The
transform carries the boundary twists, which a bare `numpy.fft` call does not: at
`beta = 0.5` the total variation distance between the correct momentum distribution
and the untwisted one reaches 0.5.

`evolve` returns a `QuantumEvolution` whose `final_state` is always `(*batch, N)`
and whose `history` is `(*batch, T, N)` when `return_history=True` and `None`
otherwise. The return **type** does not depend on the arguments, only the presence
of the optional history does, and `metadata.parameters` records which route
actually ran (`method`, with `"auto"` already resolved, alongside
`requested_method`) plus the norm audit as `norm_audits` and
`maximum_norm_drift`.

See `docs/design/quantum-instability-diagnostics.md` for the operator convention,
the closed form that pins it, and why no Lyapunov exponent is fitted from either
curve. `loschmidt_echo(reference, perturbed, state, steps=...)` returns
`M(t) = |<psi| U_b^(-t) U_a^t |psi>|**2` for `t = 0 .. steps` as an
`AnalysisResult`; batched initial states give `values` of shape
`(*batch, steps + 1)`. The values are deliberately **not** clipped into `[0, 1]`,
because a clipped diagnostic hides the numerical drift it should report. `otoc`
returns `C(t) = <|[A(t), B]|**2>` with `A(t) = U^(-t) A U^t`, defaulting to the
Weyl translation pair from `weyl_translations(N, boundary_phases=...)`; both
operators are bounded unitaries, so `C` saturates near `2` rather than growing
without limit, and the fitted rate is the order-2 generalized Lyapunov exponent
rather than `2 lambda`. `otoc` is dense-only and honours `dense_limit`.

`wigner_distribution` requires an even dimension and returns a real distribution on
the same `(q, p)` grid as `husimi_distribution`. Its marginals equal the position and
momentum probabilities exactly, and unlike the Husimi distribution it takes negative
values, which is why it exists.

## `chaos_numerics.spectral`

- Containers: `PreparedEigenphases`, `UnfoldedSpectrum`,
  `SpacingDistributionResult`, `SpectralCurve`
- Level statistics: `prepare_eigenphases`, `unfold`, `adjacent_gap_ratios`,
  `spacing_distribution`, `mean_gap_ratio_reference`
- Long-range statistics: `spectral_form_factor`, `number_variance`,
  `spectral_rigidity`, `rmt_reference`
- Type aliases: `Ensemble`, `Statistic`, `Window`

`rmt_reference` covers five statistics. `"spectral_form_factor"` and
`"number_variance"` return bulk analytic curves. `"spacing_distribution"` returns the
Wigner surmise, which is the exact result for a 2x2 matrix rather than the large-`N`
limit, and therefore rejects the `dimension` argument. `"spectral_rigidity"` is
derived from the number variance through Mehta's exact relation and rejects
`dimension` for the same reason: the relation is a bulk identity, so a finite-`N`
kernel would return a plausible-looking number that means nothing.
`"gap_ratio_distribution"` returns the ratio density, folded onto `[0, 1]` by default
to match what `adjacent_gap_ratios` produces; pass `folded=False` for the density on
`(0, inf)`. Its Poisson case is the exact `1 / (1 + r)**2`, not the `beta -> 0` limit
of the surmise, which is a different distribution with a mean 5.8% away.
`mean_gap_ratio_reference` returns the large-`N` mean adjacent gap ratio by
default and the 3x3 surmise value with `surmise=True`; the two differ in the third
decimal and comparing against the wrong one is a common error. `"coe"` is accepted
as an alias for `"goe"` and `"cse"` for `"gse"`, because bulk statistics do not
distinguish a circular ensemble from its Gaussian counterpart.

`"gse"` (`beta = 4`) is a reference-only ensemble: no model in this library
produces GSE statistics, because `beta = 4` needs an antiunitary symmetry squaring
to `-1` and every model here is spinless. Its curves are written for the
*distinct* levels of the Kramers-degenerate spectrum. `dimension` is rejected for
`"gse"`/`"cse"` on every statistic, not only on the ones where it is rejected for
the other ensembles. `K_GSE(tau)` has a real logarithmic pole at the Heisenberg
time, so `"spectral_form_factor"` raises `ValidationError` for a grid that lands
exactly on `tau = 1` rather than returning an infinity; step over it.

## `chaos_numerics.experiment`

- Configuration: `Experiment`, `cartesian_grid`, `zip_grid`
- Execution: `run_experiment`, `run_sweep`
- Persistence: `load_result`
- Summaries: `ExperimentRun`, `SweepResult`

Array shapes, dtype rules, dependency direction, and stability policy are defined
in [public-api.md](design/public-api.md). Numerical tolerances and convergence
behavior are defined in [numerical-standards.md](design/numerical-standards.md).

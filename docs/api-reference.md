# API reference

This page lists the supported v0.1 public import surface. Signatures and detailed
validation rules live in the referenced objects' docstrings; names not exported
through a package `__all__` are implementation details.

## Top level: `chaos_numerics`

- Result types: `Trajectory`, `Spectrum`, `EigenstateResult`, `AnalysisResult`
- Classical models: `StandardMap`, `CatMap`, `BakerMap`
- Quantum models: `KickedRotor`, `QuantumCatMap`, `QuantumBakerMap`
- Experiments: `Experiment`, `run_experiment`, `run_sweep`
- Package version: `__version__`

## `chaos_numerics.core`

- Protocols: `ClassicalMap`, `Flow`, `QuantumMap`, `LinearOperatorLike`, `Partition`
- Results and metadata: `Trajectory`, `Spectrum`, `EigenstateResult`,
  `AnalysisResult`, `ExperimentMetadata`, `ConvergenceInfo`, `Diagnostic`
- Error hierarchy: `ChaosNumericsError`, `ValidationError`, `NumericalError`,
  `ConvergenceError`
- Warning hierarchy: `ChaosNumericsWarning`, `NumericalWarning`,
  `ConvergenceWarning`, `ReproducibilityWarning`

## `chaos_numerics.classical`

- Models: `StandardMap`, `CatMap`, `BakerMap`
- Evolution: `iterate`
- Stability: `largest_lyapunov_exponent`, `lyapunov_spectrum`
- Transport: `autocorrelation`, `mean_square_displacement`,
  `local_diffusion_exponent`
- Periodic orbits: `find_periodic_orbits`, `PeriodicOrbitResult`

## `chaos_numerics.operators`

- Partitions: `RectangularPartition`, `UniformPartition`
- Ulam: `UlamMatrix`, `build_ulam`
- Analysis: `leading_eigenpairs`, `stationary_density`, `spectral_gap`

## `chaos_numerics.quantum`

- State conventions: `BoundaryPhases`, `QuantumBasis`, `basis_state`,
  `quantum_state`, `normalize_state`
- Models and adapters: `KickedRotor`, `QuantumCatMap`, `QuantumBakerMap`,
  `DenseUnitary`
- Evolution and spectra: `evolve`, `eigenstates`, `eigenphases`,
  `unitarity_defect`
- Phase space: `coherent_state`, `husimi_distribution`, `HusimiResult`,
  `inverse_participation_ratio`, `participation_ratio`, `shannon_entropy`

## `chaos_numerics.spectral`

- Containers: `PreparedEigenphases`, `UnfoldedSpectrum`,
  `SpacingDistributionResult`, `SpectralCurve`
- Level statistics: `prepare_eigenphases`, `unfold`, `adjacent_gap_ratios`,
  `spacing_distribution`
- Long-range statistics: `spectral_form_factor`, `number_variance`,
  `rmt_reference`

## `chaos_numerics.experiment`

- Configuration: `Experiment`, `cartesian_grid`, `zip_grid`
- Execution: `run_experiment`, `run_sweep`
- Persistence: `load_result`
- Summaries: `ExperimentRun`, `SweepResult`

Array shapes, dtype rules, dependency direction, and stability policy are defined
in [public-api.md](design/public-api.md). Numerical tolerances and convergence
behavior are defined in [numerical-standards.md](design/numerical-standards.md).

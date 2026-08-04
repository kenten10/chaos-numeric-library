# Chaos Numerics Library v0.1 MVP Scope

Status: accepted baseline for KEN-108  
Last reviewed: 2026-08-04

## 1. Product statement

Chaos Numerics Library is a Python numerical-computing library for reproducible
experiments in classical and quantum chaos. Version 0.1 focuses on finite,
discrete-time systems: classical maps, finite-dimensional quantum maps, Ulam
approximations of transfer operators, and the statistics needed to compare their
dynamics and spectra.

The v0.1 promise is deliberately narrower than the long-term product vision. It
provides a coherent path from a model through simulation and analysis to a saved,
reproducible result. It does not claim to be a general dynamical-systems framework.

## 2. Scope labels

| Label | Meaning |
| --- | --- |
| **v0.1** | Required for the first public release and covered by public API, tests, and documentation. |
| **Future** | Architecturally anticipated, but not required for v0.1 and not part of its compatibility promise. |
| **Out of scope** | No functional support is promised in v0.1; only a separately scheduled shared contract may exist. |

“Future” is not a delivery commitment. A future item enters a release only through
a separate design and implementation issue.

## 3. v0.1 capabilities

### 3.1 Shared foundation

| Capability | Scope | v0.1 boundary |
| --- | --- | --- |
| Typed model/operator protocols | **v0.1** | Separate protocols for classical maps, future flows, quantum maps, linear operators, and partitions; no universal system base class. The Flow protocol does not imply a v0.1 solver or model. |
| Result objects | **v0.1** | Trajectories, spectra/eigenstates, analysis results, convergence history, residuals, warnings, parameters, and provenance. |
| Numeric conventions | **v0.1** | NumPy/SciPy, `float64`/`complex128` defaults, documented shapes and periodic-coordinate conventions. |
| Dense and matrix-free operators | **v0.1** | Dense references for small problems; SciPy sparse/`LinearOperator` and FFT paths for scalable work. |
| Reproducibility | **v0.1** | Deterministic seed handling plus Python, dependency, platform, runtime, and Git metadata when available. |

### 3.2 Classical maps and trajectory analysis

| Capability | Scope | v0.1 boundary |
| --- | --- | --- |
| Standard map | **v0.1** | Scalar and batch stepping/iteration, Jacobian, parameter checks, periodic coordinates. |
| Cat map | **v0.1** | Two-dimensional torus map with integer-matrix and determinant validation. |
| Baker map | **v0.1** | Two-dimensional piecewise map with an explicit branch-boundary convention. |
| Trajectory generation | **v0.1** | Single and batched initial states with consistent output shapes. |
| Lyapunov analysis | **v0.1** | Largest exponent and QR-based spectrum, transient controls, finite-time history, convergence diagnostics. |
| Correlation and transport | **v0.1** | Autocorrelation, mean-square displacement, and local diffusion exponent with explicit observables and fit windows. |
| Low-period orbits | **v0.1** | Root-finding search, residuals, monodromy matrices, and stability multipliers. |

### 3.3 Transfer operators and Ulam approximation

| Capability | Scope | v0.1 boundary |
| --- | --- | --- |
| Uniform/rectangular partitions | **v0.1** | One- and two-dimensional cell lookup, bounds, sampling, index conversion, and periodic boundaries. |
| Ulam matrix construction | **v0.1** | Seeded Monte Carlo cell sampling and CSR output; probability orientation is documented and tested. |
| Leading spectral data | **v0.1** | Leading eigenpairs, stationary density, spectral gap, solver convergence, and eigenpair residuals. |
| Cat-map convergence example | **v0.1** | Reproducible partition/sample-size study; it demonstrates convergence without promising a universal rate. |

The v0.1 transfer-operator claim is an Ulam discretization. A general symbolic or
exact Perron–Frobenius/Koopman operator algebra is not implied.

### 3.4 Quantum maps

| Capability | Scope | v0.1 boundary |
| --- | --- | --- |
| Quantum states and unitary operators | **v0.1** | Normalized finite-dimensional states, basis/boundary-phase metadata, dense and matrix-free evolution. |
| Kicked rotor | **v0.1** | Finite torus quantization, a dense reference Floquet matrix, and split-operator/FFT evolution. |
| Quantum cat map | **v0.1** | Supported quantizable matrices and boundary phases with explicit rejection of invalid conditions. |
| Quantum baker map | **v0.1** | One documented finite-dimensional quantization and boundary convention. |
| Quantum diagnostics | **v0.1** | Norm/unitarity errors, eigenphases/eigenstates, residuals, degeneracy and symmetry warnings. |

Version 0.1 need not expose every quantization found in the literature. Each model
must name and test the convention it implements.

### 3.5 Spectral and phase-space analysis

| Capability | Scope | v0.1 boundary |
| --- | --- | --- |
| Eigenphase preparation | **v0.1** | Circular sorting/wrapping, endpoint spacing, degeneracy diagnosis, and preservation of source data. |
| Unfolding and spacing statistics | **v0.1** | Explicitly selected unfolding, spacing distributions, and adjacent-gap ratios with small-sample handling. |
| Long-range spectral statistics | **v0.1** | Spectral form factor and number variance with stated normalization, windows, averaging, and uncertainty. |
| Random-matrix references | **v0.1** | Reproducible Poisson, GOE, GUE, and CUE theory or numerical reference data. |
| Phase-space state analysis | **v0.1** | Periodized torus coherent states, normalized Husimi data, IPR, participation ratio, and Shannon entropy. |
| Classical/quantum comparison data | **v0.1** | Data suitable for overlaying classical trajectories and quantum phase-space densities; plotting is separate from core numerics. |

### 3.6 Experiments, persistence, quality, and distribution

| Capability | Scope | v0.1 boundary |
| --- | --- | --- |
| Experiment execution | **v0.1** | Single runs plus Cartesian-product and zip parameter sweeps. |
| Robust sweeps | **v0.1** | Deterministic derived seeds, progress records, continue-on-failure, and restart of incomplete work. |
| Result persistence | **v0.1** | JSON metadata separated from NPZ arrays; large arrays are never embedded in JSON. |
| Optional Zarr storage | **Future** | Considered for larger-than-memory or chunked data after the minimal format is stable. |
| Verification | **v0.1** | Unit, property, integration, numerical-reference, and executable-documentation tests. |
| Benchmarks | **v0.1** | Scalar/batch trajectories, Ulam construction, FFT evolution, and eigenanalysis with stated regression thresholds. |
| Documentation and packaging | **v0.1** | Quick start, tutorials, API reference, changelog, license/citation guidance, wheel and sdist, and TestPyPI validation. |

## 4. Future candidates

These capabilities should not shape the v0.1 public API beyond leaving normal
extension points such as protocols and optional dependencies.

| Capability | Scope | Rationale for deferral |
| --- | --- | --- |
| General Perron–Frobenius and Koopman methods beyond Ulam | **Future** | Requires separate choices for bases, estimators, observables, and convergence guarantees. |
| Adaptive and higher-dimensional partitions | **Future** | Depends on demonstrated limits of uniform/rectangular partitions. |
| Additional classical and quantum maps | **Future** | v0.1 validates the API with three canonical examples in each domain. |
| Additional quantization conventions and symmetry-sector tooling | **Future** | Requires model-specific design and validation data. |
| Advanced periodic-orbit methods and semiclassical formulas | **Future** | The v0.1 low-period search and comparison data are foundations, not a full semiclassical package. |
| Alternative persistence backends such as Zarr | **Future** | NPZ plus JSON is sufficient for the minimal interoperable release. |
| Numba acceleration | **Future** | Optional optimization only after NumPy/SciPy correctness and benchmarks are stable. |
| JAX and CuPy backends | **Future** | Backend abstraction and device semantics are not part of the v0.1 compatibility contract. |

## 5. Explicitly out of scope for v0.1

| Capability | Scope | Boundary |
| --- | --- | --- |
| Billiards | **Out of scope** | No collision geometry, event location, or boundary reflection API. |
| Continuous-time flows and ODE integration | **Out of scope** | No concrete flow models, integrators, variational ODE solvers, or Poincare-section machinery. A minimal shared `Flow` protocol required by KEN-112 is an interoperability contract, not functional support. |
| Quantum graphs | **Out of scope** | No graph scattering or secular-equation API. |
| Many-body systems | **Out of scope** | No tensor-product Hilbert spaces, spin chains, or many-body eigensolvers. |
| GPU execution | **Out of scope** | No GPU performance or device-array compatibility claim. |

“Out of scope” prevents partial, untested stubs in v0.1. It does not prohibit a
later project from proposing these capabilities.

## 6. User stories

1. As a researcher studying a classical map, I can generate reproducible scalar
   or batched Standard-, Cat-, or Baker-map trajectories and inspect parameters,
   initial states, and warnings in the returned trajectory.
2. As a nonlinear-dynamics user, I can estimate the largest Lyapunov exponent or
   spectrum and examine finite-time convergence rather than receiving only a
   context-free scalar.
3. As a transport researcher, I can compute autocorrelation, mean-square
   displacement, and a local diffusion exponent while explicitly selecting the
   observable and fit interval.
4. As an operator-theory user, I can build a seeded sparse Ulam approximation,
   verify probability conservation, and obtain leading eigenpairs, residuals, an
   invariant density, and a spectral gap.
5. As a quantum-chaos researcher, I can evolve a kicked-rotor state with an FFT
   implementation without materializing a dense matrix and compare it with the
   small-system dense reference.
6. As a quantum-map user, I can construct supported quantum cat and baker maps,
   reject invalid quantization parameters early, and inspect unitarity and
   eigenpair residuals.
7. As a spectral-statistics user, I can preserve raw eigenphases, unfold them with
   an explicit method, calculate short- and long-range statistics, and compare
   results with reproducible Poisson/RMT references.
8. As a phase-space-analysis user, I can calculate normalized Husimi data and
   localization measures and combine them with classical trajectory data in my
   own plotting workflow.
9. As a computational scientist, I can run or resume a deterministic parameter
   sweep and reload arrays and provenance without storing large arrays in JSON.
10. As a new user, I can install a wheel in a clean environment and execute every
    tutorial example against the documented public API.

## 7. Representative public API sketches

These examples are product-level contracts, not final signatures. KEN-109 owns
the exact module boundaries, names, shapes, and re-export policy. The examples
intentionally keep model construction, numerical analysis, and plotting separate.

### 7.1 Classical trajectory and Lyapunov exponent

```python
import numpy as np

from chaos_numerics.classical import StandardMap, iterate, lyapunov_spectrum

model = StandardMap(kick_strength=5.0)
trajectory = iterate(model, initial_state=np.array([0.1, 0.2]), steps=10_000)
result = lyapunov_spectrum(model, trajectory.initial_state, steps=10_000, seed=7)

print(result.values, result.residuals, result.warnings)
```

### 7.2 Sparse Ulam approximation

```python
from chaos_numerics.classical import CatMap
from chaos_numerics.operators import UniformPartition, build_ulam, leading_eigenpairs

model = CatMap(matrix=((2, 1), (1, 1)))
partition = UniformPartition(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(64, 64))
ulam = build_ulam(model, partition, samples_per_cell=256, seed=7)
spectrum = leading_eigenpairs(ulam, count=8)

print(spectrum.eigenvalues, spectrum.residuals)
```

### 7.3 Matrix-free kicked-rotor evolution

```python
from chaos_numerics.quantum import KickedRotor, basis_state, evolve

model = KickedRotor(dimension=4096, kick_strength=8.0, boundary_phase=0.0)
state = basis_state(dimension=model.dimension, index=0)
evolved = evolve(model, state, steps=100, method="fft")

print(evolved.norm_error, evolved.metadata)
```

### 7.4 Spectral statistics

```python
from chaos_numerics.spectral import (
    adjacent_gap_ratios,
    prepare_eigenphases,
    spectral_form_factor,
    unfold,
)

prepared = prepare_eigenphases(raw_phases, symmetry_sector="even")
unfolded = unfold(prepared, method="polynomial")
ratios = adjacent_gap_ratios(unfolded)
form_factor = spectral_form_factor(unfolded, times, window="hann", bootstrap=500, seed=7)
```

### 7.5 Reproducible parameter sweep

```python
from chaos_numerics.experiment import Experiment, cartesian_grid, run_sweep

experiment = Experiment(model="standard_map", analysis="lyapunov_spectrum", seed=7)
sweep = run_sweep(
    experiment,
    parameters=cartesian_grid(
        kick_strength=[0.5, 1.0, 5.0],
        steps=[1_000, 10_000],
    ),
    output="results/standard-map",
    resume=True,
)
```

## 8. Release boundary and acceptance

Version 0.1 is ready only when all **v0.1** rows above have typed public APIs,
documented conventions, numerical or independent-reference validation, and tests
in CI. Large problems must use sparse, matrix-free, or FFT paths where specified;
dense construction remains a small-problem reference, not the default scalable
implementation.

The release must also document unsupported parameter regimes and return residuals,
convergence information, or warnings when a numerical answer cannot be trusted.

## 9. Scope review and decisions

- The project vision names classical flows and billiards, while KEN-108 explicitly
  excludes them from v0.1. This document resolves the apparent tension by treating
  them as outside the first release rather than forcing them into a common model
  API. KEN-112's minimal `Flow` protocol may exist as a separate future-facing
  contract, but no v0.1 flow model, solver, analysis, or documentation workflow is
  promised.
- “Perron–Frobenius/Koopman operators” is broader than the scheduled Phase 3 work.
  v0.1 therefore promises the Ulam approximation and its sparse spectral analysis,
  not a general operator-method suite.
- “Semiclassical correspondence” is bounded to comparable classical-trajectory,
  periodic-orbit, spectrum, and phase-space data. Trace formulas and other advanced
  semiclassical algorithms are deferred.
- RMT support is analysis/reference data, not a general random-matrix simulation
  library.
- Plotting is kept outside the numerical core. Tutorials may use an optional plotting
  dependency, while public numerical functions return plot-ready arrays and metadata.
- Numba, JAX, CuPy, Zarr, and GPU work are not allowed to delay the NumPy/SciPy MVP.
- The API snippets above remain provisional until KEN-109 fixes naming and module
  boundaries; their workflows and separation of concerns are the stable requirement.

## 10. Traceability

This scope is derived from the Chaos Numerics Library project overview and issues
KEN-101 through KEN-129, with KEN-127 treated as a duplicate of KEN-128. Detailed
API design belongs to KEN-109, numeric thresholds to KEN-110, and implementation to
the phase and child issues that follow them.

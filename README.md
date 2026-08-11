# Chaos Numerics Library

Chaos Numerics Library is a typed Python numerical-computing library for
reproducible experiments in classical and quantum chaos. The project is currently
pre-alpha; the public v0.1 scope and API are being implemented from the design
documents in `docs/`.

## Installation

The project is pre-alpha and **not published on PyPI yet**, so install from a
checkout:

```bash
git clone https://github.com/kenten10/chaos-numeric-library
cd chaos-numeric-library
python -m pip install .
```

For a first look at what the library reproduces, open
[notebooks/known_results.ipynb](notebooks/known_results.ipynb): it runs
end-to-end and checks fourteen standard results of classical and quantum chaos
against their analytic values. Contributors should read
[Development installation](#development-installation) below.

## Requirements

- CPython 3.11, 3.12, 3.13, or 3.14
- NumPy 1.26.4 or newer
- SciPy 1.11.4 or newer, below SciPy 2

NumPy and SciPy are the only runtime dependencies. Accelerators, plotting, and
alternative storage backends are not required by the numerical core.

## Development installation

Create and activate an isolated virtual environment, then install the package in
editable mode:

```bash
python -m pip install -e .
```

For all contributor tools and tests:

```bash
python -m pip install -e ".[dev,test]"
```

Run the same checks as CI:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

Install the Git hooks after the repository has been initialized or cloned:

```bash
pre-commit install
pre-commit run --all-files
```

## Optional extras

Optional features are separated from runtime requirements:

| Extra | Install command | Purpose |
| --- | --- | --- |
| `test` | `python -m pip install -e ".[test]"` | pytest test runner |
| `dev` | `python -m pip install -e ".[dev]"` | Ruff, mypy, and pre-commit |
| `plot` | `python -m pip install -e ".[plot]"` | Matplotlib for tutorials and user-owned plots |

The `plot` extra does not make plotting part of the numerical public API. Future
Numba, JAX, CuPy, GPU, and Zarr support will receive separate design work before
an extra is added.

## Result objects

Numerical results own immutable copies of their arrays and retain reproducibility
metadata. Single trajectories use `(time, state_dim)`; batch axes may be prepended:

```python
import numpy as np

from chaos_numerics.core import ExperimentMetadata, Trajectory

trajectory = Trajectory(
    states=np.array([[1.0, 0.0], [0.8, 0.2], [0.3, 0.7]]),
    initial_state=np.array([1.0, 0.0]),
    metadata=ExperimentMetadata(parameters={"steps": 2, "dt": 0.1}, seed=42),
)

print(trajectory)  # compact: arrays are not expanded
states = trajectory.states          # read-only view owned by the result
payload = trajectory.array_payload()  # independent writable copies for storage
```

`Spectrum`, `EigenstateResult`, and `AnalysisResult` follow the same ownership
policy. `metadata_payload()` returns JSON-compatible metadata and array descriptors;
large arrays remain separate for a storage backend.

## Classical maps

Built-in maps use `(q, p)` coordinates on the half-open unit torus. `step` and
`jacobian` preserve leading batch axes, while `iterate` inserts a time axis before
the coordinate axis:

```python
import numpy as np

from chaos_numerics.classical import StandardMap, iterate

model = StandardMap(kick_strength=5.0)
initial = np.array([[0.1, 0.2], [0.3, 0.4]])  # batch shape (2, 2)
trajectory = iterate(model, initial, steps=100)

print(trajectory.shape)  # (2, 101, 2): (batch, time, coordinate)
print(model.jacobian(initial).shape)  # (2, 2, 2)
```

Each step is linear in the number of input states and `iterate` is linear in both
the step count and batch size. `CatMap` and `BakerMap` follow the same array API.

Finite-time Lyapunov spectra use periodic QR reorthogonalization and retain their
convergence history:

```python
import numpy as np

from chaos_numerics.classical import CatMap, lyapunov_spectrum

result = lyapunov_spectrum(
    CatMap(),
    np.array([0.1, 0.2]),
    steps=20_000,   # the time average converges as 1/sqrt(steps)
    transient=100,
    reorthogonalization_interval=4,
    history_interval=5,
)

print(result.values)                   # [+0.9624, -0.9624] = +/- log((3+sqrt(5))/2)
print(result.metadata.convergence.converged)
```

`largest_lyapunov_exponent` uses the lower-memory tangent-vector method. Both
algorithms cost `O(steps * d³)` for a full spectrum and `O(steps * d²)` for the
largest exponent, excluding the model's own cost. QR is performed approximately
`steps / reorthogonalization_interval` times.

Correlation, transport, and low-period orbit analysis keep averaging and fit
conventions explicit:

```python
import numpy as np

from chaos_numerics.classical import (
    BakerMap,
    CatMap,
    StandardMap,
    autocorrelation,
    find_periodic_orbits,
    iterate,
    local_diffusion_exponent,
    mean_square_displacement,
)


def momentum(states):
    return np.asarray(states)[..., 1:2]


# Ensembles throughout, because none of these error bars can be calibrated from
# one trajectory: adjacent lags of a time-origin-averaged curve are strongly
# correlated, so neither the scatter over time origins nor a fit residual says
# anything about the uncertainty. Every one of these functions returns
# `uncertainty=None` and warns for a single trajectory rather than reporting a
# number that is up to 21x too small. `per_trajectory` keeps the MSD curves
# separate so the slope fit can report the spread across them.
ensemble = iterate(StandardMap(2.0), np.random.default_rng(7).random((32, 2)), steps=4_000)
series = np.asarray(ensemble.states)[..., 0]   # one scalar observable per member
correlation = autocorrelation(series, max_lag=100, demean=True)
curves = mean_square_displacement(
    ensemble, observable=momentum, max_lag=200, unwrap=True, periods=1.0, per_trajectory=True
)
diffusion = local_diffusion_exponent(curves, fit_start=20, fit_stop=200)
orbits = find_periodic_orbits(CatMap(), np.array([[0.01, 0.02], [0.4, 0.7]]), period=1)

print(correlation.values[:3], correlation.uncertainty[:3])
print(diffusion.values, diffusion.uncertainty)
print(orbits.residuals, orbits.stability_multipliers)
baker = iterate(BakerMap(), np.random.default_rng(11).random((16, 2)), steps=500)
print(mean_square_displacement(baker, max_lag=100).values[:3])
```

`unwrap=True` reconstructs displacements across a periodic boundary from wrapped
coordinates, which is only possible when a single step moves less than half a
period. That assumption cannot be verified from wrapped data after the fact, so
the library refuses input whose reconstructed displacements reach the ambiguity
limit instead of returning a silently wrong result. For the standard map that
puts the boundary at `K = pi`, where one kick moves the momentum by half the
torus; the angle `q` hops by order one per step at any chaotic `K` and can never
be unwrapped, which is why the example selects the momentum with `observable`.
Beyond those limits, supply coordinates that were never wrapped.

Correlation and MSD use direct `O(T * max_lag)` reference algorithms. Periodic
search cost depends on the root solver and requested period; each residual and
monodromy evaluation costs `O(period)` model/Jacobian calls.

## Kicked rotor

`KickedRotor` uses a finite torus with position ordering
`q_j = (j + alpha) / N`, signed NumPy FFT momentum ordering, and
`hbar_eff = 2*pi/N`. A Floquet step applies the kick first and free rotation
second. The twisted position-to-momentum transform is orthonormal
(`norm="ortho"`), so its FFT action is unitary without manual scaling.

```python
import numpy as np

from chaos_numerics.quantum import KickedRotor, basis_state, evolve

model = KickedRotor(dimension=4096, kick_strength=8.0, boundary_phases=0.0)
state = basis_state(dimension=model.dimension, index=0)
evolved = evolve(model, state, steps=100, method="fft")

print(abs(np.linalg.norm(evolved.final_state) - 1.0))
```

A scalar `boundary_phases` applies the same phase, measured in turns modulo one,
to both position and momentum boundaries. Pass `BoundaryPhases(position, momentum)`
for independent twists. `apply_fft` costs `O(N log N)` time and `O(N)` auxiliary
memory; it never constructs the Floquet matrix. `to_dense` and `apply_dense` are
small-`N` reference operations requiring `O(N^2)` memory and dense algebra.

The other built-in torus maps use deliberately narrower quantization sectors:

```python
from chaos_numerics import BakerMap, CatMap, QuantumBakerMap, QuantumCatMap
from chaos_numerics.quantum import eigenstates, unitarity_defect

classical_cat = CatMap()
quantum_cat = QuantumCatMap(dimension=64, matrix=classical_cat.matrix)
classical_baker = BakerMap(cut=0.5)
quantum_baker = QuantumBakerMap(dimension=64)

result = eigenstates(quantum_baker)
print(unitarity_defect(quantum_cat), result.metadata.parameters["symmetry_defects"])
```

The cat map uses an even-dimensional periodic metaplectic sector with
`matrix[0][1] = +/-1`. The baker map uses the even-dimensional, anti-periodic
Saraceno quantization corresponding to the symmetric classical cut. See
`docs/design/quantum-cat-baker.md` for the kernels and rejected conditions.

## Eigenphase statistics

Circular level statistics preserve the input phases and require an explicit
symmetry-sector label:

```python
from chaos_numerics.quantum import BoundaryPhases, KickedRotor, eigenstates
from chaos_numerics.spectral import (
    adjacent_gap_ratios,
    prepare_eigenphases,
    spacing_distribution,
    unfold,
)

# Generic Bloch phases break parity and time reversal, leaving a single CUE sector.
rotor = KickedRotor(128, 10.0, BoundaryPhases(position=0.25, momentum=0.13))
raw_phases = eigenstates(rotor).eigenphases

prepared = prepare_eigenphases(raw_phases, symmetry_sector="parity and time reversal broken")
unfolded = unfold(prepared, method="mean")
ratios = adjacent_gap_ratios(unfolded, degeneracy="drop")
histogram = spacing_distribution(unfolded, bins=30, value_range=(0.0, 4.0))

print(ratios.values.mean())   # near 0.5996, the CUE value
```

Preparation wraps to `[0, 2*pi)`, sorts, and retains the gap across the circular
branch cut. The result stores raw, processed, and unfolded arrays; omitted sector
information and degeneracies produce numerical diagnostics. See
`docs/design/spectral-level-statistics.md` for unfolding and zero-gap policies.

Long-range statistics return plot-ready curves without importing a plotting
library:

```python
import numpy as np

from chaos_numerics.quantum import BoundaryPhases, KickedRotor, eigenstates
from chaos_numerics.spectral import (
    number_variance,
    prepare_eigenphases,
    rmt_reference,
    spectral_form_factor,
    unfold,
)

rotor = KickedRotor(128, 10.0, BoundaryPhases(position=0.25, momentum=0.13))
unfolded = unfold(prepare_eigenphases(eigenstates(rotor).eigenphases, symmetry_sector="cue"))

times = np.linspace(0.0, 2.0, 101)
lengths = np.linspace(0.0, 8.0, 41)
sff = spectral_form_factor(unfolded, times, window="hann", bootstrap=100, seed=7)
variance = number_variance(unfolded, lengths, samples=4096)
cue_sff = rmt_reference("spectral_form_factor", "cue", times)
cue_variance = rmt_reference("number_variance", "cue", lengths)
```

A single spectrum does not self-average: the form factor of one realization
fluctuates by order one about the RMT curve. Published ramp-and-plateau figures
average over an ensemble of spectra (for the kicked rotor, over Bloch phases).
The `bootstrap` argument estimates the resampling error of one spectrum and is
not a substitute for that ensemble average.

That ensemble is a parameter sweep, so it belongs in `run_sweep` rather than in a
hand-written loop. `cartesian_grid(boundary_phases=[...])` over the `kicked_rotor`
model with the `spectral_form_factor` analysis produces one persisted `SpectralCurve`
per phase, and averaging them recovers the CUE ramp and plateau: measured over twelve
phases at `N = 128`, the ramp slope came to 1.08 and the plateau to 0.98, and the
ensemble mean sat four times closer to the reference than any single member.

`spectral_rigidity` adds the Dyson-Mehta `Delta_3(L)`, computed by solving the
in-window least squares in closed form; its reference curve follows from the number
variance through Mehta's exact relation, so both share one cluster-function
implementation. `rmt_reference("gap_ratio_distribution", ...)` completes the pair with
`mean_gap_ratio_reference`, and is folded onto `[0, 1]` to match what
`adjacent_gap_ratios` returns.

SFF time is normalized by the Heisenberg time and its plateau is one. Number
variance uses circular half-open counting windows. RMT references cover Poisson,
GOE, GUE, CUE, and GSE, including an optional finite-`N` CUE number-variance
kernel. Bulk statistics do not distinguish a circular ensemble from its Gaussian
counterpart, so `"coe"` is accepted as an alias for `"goe"` and `"cse"` for
`"gse"`, and the returned metadata records both the requested and the canonical
name. **No model in this library produces GSE (`beta = 4`) statistics** -- that
needs an antiunitary symmetry squaring to `-1`, so half-integer spin, and every
model here is spinless. The GSE curves are shipped for a caller comparing an
external spectrum against the library's conventions; comparing a library model
against `"gse"` almost certainly means the wrong ensemble was picked.

## Quantum phase space

Torus coherent states, Husimi data, and the Wigner distribution share the
position-basis and boundary-phase conventions used by the quantum maps. Pass several
states at once to amortize the coherent-state grid: measured at `N = 1024` on a
`64 x 64` grid, twenty states in one call cost the same as one, a 19x saving.
`wigner_distribution` needs an even dimension, reproduces the position and momentum
marginals exactly, and takes negative values where the Husimi distribution cannot,
which is what makes interference visible:

```python
from chaos_numerics.quantum import (
    coherent_state,
    husimi_distribution,
    inverse_participation_ratio,
)

state = coherent_state(dimension=64, position=0.2, momentum=0.3)
husimi = husimi_distribution(state, grid_shape=(64, 64))

print(husimi.grid_points.shape, husimi.values.shape, husimi.integral)
print(inverse_participation_ratio(state))
```

`grid_points` uses classical `(q,p)` coordinate order, while plotting remains
outside the numerical package. The Husimi result retains its raw quadrature
integral and normalized density. IPR, participation ratio, and Shannon entropy
support both scalar and batched states.

## Partitions

Partitions use half-open cells and C-order flat indices. Internal cuts belong to
the right cell, while periodic upper endpoints wrap to the lower endpoint:

```python
import numpy as np

from chaos_numerics.operators import RectangularPartition, UniformPartition

uniform = UniformPartition(
    bounds=((0.0, 1.0), (0.0, 1.0)),
    shape=(8, 8),
    periodic=(True, True),
)
indices = uniform.locate(np.array([[0.1, 0.2], [1.0, -0.1]]))
samples = uniform.sample(index=0, count=256, seed=7)

nonuniform = RectangularPartition(
    edges=(np.array([0.0, 0.2, 1.0]), np.array([0.0, 0.5, 1.0])),
)
multi = nonuniform.unravel_index(np.array([0, 3]))
```

Ulam matrices use the column-stochastic convention
`P[target_cell, source_cell] = Pr(target_cell | source_cell)`:

```python
from chaos_numerics.classical import CatMap
from chaos_numerics.operators import UniformPartition, build_ulam

partition = UniformPartition(
    bounds=((0.0, 1.0), (0.0, 1.0)),
    shape=(32, 32),
    periodic=(True, True),
)
ulam = build_ulam(CatMap(), partition, samples_per_cell=256, seed=7, batch_size=64)

print(ulam.shape, ulam.nnz)  # (1024, 1024) 4096
print(ulam.matrix.format)  # csr
print(ulam.escape_probabilities)  # zero for this closed system
```

`UlamMatrix` is an immutable container, not a SciPy sparse subclass. It forwards
`shape`, `dtype`, `nnz`, `matvec`, `toarray()`, and `@`, which is enough to satisfy
the `LinearOperatorLike` protocol and to pass it straight to the eigensolvers
below. Every other SciPy operation goes through the `.matrix` attribute. Escape
diagnostics do not survive such an operation, because the escaped mass of a product
is not a function of the operands' escaped mass, so carry them yourself when
composing matrices.

Set `open_system=True` to retain escaped mass as substochastic column deficits.
The constructor enforces column sums no greater than one, so substochasticity is a
property of the type rather than something each consumer rechecks. Sampling uses
independent per-cell child seeds, so changing `batch_size` does not change the
matrix. Source cells are the future parallelization boundary. Empty geometric cells
and `samples_per_cell=0` are rejected during validation.

Leading right or left eigenpairs are ordered by eigenvalue modulus and include
independently recomputed residuals:

```python
from chaos_numerics.classical import CatMap
from chaos_numerics.operators import (
    UniformPartition,
    build_ulam,
    leading_eigenpairs,
    spectral_gap,
    stationary_density,
)

partition = UniformPartition([[0.0, 1.0], [0.0, 1.0]], (16, 16), periodic=(True, True))
ulam = build_ulam(CatMap(), partition, samples_per_cell=256, seed=7)

spectrum = leading_eigenpairs(ulam, count=8, side="right", tolerance=1e-10)
density = stationary_density(ulam)
gap = spectral_gap(spectrum)

print(spectrum.eigenvalues, spectrum.residuals)
print(density.values.sum(), gap.values[0])
# The Cat map preserves Lebesgue measure, so density.values is uniform at 1/256.
```

Run `python examples/cat_ulam_convergence.py --seed 7` for a reproducible Cat-map
study over partition resolutions and samples per cell.

## Experiments and parameter sweeps

Experiments combine a built-in model and analysis with explicit parameters and a
root seed. Sweeps derive stable per-job seeds, continue after individual failures,
and resume incomplete work without rerunning valid completed jobs:

<!-- docs-test: skip - writes a sweep directory to the filesystem -->
```python
from chaos_numerics.experiment import Experiment, cartesian_grid, run_sweep

experiment = Experiment(
    "standard_map",
    "lyapunov_spectrum",
    parameters={"initial_state": [0.1, 0.2], "transient": 50},
    seed=7,
)
summary = run_sweep(
    experiment,
    parameters=cartesian_grid(kick_strength=[2.0, 4.0], steps=[20_000, 40_000]),
    output="results/readme-standard-map",
    resume=True,
)
```

Pick a distinct `output` directory per study. A directory already holding a
different experiment or grid is rejected instead of being silently merged.

Expect a `ConvergenceWarning` from these jobs. A Lyapunov time average converges
only as `1/sqrt(steps)`, and how slowly depends strongly on the kick strength. The
two-window disagreement after 400,000 steps, against a default tolerance near
`1e-3`:

| `kick_strength` | residual after 400,000 steps |
| --- | --- |
| 2.0 | `1.2e-1` |
| 4.0 | `1.1e-2` |
| 5.0 | `3.8e-3` |
| 8.0 | `1.4e-3` |

Small `K` leaves large regular islands whose sticky orbits dominate the average,
so the sweep above (`K = 2` and `4`) is nowhere near the default tolerance and
never will be. Either relax `convergence_rtol` to the accuracy you actually need
or pass `strict=True` to turn non-convergence into a `ConvergenceError` instead of
a warning. Uniformly hyperbolic models converge quickly by comparison: `CatMap`
satisfies the default tolerance within 1,000 steps.

Each run stores metadata and array descriptors in JSON while keeping numerical
arrays in NPZ. Use `load_result(path)` to validate and reconstruct a stored core
result. See `docs/design/experiment-persistence.md` for the seed, resume, format,
and compatibility policy, or run `python examples/parameter_sweep.py --help`.

## Quantum foundations

Quantum states use a trailing Hilbert-space axis, `complex128`, and unit norm.
Dense and matrix-free maps share the same evolution and eigensystem API:

```python
import numpy as np

from chaos_numerics.quantum import DenseUnitary, basis_state, eigenstates, evolve

hadamard = DenseUnitary(
    np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
    name="hadamard",
)
initial = basis_state(dimension=2, index=0)
run = evolve(hadamard, initial, steps=8, return_history=True)
spectrum = eigenstates(hadamard)

print(run.history.shape, spectrum.eigenphases, spectrum.residuals)
```

`CylinderKickedRotor` is the momentum-basis model whose `effective_hbar` is
independent of the basis size, which is what dynamical localization needs; it
accepts the same `method="fft"` evolution. Both rotors expose `momentum_numbers`,
`to_momentum_basis`, and `to_position_basis`; the twisted transform carries the
boundary phases, which a bare `numpy.fft` call does not.

`evolve` returns a `QuantumEvolution`, not a bare array: `final_state` is always
`(*batch, N)` and `history` is `None` unless `return_history=True` asked for it,
so the return type no longer changes shape with an argument. The container also
carries which route actually ran and what the norm audit saw, both of which used
to be discarded at the return.

Two standard stand-ins for the classical Lyapunov instability ship alongside it.
`loschmidt_echo(reference, perturbed, state, steps=...)` measures how fast two
slightly different Floquet operators pull the same state apart; `otoc(model,
steps=...)` measures how fast one of them spreads an operator until it stops
commuting with a second one, defaulting to the Weyl translation pair that
`weyl_translations` builds. Both take the whole model rather than a perturbation
size, because which two operators are being compared is what a reader of the
result needs to know.

Boundary phases are represented in turns by `BoundaryPhases` and canonicalized
modulo one. When both twists are in `{0, 1/2}` the Floquet operator keeps a
reflection symmetry that `symmetry_operators` publishes, and a spectrum must be
split with `desymmetrize` before it is compared with a random-matrix ensemble. Dense diagnostics use `||U.H U - I||_F / sqrt(N)`; matrix-free
eigensystems materialize only up to an explicit `dense_limit`.

## Notebooks

[notebooks/known_results.ipynb](notebooks/known_results.ipynb) reproduces
published numerical results and compares each one against its analytic value:
the standard map's Poincare sections and the `K_c = 0.9716` KAM threshold, exact
Cat-map Lyapunov exponents, Rechester-White diffusion oscillations, the Ulam
approximation of the arcsine law, Husimi distributions against the classical
phase space, the Bohigas-Giannoni-Schmit conjecture across the Poisson, COE, and
CUE symmetry classes, the spectral form factor's ramp and plateau, number
variance, the arithmetic degeneracies that make the quantum Cat map a
counterexample to RMT universality, dynamical localization, and the two standard
quantum diagnostics of instability -- the Loschmidt echo's separation across the
KAM threshold and the OTOC, whose `C(0) = 4 sin^2(pi/N)` closed form pins the
whole operator convention.

It also shows how to plug in a model the library does not ship: a tent map,
written as a plain dataclass that satisfies the `ClassicalMap` protocol, goes
straight into `build_ulam` and `largest_lyapunov_exponent` and reproduces its
exact uniform invariant density and `log 2` exponent. Plotting needs the `plot`
extra; CI executes the notebook on the default branch.

## Design documents

- [docs/api-reference.md](docs/api-reference.md): supported v0.1 public names
- [docs/tutorials/](docs/tutorials/): CI-executed quickstart and domain tutorials
- [docs/product/mvp-scope.md](docs/product/mvp-scope.md): v0.1 product boundary
- [docs/design/public-api.md](docs/design/public-api.md): modules, public names, shapes, and dtypes
- [docs/design/numerical-standards.md](docs/design/numerical-standards.md): accuracy, validation, and performance gates
- [docs/design/experiment-persistence.md](docs/design/experiment-persistence.md): execution, resume, and storage contract
- [docs/design/kicked-rotor.md](docs/design/kicked-rotor.md): torus and cylinder quantization, FFT route, basis transforms
- [docs/design/quantum-cat-baker.md](docs/design/quantum-cat-baker.md): quantization kernels and the conditions that are rejected
- [docs/design/quantum-phase-space.md](docs/design/quantum-phase-space.md): coherent-state, Husimi, and Wigner conventions
- [docs/design/quantum-instability-diagnostics.md](docs/design/quantum-instability-diagnostics.md): Loschmidt echo and OTOC, and why no Lyapunov exponent is fitted from them
- [docs/design/spectral-level-statistics.md](docs/design/spectral-level-statistics.md): unfolding, spacings, gap ratios, and zero-gap policy
- [docs/design/long-range-spectral-statistics.md](docs/design/long-range-spectral-statistics.md): form factor, number variance, rigidity, and RMT references
- [docs/release-checklist.md](docs/release-checklist.md): the gates a release has to clear

## Citation and license

Citation metadata is available in [CITATION.cff](CITATION.cff); release changes
are recorded in [CHANGELOG.md](CHANGELOG.md). Chaos Numerics Library is
distributed under the BSD 3-Clause License. See [LICENSE](LICENSE) for the full
terms.

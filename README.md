# Chaos Numerics Library

Chaos Numerics Library is a typed Python numerical-computing library for
reproducible experiments in classical and quantum chaos. The project is currently
pre-alpha; the public v0.1 scope and API are being implemented from the design
documents in `docs/`.

## Requirements

- CPython 3.11, 3.12, 3.13, or 3.14
- NumPy 1.26.4 or newer
- SciPy 1.17 or newer, below SciPy 2

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
from chaos_numerics.classical import CatMap, lyapunov_spectrum

result = lyapunov_spectrum(
    CatMap(),
    np.array([0.1, 0.2]),
    steps=1_000,
    transient=100,
    reorthogonalization_interval=4,
    history_interval=5,
)

print(result.values)
print(result.metadata.convergence)
```

`largest_lyapunov_exponent` uses the lower-memory tangent-vector method. Both
algorithms cost `O(steps * d³)` for a full spectrum and `O(steps * d²)` for the
largest exponent, excluding the model's own cost. QR is performed approximately
`steps / reorthogonalization_interval` times.

Correlation, transport, and low-period orbit analysis keep averaging and fit
conventions explicit:

```python
from chaos_numerics.classical import (
    CatMap,
    autocorrelation,
    find_periodic_orbits,
    local_diffusion_exponent,
    mean_square_displacement,
)

correlation = autocorrelation(series, max_lag=100, demean=True)
msd = mean_square_displacement(positions, max_lag=100, unwrap=True, periods=[1.0, 1.0])
diffusion = local_diffusion_exponent(msd, fit_start=10, fit_stop=80)
orbits = find_periodic_orbits(CatMap(), guesses, period=1)

print(diffusion.values, orbits.residuals, orbits.stability_multipliers)
```

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

model = KickedRotor(dimension=4096, kick_strength=8.0, boundary_phase=0.0)
state = basis_state(dimension=model.dimension, index=0)
evolved = evolve(model, state, steps=100, method="fft")

print(abs(np.linalg.norm(evolved) - 1.0))
```

The scalar `boundary_phase` applies the same phase, measured in turns modulo one,
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
from chaos_numerics.spectral import (
    adjacent_gap_ratios,
    prepare_eigenphases,
    spacing_distribution,
    unfold,
)

prepared = prepare_eigenphases(raw_phases, symmetry_sector="even")
unfolded = unfold(prepared, method="polynomial", degree=5)
ratios = adjacent_gap_ratios(unfolded, degeneracy="drop")
histogram = spacing_distribution(unfolded, bins=30, value_range=(0.0, 4.0))
```

Preparation wraps to `[0, 2*pi)`, sorts, and retains the gap across the circular
branch cut. The result stores raw, processed, and unfolded arrays; omitted sector
information and degeneracies produce numerical diagnostics. See
`docs/design/spectral-level-statistics.md` for unfolding and zero-gap policies.

Long-range statistics return plot-ready curves without importing a plotting
library:

```python
import numpy as np

from chaos_numerics.spectral import number_variance, rmt_reference, spectral_form_factor

times = np.linspace(0.0, 2.0, 101)
lengths = np.linspace(0.0, 8.0, 41)
sff = spectral_form_factor(unfolded, times, window="hann", bootstrap=100, seed=7)
variance = number_variance(unfolded, lengths, samples=4096)
goe_sff = rmt_reference("spectral_form_factor", "goe", times)
goe_variance = rmt_reference("number_variance", "goe", lengths)
```

SFF time is normalized by the Heisenberg time and its plateau is one. Number
variance uses circular half-open counting windows. RMT references cover Poisson,
GOE, GUE, and CUE, including an optional finite-`N` CUE number-variance kernel.

## Quantum phase space

Torus coherent states and Husimi data share the position-basis and boundary-phase
conventions used by the quantum maps:

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

print(ulam.format, ulam.shape)  # csr, (1024, 1024)
print(ulam.escape_probabilities)  # zero for this closed system
```

Set `open_system=True` to retain escaped mass as substochastic column deficits.
Sampling uses independent per-cell child seeds, so changing `batch_size` does not
change the matrix. Source cells are the future parallelization boundary. Empty
geometric cells and `samples_per_cell=0` are rejected during validation.

Leading right or left eigenpairs are ordered by eigenvalue modulus and include
independently recomputed residuals:

```python
from chaos_numerics.operators import leading_eigenpairs, spectral_gap, stationary_density

spectrum = leading_eigenpairs(ulam, count=8, side="right", tolerance=1e-10)
density = stationary_density(ulam)
gap = spectral_gap(spectrum)

print(spectrum.eigenvalues, spectrum.residuals)
print(density.values.sum(), gap.values[0])
```

Run `python examples/cat_ulam_convergence.py --seed 7` for a reproducible Cat-map
study over partition resolutions and samples per cell.

## Experiments and parameter sweeps

Experiments combine a built-in model and analysis with explicit parameters and a
root seed. Sweeps derive stable per-job seeds, continue after individual failures,
and resume incomplete work without rerunning valid completed jobs:

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
    parameters=cartesian_grid(kick_strength=[2.0, 4.0], steps=[200, 400]),
    output="results/standard-map",
    resume=True,
)
```

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
history = evolve(hadamard, initial, steps=8, return_history=True)
spectrum = eigenstates(hadamard)

print(history.shape, spectrum.eigenphases, spectrum.residuals)
```

Boundary phases are represented in turns by `BoundaryPhases` and canonicalized
modulo one. Dense diagnostics use `||U.H U - I||_F / sqrt(N)`; matrix-free
eigensystems materialize only up to an explicit `dense_limit`.

## Design documents

- `docs/api-reference.md`: supported v0.1 public names
- `docs/tutorials/`: CI-executed quickstart and domain tutorials
- `docs/product/mvp-scope.md`: v0.1 product boundary
- `docs/design/public-api.md`: modules, public names, shapes, and dtypes
- `docs/design/numerical-standards.md`: accuracy, validation, and performance gates
- `docs/design/experiment-persistence.md`: execution, resume, and storage contract

## Citation and license

Citation metadata is available in `CITATION.cff`; release changes are recorded in
`CHANGELOG.md`. Chaos Numerics Library is distributed under the BSD 3-Clause
License. See `LICENSE` for the full terms.

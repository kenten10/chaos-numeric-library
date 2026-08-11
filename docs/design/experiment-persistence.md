# Experiment execution and persistence

Status: implemented baseline for KEN-128  
Applies to: Chaos Numerics Library v0.1  
Last reviewed: 2026-08-05

## Public workflow

`Experiment` declares a built-in model, analysis, JSON-compatible base parameters,
and an optional root seed. `run_experiment` executes one declaration;
`run_sweep` overlays each parameter row and persists a resumable collection.

<!-- docs-test: skip - writes a sweep directory to the filesystem -->
```python
from chaos_numerics.experiment import Experiment, cartesian_grid, run_sweep

experiment = Experiment(
    model="standard_map",
    analysis="lyapunov_spectrum",
    parameters={"initial_state": [0.1, 0.2], "transient": 100},
    seed=7,
)
summary = run_sweep(
    experiment,
    parameters=cartesian_grid(kick_strength=[2.0, 4.0], steps=[20_000, 40_000]),
    output="results/persistence-standard-map",
    resume=True,
)
```

Give every study its own `output` directory. A directory that already holds a
different experiment or grid is rejected rather than merged, so reusing one path
across documents and examples fails on the second run.

## Built-in registry

The registry has two families, and an analysis belongs to exactly one of them.

| Family | Models | Analyses | Result type |
| --- | --- | --- | --- |
| classical | `standard_map`, `cat_map`, `baker_map` | `iterate`/`trajectory` | `Trajectory` |
| | | `largest_lyapunov_exponent`, `lyapunov_spectrum` | `AnalysisResult` |
| | | `ulam_stationary_density` | `AnalysisResult` |
| quantum | `kicked_rotor`, `cylinder_kicked_rotor`, `quantum_cat_map`, `quantum_baker_map` | `eigenphases` | `AnalysisResult` |
| | | `eigenstates` | `EigenstateResult` |
| | | `spectral_form_factor`, `number_variance` | `SpectralCurve` |

Unknown names and unused parameters fail with `ValidationError`. Crossing the two
families — `kicked_rotor` with `lyapunov_spectrum`, say — is reported as a pairing
error that names the family of each side and lists what the model does support,
rather than as an unknown analysis or as a failure inside the classical routine.

Quantum model parameters are the constructor arguments spelled exactly as the
classes spell them: `dimension` (required), `kick_strength`, `effective_hbar`,
`matrix`, `boundary_phases`. `dimension` must be even for the cat and baker maps.
`dense_limit` guards the `O(N**2)` densification and defaults to
`DEFAULT_DENSE_LIMIT`; the sweep never raises it to `dimension` automatically,
because a sweep that quietly densifies a 4096-dimensional model is what the guard
exists to prevent.

`eigenstates` is in the registry even though eigenvectors cost `O(N**2)` on disk
(4 MiB of `complex128` at `N = 512`, against 4 KiB for the phases): a later
desymmetrization needs the basis, not just the phases, and the 256 MiB
uncompressed-archive budget already bounds the damage. Prefer `eigenphases` when
the basis is not needed.

### `boundary_phases` in JSON

`BoundaryPhases` is a dataclass and `Experiment.parameters` must survive
`json.dump`, so the parameter accepts three JSON spellings that all build the same
twist:

| JSON | Meaning |
| --- | --- |
| `0.25` | scalar shorthand for `alpha = beta = 0.25`, as the constructors accept |
| `[0.25, 0.13]` | `[position, momentum]`; the form to use in a grid |
| `{"position": 0.25, "momentum": 0.13}` | exactly `BoundaryPhases.to_dict()`, so a phase read back off a persisted result is reusable as-is |

Both phases are reduced modulo one by `BoundaryPhases`. Job IDs are hashed from
the parameters as written, not from the reduced twist, so `1.25` and `0.25` are two
distinct jobs that compute the identical spectrum. The duplicate-row check cannot
catch that; keep grid values inside `[0, 1)`.

`quantum_cat_map` accepts only `(0, 0)` and `quantum_baker_map` only
`(0.5, 0.5)` — their quantizations require it — so `boundary_phases` is a real
sweep axis for `kicked_rotor` alone. `cylinder_kicked_rotor` has no boundary
phases and takes `effective_hbar` instead; pairing it with a spectral statistic
warns with `NumericalWarning`, because its momentum lattice is a truncation of an
infinite one rather than a torus quantization and its level statistics belong to
the truncation as much as to the rotor.

### Spectral statistics are a chain

`spectral_form_factor` and `number_variance` need an unfolded spectrum, so the
quantum path runs the whole chain — `eigenstates`, optional `desymmetrize`,
`prepare_eigenphases`, `unfold` — and takes each stage's knobs from the
parameters: `parity_sector`, `symmetry_sector`, `unfold_method`, `unfold_degree`,
then `times`/`window`/`connected`/`bootstrap` or
`lengths`/`samples`/`bootstrap`/`finite_size_correction`. Every parameter is
validated before the diagonalization runs, so a misspelled `window` is a fast
refusal.

This is what lets the ensemble average the README requires for the ramp–plateau
figure be a sweep rather than a hand-written loop:

<!-- docs-test: skip - writes a sweep directory to the filesystem -->
```python
experiment = Experiment(
    model="kicked_rotor",
    analysis="spectral_form_factor",
    parameters={
        "dimension": 128,
        "kick_strength": 10.0,
        "symmetry_sector": "parity and antiunitary symmetry both broken",
        "times": list(np.linspace(0.05, 2.0, 79)),
    },
    seed=20240808,
)
summary = run_sweep(
    experiment,
    parameters=cartesian_grid(boundary_phases=[[0.20, 0.34], [0.67, 0.66], ...]),
    output="results/bloch-ensemble",
)
average = np.mean([run.result.values for run in summary.runs], axis=0)
```

Measured over twelve generic Bloch phases at `N = 128`, `K = 10`: the average
curve sits at 1.08 times the CUE reference over the ramp `0.1 <= tau <= 0.9`, its
plateau over `tau >= 1.2` is 0.98 against 1, and its RMS distance to CUE is 0.205
against 0.770 for a typical single member. `K(tau)` does not self-average, which
is why the single-member number stays large however big `N` gets.

### `symmetry_sector`

Omitting `symmetry_sector` makes `prepare_eigenphases` warn with
`NumericalWarning`, and the sweep passes that warning through untouched: it is
neither suppressed nor promoted to a refusal. Comparing a spectrum with
unresolved symmetries against RMT pulls every statistic toward Poisson, so the
diagnostic has to reach the caller; refusing the run outright would be wrong
because looking at the raw spectrum is a legitimate thing to do. The same
diagnostic is recorded in the persisted `metadata.warnings`, so a run made
without a sector label says so on disk as well.

`parity_sector` (`"even"` or `"odd"`) actually splits the spectrum with the
model's `symmetry_operators["parity"]` and supplies `symmetry_sector` as
`"parity-<sector>"` by default, so the desymmetrized path does not warn. A model
with no parity at the requested parameters — the kicked rotor at generic Bloch
phases, which breaks parity on purpose — refuses `parity_sector` with a message
that says so.

## Seed and resume policy

Each sweep job ID is the first 16 hexadecimal characters of SHA-256 over its
canonical merged parameters. A child seed is derived with `sha256-v1` from the
root seed and job ID, then reduced to `[0, 2**63)`. Parameter ordering, sweep
ordering, failure timing, and resume order therefore do not affect a job's seed.

Persisting an experiment without a root seed emits `ReproducibilityWarning`.
`strict_reproducibility=True` turns this into `ValidationError`. Failures retain
the parameter row, derived seed, exception type, and message. With the default
`continue_on_error=True`, later jobs continue. Resume skips valid successful jobs
and reruns failed, running, pending, or incomplete jobs. The manifest fingerprint
prevents a different experiment or grid from reusing the same directory.

## Storage format

Every successful run directory contains:

```text
metadata.json  # declaration, provenance, status, result metadata, array descriptors
arrays.npz     # numerical arrays only
```

A sweep adds `manifest.json` and stores runs below `jobs/<job-id>/`. Manifest and
run files are written through a same-directory temporary file and atomically
replaced. Runtime metadata includes Python, implementation, platform, package and
NumPy versions, elapsed runtime, and the Git commit when available.

The JSON schema version is `1`. JSON contains shapes and dtype names but never
embeds numerical result arrays. `load_result` validates the NPZ keys, shapes, and
dtypes against those descriptors before reconstructing the immutable core result.
Readers must reject unknown result types or incompatible descriptors. Additive
JSON fields may be ignored within schema version 1; removing or changing an
existing field requires a schema-version increment and migration design.

### Result payloads

`result.result_type` selects the restore branch. Every array is `float64` or
`complex128`; nothing else is accepted, which is what keeps `allow_pickle=False`
from being the only guard on a forged archive.

| `result_type` | Required arrays | Optional arrays |
| --- | --- | --- |
| `trajectory` | `states`, `initial_state` | — |
| `spectrum` | `eigenvalues` | `eigenvectors`, `residuals` |
| `eigenstate` | `eigenphases`, `eigenstates`, `residuals` | `convergence_history` |
| `analysis` | `values` (plus a JSON `name`) | `uncertainty`, `residuals`, `convergence_history` |
| `spectral_curve` | `x`, `values` | `uncertainty`, `variance` |

Adding a result type is a new restore branch and a new member of the `Result`
alias, not a schema-version bump: schema version `1` describes the directory
layout and the descriptor format, both of which are unchanged.

`spectral_curve` carries the `x`/`values` curve of `SpectralCurve` plus its two
optional error arrays. Absent is not the same as zero: a curve produced without
`bootstrap` omits `uncertainty` and `variance` from the archive entirely and comes
back with both as `None`, whereas `rmt_reference` deliberately stores exact zero
arrays. Writing zeros for a missing error bar would turn "no uncertainty was
estimated" into "the uncertainty is zero", which is the opposite claim.

NPZ/JSON is the minimal v0.1 format. Zarr, remote/object storage, concurrent
writers, and cross-version migrations are future work and must not silently alter
this directory contract.

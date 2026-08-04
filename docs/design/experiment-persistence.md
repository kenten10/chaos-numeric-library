# Experiment execution and persistence

Status: implemented baseline for KEN-128  
Applies to: Chaos Numerics Library v0.1  
Last reviewed: 2026-08-05

## Public workflow

`Experiment` declares a built-in model, analysis, JSON-compatible base parameters,
and an optional root seed. `run_experiment` executes one declaration;
`run_sweep` overlays each parameter row and persists a resumable collection.

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
    parameters=cartesian_grid(kick_strength=[2.0, 4.0], steps=[500, 1_000]),
    output="results/standard-map",
    resume=True,
)
```

The v0.1 built-in registry supports `standard_map`, `cat_map`, and `baker_map`
with `iterate`/`trajectory`, `largest_lyapunov_exponent`,
`lyapunov_spectrum`, and `ulam_stationary_density` where applicable. Unknown
names and unused parameters fail with `ValidationError`.

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

NPZ/JSON is the minimal v0.1 format. Zarr, remote/object storage, concurrent
writers, and cross-version migrations are future work and must not silently alter
this directory contract.

"""Performance regression suite implementing docs/design/numerical-standards.md section 12.

Three kinds of check live here, and they are blocking in different situations:

* Timing. Wall-clock medians are compared against recorded baseline medians with
  the dual gate from section 12.2. Shared CI runners are too noisy for that to be
  meaningful, so timing is reported and only fails the run when ``--check-timing``
  is passed, which is reserved for the pinned benchmark machine.
* Peak allocation. ``tracemalloc`` high-water marks catch the failure this suite
  exists to prevent: a path that must stay sparse or matrix-free quietly
  materializing a dense ``(N, N)`` array. Deterministic, so always blocking.
  Section 12.1 also asks for native peak RSS; that collector is still deferred
  (see section 14), and traced allocation is the portable stand-in.
* Scaling. Time ratios when one input size doubles, per section 12.3. Ratios use
  generous guardrails and are always blocking, because crossing them means an
  algorithmic change rather than runner noise.

The suite is dependency-free apart from the package itself.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import scipy  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

import chaos_numerics
from chaos_numerics.classical import CatMap, StandardMap, iterate
from chaos_numerics.operators import UniformPartition, build_ulam, leading_eigenpairs
from chaos_numerics.quantum import DenseUnitary, KickedRotor, eigenstates, evolve

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline.json"
SCHEMA_VERSION = 2

Mode = Literal["smoke", "standard"]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Section 12.2: aggregate short operations until one sample lasts this long, so
# that the gate never thresholds timer noise.
MINIMUM_SAMPLE_SECONDS = 0.100
MINIMUM_REPEATS_SHORT = 10
MINIMUM_REPEATS_LONG = 5
LONG_CASE_SECONDS = 1.0

# Section 12.2 regression gate. The specification pairs a 1.30x ratio with an
# absolute floor, and originally set that floor at 5 ms. Measured against these
# cases that floor is far above the noise it was meant to absorb and silences real
# regressions: `quantum_fft_evolve` has a median near 1.2 ms, so a 4.8x slowdown
# stays under 5 ms and passes. Short cases are already aggregated to at least
# MINIMUM_SAMPLE_SECONDS per sample, so the median is not timer-resolution limited,
# and 6 * MAD is the noise term that actually scales with the case. The floor is
# therefore reduced to a value that only guards against a degenerate zero MAD.
REGRESSION_RATIO = 1.30
REGRESSION_ABSOLUTE_SECONDS = 5e-5
REGRESSION_MAD_FACTOR = 6.0
WARNING_RATIO = 1.15


# ---------------------------------------------------------------- benchmark cases


def _standard_map_orbit(steps: int) -> Callable[[], None]:
    model = StandardMap(5.0)
    initial = np.array([0.1, 0.2])

    def run() -> None:
        iterate(model, initial, steps=steps)

    return run


def _standard_map_batch(batch: int, steps: int) -> Callable[[], None]:
    model = StandardMap(5.0)
    initial = np.random.default_rng(7).random((batch, 2))

    def run() -> None:
        iterate(model, initial, steps=steps)

    return run


def _ulam(cells: int, samples_per_cell: int) -> Callable[[], None]:
    partition = UniformPartition(((0.0, 1.0), (0.0, 1.0)), (cells, cells), periodic=(True, True))
    model = CatMap()

    def run() -> None:
        build_ulam(model, partition, samples_per_cell=samples_per_cell, seed=7)

    return run


def _fft_evolution(dimension: int, steps: int) -> Callable[[], None]:
    model = KickedRotor(dimension, 7.0)
    state = np.ones(dimension, dtype=np.complex128) / np.sqrt(dimension)

    def run() -> None:
        evolve(model, state, steps=steps, method="fft")

    return run


def _seeded_unitary(dimension: int) -> DenseUnitary:
    generator = np.random.default_rng(11)
    real = generator.standard_normal((dimension, dimension))
    imaginary = generator.standard_normal((dimension, dimension))
    factor, upper = np.linalg.qr(real + 1j * imaginary)
    phase = np.diag(upper).copy()
    phase /= np.abs(phase)
    return DenseUnitary(factor * phase[None, :], name="seeded_unitary")


def _dense_eigensystem(dimension: int) -> Callable[[], None]:
    model = _seeded_unitary(dimension)

    def run() -> None:
        eigenstates(model, dense_limit=dimension)

    return run


def _sparse_column_stochastic(dimension: int, per_column: int) -> csr_matrix:
    generator = np.random.default_rng(23)
    rows = generator.integers(0, dimension, size=(per_column, dimension))
    weights = generator.random((per_column, dimension))
    weights /= weights.sum(axis=0, keepdims=True)
    columns = np.repeat(np.arange(dimension), per_column)
    matrix = csr_matrix(
        (weights.T.ravel(), (rows.T.ravel(), columns)), shape=(dimension, dimension)
    )
    matrix.sum_duplicates()
    # Renormalize after duplicate collapse so the operator stays column-stochastic.
    sums = np.asarray(matrix.sum(axis=0)).ravel()
    scale = csr_matrix((1.0 / sums, (np.arange(dimension), np.arange(dimension))))
    return csr_matrix(matrix @ scale)


def _sparse_leading_eigenpairs(dimension: int, count: int) -> Callable[[], None]:
    matrix = _sparse_column_stochastic(dimension, per_column=9)

    def run() -> None:
        leading_eigenpairs(matrix, count=count)

    return run


@dataclass(frozen=True)
class Case:
    """One benchmark case with its size-dependent factory and allocation bound."""

    identifier: str
    build: Callable[[Mode], Callable[[], None]]
    # Upper bound on traced peak allocation, in bytes, as a function of the mode.
    allocation_bound: Callable[[Mode], int]
    allocation_note: str
    # Optional scaling pair: (label, doubled-size factory, maximum time ratio).
    scaling: (
        tuple[str, Callable[[Mode], tuple[Callable[[], None], Callable[[], None]]], float] | None
    ) = None


def _iterate_sizes(mode: Mode) -> int:
    return 100_000 if mode == "standard" else 20_000


def _batch_sizes(mode: Mode) -> tuple[int, int]:
    return (1_024, 1_000) if mode == "standard" else (256, 100)


def _ulam_sizes(mode: Mode) -> tuple[int, int]:
    return (64, 128) if mode == "standard" else (16, 64)


def _fft_sizes(mode: Mode) -> tuple[int, int]:
    return (16_384, 100) if mode == "standard" else (4_096, 20)


def _dense_size(mode: Mode) -> int:
    return 256 if mode == "standard" else 96


def _sparse_sizes(mode: Mode) -> tuple[int, int]:
    return (4_096, 8) if mode == "standard" else (1_024, 8)


CASES: tuple[Case, ...] = (
    Case(
        identifier="classical_scalar_iterate",
        build=lambda mode: _standard_map_orbit(_iterate_sizes(mode)),
        # states plus the defensive copy taken by Trajectory, and nothing more.
        allocation_bound=lambda mode: 6 * 8 * 2 * (_iterate_sizes(mode) + 1),
        allocation_note="no extra full-size copy beyond the stored trajectory",
        scaling=(
            "double steps",
            lambda mode: (
                _standard_map_orbit(_iterate_sizes(mode)),
                _standard_map_orbit(2 * _iterate_sizes(mode)),
            ),
            2.6,
        ),
    ),
    Case(
        identifier="classical_batch_iterate",
        build=lambda mode: _standard_map_batch(*_batch_sizes(mode)),
        allocation_bound=lambda mode: (
            6 * 8 * 2 * _batch_sizes(mode)[0] * (_batch_sizes(mode)[1] + 1)
        ),
        allocation_note="no extra full-size copy beyond the stored trajectory",
        scaling=(
            "double steps",
            lambda mode: (
                _standard_map_batch(*_batch_sizes(mode)),
                _standard_map_batch(_batch_sizes(mode)[0], 2 * _batch_sizes(mode)[1]),
            ),
            2.6,
        ),
    ),
    Case(
        identifier="ulam_build",
        build=lambda mode: _ulam(*_ulam_sizes(mode)),
        # A dense (cell_count, cell_count) array would need 8 * cells**4 bytes.
        # Allow generous slack for the CSR triplets and still reject that.
        allocation_bound=lambda mode: max(64 << 20, 8 * _ulam_sizes(mode)[0] ** 4 // 8),
        allocation_note="no (cell_count, cell_count) dense allocation",
        scaling=(
            "double samples_per_cell",
            lambda mode: (
                _ulam(*_ulam_sizes(mode)),
                _ulam(_ulam_sizes(mode)[0], 2 * _ulam_sizes(mode)[1]),
            ),
            2.8,
        ),
    ),
    Case(
        identifier="quantum_fft_evolve",
        build=lambda mode: _fft_evolution(*_fft_sizes(mode)),
        # Matrix-free evolution must stay O(N); an (N, N) complex array needs
        # 16 * N**2 bytes, which this bound is far below for the standard case.
        allocation_bound=lambda mode: max(8 << 20, 512 * _fft_sizes(mode)[0]),
        allocation_note="no (N, N) allocation in the matrix-free path",
        scaling=(
            "double N",
            lambda mode: (
                _fft_evolution(*_fft_sizes(mode)),
                _fft_evolution(2 * _fft_sizes(mode)[0], _fft_sizes(mode)[1]),
            ),
            2.8,
        ),
    ),
    Case(
        identifier="dense_eigensystem",
        build=lambda mode: _dense_eigensystem(_dense_size(mode)),
        # A dense eigensystem is inherently O(N**2); bound the number of
        # simultaneous N x N complex temporaries instead of forbidding them.
        allocation_bound=lambda mode: 16 * _dense_size(mode) ** 2 * 12,
        allocation_note="bounded number of simultaneous N x N temporaries",
    ),
    Case(
        identifier="sparse_leading_eigenpairs",
        build=lambda mode: _sparse_leading_eigenpairs(*_sparse_sizes(mode)),
        # A dense conversion would need 16 * N**2 bytes for complex128.
        allocation_bound=lambda mode: max(16 << 20, 16 * _sparse_sizes(mode)[0] ** 2 // 4),
        allocation_note="no dense conversion of the sparse operator",
        scaling=(
            "double N",
            lambda mode: (
                _sparse_leading_eigenpairs(*_sparse_sizes(mode)),
                _sparse_leading_eigenpairs(2 * _sparse_sizes(mode)[0], _sparse_sizes(mode)[1]),
            ),
            3.5,
        ),
    ),
)


# ---------------------------------------------------------------- measurement


@dataclass
class Timing:
    """Median, MAD, and IQR of a timing sample, per section 12.2."""

    median: float
    mad: float
    iqr: float
    inner_repeats: int
    outer_repeats: int
    samples: list[float] = field(repr=False, default_factory=list)


def _single_duration(function: Callable[[], None]) -> float:
    started = time.perf_counter()
    function()
    return time.perf_counter() - started


def _inner_repeats_for(function: Callable[[], None]) -> tuple[int, float]:
    """Choose an aggregation count so one sample lasts at least 100 ms."""
    probe = _single_duration(function)
    if probe >= MINIMUM_SAMPLE_SECONDS:
        return 1, probe
    repeats = max(1, int(np.ceil(MINIMUM_SAMPLE_SECONDS / max(probe, 1e-9))))
    return min(repeats, 10_000), probe


def measure_ratio(small: Callable[[], None], large: Callable[[], None]) -> float:
    """Return the median of interleaved size ratios.

    Timing the two sizes one after the other lets a load spike on the machine land
    entirely on one of them: measured on a contended laptop, a case whose true ratio
    is 2.0 reported 3.5 and failed a blocking gate. Alternating the two sizes and
    taking the median of per-pair ratios makes the ratio depend on the algorithm
    rather than on when each half happened to run.
    """
    small()
    large()
    small_inner, small_probe = _inner_repeats_for(small)
    large_inner, _ = _inner_repeats_for(large)
    outer = MINIMUM_REPEATS_SHORT if small_probe < LONG_CASE_SECONDS else MINIMUM_REPEATS_LONG
    ratios: list[float] = []
    for _ in range(outer):
        small_seconds = _aggregate(small, small_inner)
        large_seconds = _aggregate(large, large_inner)
        ratios.append(large_seconds / small_seconds)
    return statistics.median(ratios)


def _aggregate(function: Callable[[], None], repeats: int) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - started) / repeats


def measure(function: Callable[[], None]) -> Timing:
    function()  # warm up imports, FFT plans, and any lazy allocation
    inner, probe = _inner_repeats_for(function)
    outer = MINIMUM_REPEATS_SHORT if probe < LONG_CASE_SECONDS else MINIMUM_REPEATS_LONG
    samples: list[float] = []
    for _ in range(outer):
        started = time.perf_counter()
        for _ in range(inner):
            function()
        samples.append((time.perf_counter() - started) / inner)
    median = statistics.median(samples)
    mad = statistics.median([abs(sample - median) for sample in samples])
    if len(samples) >= 4:
        first, third = np.percentile(samples, [25.0, 75.0])
        iqr = float(third - first)
    else:  # pragma: no cover - outer repeats are always >= 5
        iqr = 0.0
    return Timing(median, mad, iqr, inner, outer, samples)


def peak_allocation(function: Callable[[], None]) -> int:
    """Return the traced peak allocation of one call, in bytes."""
    function()  # keep one-off caches out of the measurement
    tracemalloc.start()
    try:
        function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


# ---------------------------------------------------------------- reporting


def _runner_class() -> str:
    """Coarse identity of the machine, used to refuse cross-runner comparisons."""
    return f"{platform.system()}-{platform.machine()}"


def _environment() -> dict[str, object]:
    threads = {name: os.environ[name] for name in THREAD_VARIABLES if name in os.environ}
    return {
        "chaos_numerics": chaos_numerics.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "runner_class": _runner_class(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": threads,
    }


def _load_baseline() -> dict[str, object]:
    if not BASELINE.exists():  # pragma: no cover - the file is committed
        raise SystemExit(f"missing baseline file {BASELINE}")
    baseline: dict[str, object] = json.loads(BASELINE.read_text(encoding="utf-8"))
    version = baseline.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SystemExit(
            f"baseline schema_version {version!r} does not match {SCHEMA_VERSION}; "
            "regenerate it with --update-baseline on the pinned runner"
        )
    return baseline


def _timing_verdict(timing: Timing, reference: Mapping[str, float] | None) -> tuple[str, str]:
    if reference is None:
        return "NEW", "no baseline entry"
    base_median = float(reference["median_seconds"])
    base_mad = float(reference.get("mad_seconds", 0.0))
    ratio = timing.median / base_median if base_median > 0 else float("inf")
    excess = timing.median - base_median
    threshold = max(REGRESSION_ABSOLUTE_SECONDS, REGRESSION_MAD_FACTOR * base_mad)
    detail = f"ratio={ratio:.2f} excess={excess * 1e3:+.2f}ms gate={threshold * 1e3:.2f}ms"
    if ratio > REGRESSION_RATIO and excess > threshold:
        return "REGRESSION", detail
    if ratio > WARNING_RATIO:
        return "WARN", detail
    return "OK", detail


def run(mode: Mode, *, include_scaling: bool) -> dict[str, object]:
    results: dict[str, object] = {}
    for case in CASES:
        function = case.build(mode)
        timing = measure(function)
        peak = peak_allocation(function)
        bound = case.allocation_bound(mode)
        entry: dict[str, object] = {
            "median_seconds": timing.median,
            "mad_seconds": timing.mad,
            "iqr_seconds": timing.iqr,
            "inner_repeats": timing.inner_repeats,
            "outer_repeats": timing.outer_repeats,
            "peak_allocation_bytes": peak,
            "peak_allocation_bound_bytes": bound,
            "allocation_note": case.allocation_note,
        }
        if include_scaling and case.scaling is not None:
            label, factory, limit = case.scaling
            small, large = factory(mode)
            ratio = measure_ratio(small, large)
            entry["scaling"] = {
                "check": label,
                "time_ratio": ratio,
                "maximum_time_ratio": limit,
            }
        results[case.identifier] = entry
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "standard"),
        default="smoke",
        help="smoke sizes for shared runners, standard sizes for the pinned runner",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail on a blocking allocation or scaling violation",
    )
    parser.add_argument(
        "--check-timing",
        action="store_true",
        help="also fail on a timing regression; only meaningful on the pinned runner",
    )
    parser.add_argument(
        "--skip-scaling",
        action="store_true",
        help="skip the scaling pairs, which roughly triples the runtime",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite benchmarks/baseline.json from this run; never do this in CI",
    )
    parser.add_argument("--json", type=Path, help="also write the full report to this path")
    args = parser.parse_args(argv)

    mode: Mode = args.mode
    if args.check_timing and not any(name in os.environ for name in THREAD_VARIABLES):
        print(
            "warning: --check-timing without a pinned thread count; "
            f"set one of {', '.join(THREAD_VARIABLES)}",
            file=sys.stderr,
        )

    results = run(mode, include_scaling=not args.skip_scaling)
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "environment": _environment(),
        "cases": results,
    }

    if args.update_baseline:
        BASELINE.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {BASELINE}")
        return 0

    baseline = _load_baseline()
    reference_cases = baseline.get("cases", {})
    baseline_environment = baseline.get("environment")
    baseline_runner = (
        baseline_environment.get("runner_class")
        if isinstance(baseline_environment, Mapping)
        else None
    )
    # Section 12.2: only compare results from the same runner class. A recorded
    # baseline from another machine says nothing about this one.
    same_runner = baseline_runner == _runner_class()
    comparable = baseline.get("mode") == mode and same_runner
    blocking: list[str] = []
    regressions: list[str] = []

    for identifier, entry in results.items():
        assert isinstance(entry, dict)
        timing = Timing(
            float(entry["median_seconds"]),
            float(entry["mad_seconds"]),
            float(entry["iqr_seconds"]),
            int(entry["inner_repeats"]),
            int(entry["outer_repeats"]),
        )
        reference = None
        if comparable and isinstance(reference_cases, Mapping):
            candidate = reference_cases.get(identifier)
            if isinstance(candidate, Mapping):
                reference = {
                    "median_seconds": float(candidate["median_seconds"]),
                    "mad_seconds": float(candidate.get("mad_seconds", 0.0)),
                }
        verdict, detail = _timing_verdict(timing, reference)
        print(
            f"{identifier:26} median={timing.median * 1e3:9.3f}ms "
            f"mad={timing.mad * 1e3:7.3f}ms n={timing.outer_repeats}x{timing.inner_repeats} "
            f"{verdict:10} {detail}"
        )
        if verdict == "REGRESSION":
            regressions.append(identifier)

        peak = int(entry["peak_allocation_bytes"])
        bound = int(entry["peak_allocation_bound_bytes"])
        allocation_ok = peak <= bound
        print(
            f"{'':26} peak={peak / 2**20:8.2f}MiB bound={bound / 2**20:8.2f}MiB "
            f"{'OK' if allocation_ok else 'EXCEEDED':10} {entry['allocation_note']}"
        )
        if not allocation_ok:
            blocking.append(f"{identifier}: peak allocation {peak} > {bound}")

        scaling = entry.get("scaling")
        if isinstance(scaling, Mapping):
            ratio = float(scaling["time_ratio"])
            limit = float(scaling["maximum_time_ratio"])
            scaling_ok = ratio <= limit
            print(
                f"{'':26} scaling[{scaling['check']}] ratio={ratio:.2f} "
                f"limit={limit:.2f} {'OK' if scaling_ok else 'EXCEEDED'}"
            )
            if not scaling_ok:
                blocking.append(f"{identifier}: scaling ratio {ratio:.2f} > {limit:.2f}")

    if not comparable:
        print(
            f"\nnote: baseline was recorded in mode {baseline.get('mode')!r} on runner class "
            f"{baseline_runner!r}; this run is mode {mode!r} on {_runner_class()!r}, so the "
            "timing comparison is skipped. Allocation and scaling gates still apply."
        )

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failures = list(blocking)
    if args.check_timing:
        failures.extend(f"{name}: timing regression" for name in regressions)
    elif regressions:
        print(f"\ntiming regressions reported but not blocking: {', '.join(regressions)}")

    if args.check and failures:
        print("\n" + "\n".join(f"FAIL {failure}" for failure in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

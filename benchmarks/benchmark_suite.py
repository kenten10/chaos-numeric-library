"""Small, dependency-free performance regression suite for CI and releases."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from chaos_numerics.classical import CatMap, StandardMap, iterate
from chaos_numerics.operators import UniformPartition, build_ulam
from chaos_numerics.quantum import KickedRotor, QuantumBakerMap, eigenstates, evolve

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "baseline.json"


def _scalar_trajectory() -> None:
    iterate(StandardMap(5.0), np.array([0.1, 0.2]), steps=20_000)


def _batch_trajectory() -> None:
    generator = np.random.default_rng(7)
    iterate(StandardMap(5.0), generator.random((1_000, 2)), steps=100)


def _ulam_build() -> None:
    partition = UniformPartition(((0.0, 1.0), (0.0, 1.0)), (8, 8), periodic=(True, True))
    build_ulam(CatMap(), partition, samples_per_cell=64, seed=7)


def _fft_evolution() -> None:
    model = KickedRotor(8_192, 7.0)
    state = np.ones(model.dimension, dtype=np.complex128) / np.sqrt(model.dimension)
    evolve(model, state, steps=20, method="fft")


def _dense_eigenanalysis() -> None:
    eigenstates(QuantumBakerMap(64))


CASES: Mapping[str, Callable[[], None]] = {
    "scalar_trajectory": _scalar_trajectory,
    "batch_trajectory": _batch_trajectory,
    "ulam_build": _ulam_build,
    "fft_evolution": _fft_evolution,
    "dense_eigenanalysis": _dense_eigenanalysis,
}


def _measure(function: Callable[[], None], *, repeats: int) -> float:
    function()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when a threshold is exceeded")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    thresholds = baseline["thresholds_seconds"]
    failures: list[str] = []
    for name, function in CASES.items():
        duration = _measure(function, repeats=args.repeats)
        threshold = float(thresholds[name])
        outcome = "PASS" if duration <= threshold else "FAIL"
        print(f"{name:24} median={duration:.6f}s threshold={threshold:.3f}s {outcome}")
        if duration > threshold:
            failures.append(name)
    if args.check and failures:
        raise SystemExit(f"performance thresholds exceeded: {', '.join(failures)}")


if __name__ == "__main__":
    main()

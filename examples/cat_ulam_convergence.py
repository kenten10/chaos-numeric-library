"""Reproducible Cat-map Ulam resolution and sampling study.

Run from an editable installation:
    python examples/cat_ulam_convergence.py --seed 7
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from chaos_numerics.classical import CatMap
from chaos_numerics.operators import (
    UniformPartition,
    build_ulam,
    leading_eigenpairs,
    spectral_gap,
    stationary_density,
)


def run_study(
    resolutions: Sequence[int],
    sample_counts: Sequence[int],
    *,
    seed: int,
) -> list[dict[str, int | float]]:
    """Evaluate fixed-seed Ulam diagnostics over resolution/sample pairs."""
    rows: list[dict[str, int | float]] = []
    for resolution in resolutions:
        partition = UniformPartition(
            bounds=((0.0, 1.0), (0.0, 1.0)),
            shape=(resolution, resolution),
            periodic=(True, True),
        )
        for samples in sample_counts:
            matrix = build_ulam(
                CatMap(),
                partition,
                samples_per_cell=samples,
                seed=seed,
                batch_size=min(samples, 256),
            )
            spectrum = leading_eigenpairs(matrix, count=4)
            density = stationary_density(matrix)
            gap = spectral_gap(spectrum)
            rows.append(
                {
                    "resolution": resolution,
                    "cells": partition.size,
                    "samples_per_cell": samples,
                    "leading_eigenvalue_error": float(abs(spectrum.eigenvalues[0] - 1.0)),
                    "max_eigenpair_residual": float(np.max(spectrum.residuals)),
                    "stationary_invariance_l1": float(
                        np.linalg.norm(matrix @ density.values - density.values, ord=1)
                    ),
                    "spectral_gap": float(gap.values[0]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--samples", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_study(arguments.resolutions, arguments.samples, seed=arguments.seed),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

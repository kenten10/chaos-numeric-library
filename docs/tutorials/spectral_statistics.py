"""Spectral statistics: prepare, unfold, and summarize resolved eigenphases."""

import numpy as np

from chaos_numerics.spectral import adjacent_gap_ratios, prepare_eigenphases, unfold

generator = np.random.default_rng(7)
raw_phases = np.sort(generator.uniform(-np.pi, np.pi, size=128))
prepared = prepare_eigenphases(raw_phases, symmetry_sector="synthetic-single-sector")
unfolded = unfold(prepared, method="polynomial", degree=3)
ratios = adjacent_gap_ratios(unfolded)

assert np.isclose(np.mean(unfolded.spacings), 1.0)
print(unfolded.count, ratios.metadata.parameters["mean_ratio"])

"""Prepare, unfold, and summarize one symmetry-resolved quantum spectrum."""

import numpy as np

from chaos_numerics import QuantumBakerMap
from chaos_numerics.quantum import eigenstates
from chaos_numerics.spectral import (
    adjacent_gap_ratios,
    prepare_eigenphases,
    spacing_distribution,
    unfold,
)

model = QuantumBakerMap(dimension=128)
eigensystem = eigenstates(model)
parity = model.symmetry_operators["parity"]
parity_expectation = np.einsum(
    "ij,ij->j",
    eigensystem.eigenstates.conj(),
    parity @ eigensystem.eigenstates,
).real
raw = eigensystem.eigenphases[parity_expectation > 0.0]
prepared = prepare_eigenphases(raw, symmetry_sector="parity-even")
unfolded = unfold(prepared, method="polynomial", degree=5)
ratios = adjacent_gap_ratios(unfolded)
distribution = spacing_distribution(unfolded, bins=30, value_range=(0.0, 4.0))

print(f"levels={unfolded.count}, mean_spacing={np.mean(unfolded.spacings):.12f}")
print(f"mean_gap_ratio={ratios.metadata.parameters['mean_ratio']:.6f}")
print(f"histogram_integral={np.sum(distribution.values * np.diff(distribution.bin_edges)):.12f}")

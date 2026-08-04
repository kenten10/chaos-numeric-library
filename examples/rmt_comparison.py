"""Produce plot-ready baker-map and GOE long-range spectral curves."""

import numpy as np

from chaos_numerics import QuantumBakerMap
from chaos_numerics.quantum import eigenstates
from chaos_numerics.spectral import (
    number_variance,
    prepare_eigenphases,
    rmt_reference,
    spectral_form_factor,
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
prepared = prepare_eigenphases(
    eigensystem.eigenphases[parity_expectation > 0.0],
    symmetry_sector="parity-even",
)
unfolded = unfold(prepared, method="mean")

times = np.linspace(0.0, 2.0, 101)
lengths = np.linspace(0.0, 8.0, 41)
form_factor = spectral_form_factor(unfolded, times, window="hann", bootstrap=100, seed=7)
variance = number_variance(unfolded, lengths, samples=4096)
goe_form_factor = rmt_reference("spectral_form_factor", "goe", times)
goe_variance = rmt_reference("number_variance", "goe", lengths)

print(form_factor.x.shape, form_factor.values.shape, form_factor.uncertainty.shape)
print(variance.x.shape, variance.values.shape, variance.uncertainty.shape)
print(goe_form_factor.values.shape, goe_variance.values.shape)

"""Generate plot-ready kicked-rotor Husimi and classical overlay data."""

import numpy as np

from chaos_numerics.classical import StandardMap, iterate
from chaos_numerics.quantum import (
    KickedRotor,
    coherent_state,
    evolve,
    husimi_distribution,
    inverse_participation_ratio,
)

dimension = 64
initial = coherent_state(dimension=dimension, position=0.2, momentum=0.3)
quantum = KickedRotor(dimension=dimension, kick_strength=5.0)
evolved = evolve(quantum, initial, steps=8, method="fft")
husimi = husimi_distribution(evolved, grid_shape=(64, 64))

classical = StandardMap(kick_strength=5.0)
trajectory = iterate(classical, np.asarray([0.2, 0.3]), steps=8)

print(husimi.grid_points.shape, husimi.values.shape, husimi.integral)
print(trajectory.states.shape, inverse_participation_ratio(evolved))

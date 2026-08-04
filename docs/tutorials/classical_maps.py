"""Classical maps: batched trajectories and a finite-time Lyapunov spectrum."""

import numpy as np

from chaos_numerics.classical import CatMap, StandardMap, iterate, lyapunov_spectrum

model = StandardMap(kick_strength=5.0)
initial = np.array([[0.1, 0.2], [0.3, 0.4]])
trajectory = iterate(model, initial, steps=50)
spectrum = lyapunov_spectrum(CatMap(), initial[0], steps=100, transient=100)

assert trajectory.shape == (2, 51, 2)
assert spectrum.values.shape == (2,)
print(trajectory.shape, spectrum.values)

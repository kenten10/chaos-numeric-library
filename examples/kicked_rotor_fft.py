"""Evolve a kicked rotor without allocating its dense Floquet matrix."""

import numpy as np

from chaos_numerics.quantum import KickedRotor, basis_state, evolve

model = KickedRotor(dimension=4096, kick_strength=8.0, boundary_phase=0.0)
initial = basis_state(dimension=model.dimension, index=0)
final = evolve(model, initial, steps=100, method="fft")

print(f"dimension={model.dimension}, norm_error={abs(np.linalg.norm(final) - 1.0):.3e}")

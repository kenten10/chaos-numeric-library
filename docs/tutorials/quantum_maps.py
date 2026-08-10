"""Quantum maps: FFT evolution and dense-reference eigensystem analysis."""

import numpy as np

from chaos_numerics.quantum import KickedRotor, basis_state, eigenstates, evolve

rotor = KickedRotor(dimension=128, kick_strength=7.0)
initial = basis_state(dimension=rotor.dimension, index=0)
evolved = evolve(rotor, initial, steps=20, method="fft").final_state
eigensystem = eigenstates(KickedRotor(dimension=32, kick_strength=7.0))

assert np.isclose(np.linalg.norm(evolved), 1.0)
assert eigensystem.count == 32
print(np.linalg.norm(evolved), eigensystem.eigenphases[:4])

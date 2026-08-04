"""Ulam method: build a transfer matrix and obtain its stationary density."""

import numpy as np

from chaos_numerics.classical import CatMap
from chaos_numerics.operators import UniformPartition, build_ulam, stationary_density

partition = UniformPartition(
    ((0.0, 1.0), (0.0, 1.0)),
    (4, 4),
    periodic=(True, True),
)
operator = build_ulam(CatMap(), partition, samples_per_cell=64, seed=7)
density = stationary_density(operator)

assert operator.shape == (16, 16)
assert np.isclose(np.sum(density.values), 1.0)
print(operator.shape, density.values[:4])

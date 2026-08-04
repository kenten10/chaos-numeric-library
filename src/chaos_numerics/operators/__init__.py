"""Partitions, transfer-operator approximations, and sparse eigensolvers."""

from chaos_numerics.operators.eigensolvers import (
    leading_eigenpairs,
    spectral_gap,
    stationary_density,
)
from chaos_numerics.operators.partitions import RectangularPartition, UniformPartition
from chaos_numerics.operators.ulam import UlamMatrix, build_ulam

__all__ = [
    "RectangularPartition",
    "UlamMatrix",
    "UniformPartition",
    "build_ulam",
    "leading_eigenpairs",
    "spectral_gap",
    "stationary_density",
]

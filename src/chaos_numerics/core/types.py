"""NumPy type aliases and the runtime shape conventions they accompany.

NumPy's current static typing does not encode array shapes. ``ClassicalState``
and related aliases therefore express dtype and semantic intent; public boundary
validators enforce the documented trailing-axis shapes at runtime.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike as ArrayLike
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
IndexArray: TypeAlias = NDArray[np.intp]
BoolArray: TypeAlias = NDArray[np.bool_]

# State coordinates or amplitudes occupy the final axis. The scalar and batched
# aliases intentionally share a runtime type because NumPy typing cannot yet
# distinguish ``(d,)`` from ``(*batch, d)``.
ClassicalState: TypeAlias = FloatArray
ClassicalBatch: TypeAlias = FloatArray
QuantumState: TypeAlias = ComplexArray
QuantumBatch: TypeAlias = ComplexArray
OperatorArray: TypeAlias = ComplexArray

Shape: TypeAlias = tuple[int, ...]
StateShape: TypeAlias = tuple[int]
BatchShape: TypeAlias = tuple[int, ...]
OperatorShape: TypeAlias = tuple[int, int]

__all__ = [
    "ArrayLike",
    "BatchShape",
    "BoolArray",
    "ClassicalBatch",
    "ClassicalState",
    "ComplexArray",
    "FloatArray",
    "IndexArray",
    "OperatorArray",
    "OperatorShape",
    "QuantumBatch",
    "QuantumState",
    "Shape",
    "StateShape",
]

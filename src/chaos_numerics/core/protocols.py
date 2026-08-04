"""Structural protocols shared by numerical domains.

Protocols describe the minimum behavior consumed by algorithms. User-defined
models conform structurally and never need to inherit from a library base class.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np

from chaos_numerics.core.types import (
    ArrayLike,
    ComplexArray,
    FloatArray,
    IndexArray,
    OperatorShape,
)


@runtime_checkable
class ClassicalMap(Protocol):
    """A discrete classical map with a state Jacobian."""

    @property
    def state_dim(self) -> int:
        """Number of coordinates in one state."""
        ...

    @property
    def is_periodic(self) -> tuple[bool, ...]:
        """Per-coordinate periodicity flags."""
        ...

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Per-coordinate lower and upper bounds."""
        ...

    def step(self, state: ArrayLike, /) -> FloatArray:
        """Advance scalar or batched states by one discrete step."""
        ...

    def jacobian(self, state: ArrayLike, /) -> FloatArray:
        """Evaluate the state Jacobian for scalar or batched states."""
        ...


@runtime_checkable
class Flow(Protocol):
    """A continuous-time vector field contract without an integration policy."""

    @property
    def state_dim(self) -> int:
        """Number of coordinates in one state."""
        ...

    def vector_field(self, time: float, state: ArrayLike, /) -> FloatArray:
        """Evaluate the vector field at a time and scalar or batched states."""
        ...

    def jacobian(self, time: float, state: ArrayLike, /) -> FloatArray:
        """Evaluate the vector-field Jacobian."""
        ...


@runtime_checkable
class LinearOperatorLike(Protocol):
    """The matrix-free operation required by v0.1 algorithms."""

    @property
    def shape(self) -> OperatorShape:
        """Two-dimensional operator shape."""
        ...

    @property
    def dtype(self) -> np.dtype[Any]:
        """Operator element dtype."""
        ...

    def matvec(self, vector: ArrayLike, /) -> ComplexArray:
        """Apply the operator to one vector."""
        ...


@runtime_checkable
class QuantumMap(Protocol):
    """A finite-dimensional quantum map with matrix-free application."""

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension."""
        ...

    def apply(self, state: ArrayLike, /) -> ComplexArray:
        """Apply one quantum-map step to scalar or batched states."""
        ...

    def as_linear_operator(self) -> LinearOperatorLike:
        """Return a matrix-free representation of this map."""
        ...


@runtime_checkable
class Partition(Protocol):
    """A finite partition used by transfer-operator approximations."""

    @property
    def ndim(self) -> int:
        """Number of partitioned coordinates."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Cell counts in coordinate order."""
        ...

    @property
    def size(self) -> int:
        """Total number of cells."""
        ...

    def locate(self, points: ArrayLike, /) -> IndexArray:
        """Return C-order flat cell indices for scalar or batched points."""
        ...

    def ravel_index(self, multi_index: ArrayLike, /) -> IndexArray:
        """Convert scalar or batched multi-indices to C-order flat indices."""
        ...

    def unravel_index(self, flat_index: ArrayLike, /) -> IndexArray:
        """Convert scalar or batched flat indices to trailing-axis multi-indices."""
        ...

    def cell_bounds(self, index: int, /) -> FloatArray:
        """Return ``(ndim, 2)`` lower and upper bounds for one cell."""
        ...

    def sample(
        self,
        index: int,
        count: int,
        *,
        seed: int | None = None,
    ) -> FloatArray:
        """Draw ``count`` reproducible samples from one cell."""
        ...


__all__ = [
    "ClassicalMap",
    "Flow",
    "LinearOperatorLike",
    "Partition",
    "QuantumMap",
]

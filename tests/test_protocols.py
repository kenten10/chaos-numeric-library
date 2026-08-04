"""Runtime and static conformance examples for core protocols."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chaos_numerics.core import (
    ClassicalMap,
    ComplexArray,
    FloatArray,
    Flow,
    IndexArray,
    LinearOperatorLike,
    Partition,
    QuantumMap,
)


@dataclass(frozen=True)
class DummyClassicalMap:
    """A structural ``ClassicalMap`` implementation with no inheritance."""

    state_dim: int = 2
    is_periodic: tuple[bool, ...] = (True, True)
    bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))

    def step(self, state: ArrayLike, /) -> FloatArray:
        return np.asarray(state, dtype=np.float64)

    def jacobian(self, state: ArrayLike, /) -> FloatArray:
        state_array = np.asarray(state)
        return np.broadcast_to(np.eye(self.state_dim), (*state_array.shape[:-1], 2, 2)).copy()


@dataclass(frozen=True)
class DummyFlow:
    """A structural ``Flow`` implementation."""

    state_dim: int = 2

    def vector_field(self, time: float, state: ArrayLike, /) -> FloatArray:
        del time
        return np.zeros_like(np.asarray(state), dtype=np.float64)

    def jacobian(self, time: float, state: ArrayLike, /) -> FloatArray:
        del time
        state_array = np.asarray(state)
        return np.zeros((*state_array.shape[:-1], 2, 2), dtype=np.float64)


@dataclass(frozen=True)
class DummyLinearOperator:
    """A structural matrix-free identity operator."""

    shape: tuple[int, int] = (4, 4)
    dtype: np.dtype[Any] = field(default_factory=lambda: np.dtype(np.complex128))

    def matvec(self, vector: ArrayLike, /) -> ComplexArray:
        return np.asarray(vector, dtype=np.complex128)


@dataclass(frozen=True)
class DummyQuantumMap:
    """A structural ``QuantumMap`` implementation."""

    dimension: int = 4

    def apply(self, state: ArrayLike, /) -> ComplexArray:
        return np.asarray(state, dtype=np.complex128)

    def as_linear_operator(self) -> LinearOperatorLike:
        return DummyLinearOperator()


@dataclass(frozen=True)
class DummyPartition:
    """A structural one-dimensional ``Partition`` implementation."""

    ndim: int = 1
    shape: tuple[int, ...] = (4,)
    size: int = 4

    def locate(self, points: ArrayLike, /) -> IndexArray:
        values = np.asarray(points, dtype=np.float64)
        return np.floor(values[..., 0] * self.size).astype(np.intp)

    def ravel_index(self, multi_index: ArrayLike, /) -> IndexArray:
        values = np.asarray(multi_index, dtype=np.intp)
        return values[..., 0]

    def unravel_index(self, flat_index: ArrayLike, /) -> IndexArray:
        values = np.asarray(flat_index, dtype=np.intp)
        return values[..., None]

    def cell_bounds(self, index: int, /) -> FloatArray:
        return np.array([[index / self.size, (index + 1) / self.size]], dtype=np.float64)

    def sample(
        self,
        index: int,
        count: int,
        *,
        seed: int | None = None,
    ) -> FloatArray:
        generator = np.random.default_rng(seed)
        lower, upper = self.cell_bounds(index)[0]
        return generator.uniform(lower, upper, size=(count, 1))


# These assignments are compile-time conformance tests under mypy strict mode.
CLASSICAL_MAP: ClassicalMap = DummyClassicalMap()
FLOW: Flow = DummyFlow()
LINEAR_OPERATOR: LinearOperatorLike = DummyLinearOperator()
QUANTUM_MAP: QuantumMap = DummyQuantumMap()
PARTITION: Partition = DummyPartition()


def test_representative_implementations_satisfy_runtime_protocols() -> None:
    """Representative implementations conform without inheriting library classes."""
    assert isinstance(CLASSICAL_MAP, ClassicalMap)
    assert isinstance(FLOW, Flow)
    assert isinstance(LINEAR_OPERATOR, LinearOperatorLike)
    assert isinstance(QUANTUM_MAP, QuantumMap)
    assert isinstance(PARTITION, Partition)


def test_incomplete_implementation_fails_runtime_protocol() -> None:
    """Runtime protocol checks reject objects missing required members."""

    class IncompleteMap:
        state_dim = 2

    assert not isinstance(IncompleteMap(), ClassicalMap)

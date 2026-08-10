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
    """A continuous-time vector field contract without an integration policy.

    **v0.1 ships no implementation of this protocol, deliberately.** It is the only
    protocol in :mod:`chaos_numerics.core` with no concrete model or algorithm
    behind it, and it exists so that a solver can be added later without changing
    ``core``. Continuous-time flows are the named v0.2 candidate in
    ``docs/product/mvp-scope.md``; the reason they are not here is that the open
    questions -- integrator family and error control, whether the variational
    equations travel with the state, how event detection meets a fixed output grid
    -- are policy decisions that a design document settles and an implementation
    does not.

    Do not mistake :func:`~chaos_numerics.classical.poincare_section` for the
    missing piece: it is the stroboscopic section of a discrete map, which needs no
    integrator and no event detection.
    """

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
    """The matrix-free operation required by v0.1 algorithms.

    Result dtype
    ------------
    :meth:`matvec` returns ``complex128`` for every operator, including a real
    one. That is deliberate and it is frozen for v0.1. :attr:`dtype` still reports
    the operator's own element dtype, so nothing is hidden: a
    :class:`~chaos_numerics.operators.UlamMatrix` reports ``float64`` there and
    offers ``.matrix`` and ``@`` for the natural ``float64`` product.

    A single result dtype is chosen because this protocol exists to feed spectral
    analysis, where the answer is complex whatever the operator is: the spectrum of
    a real non-symmetric transfer operator is complex, and its eigenvectors are
    mixed with complex quantum states downstream. One dtype means one code path
    instead of a real branch and a complex branch that must agree.

    Making the parameter explicit -- ``Protocol[ScalarT]`` with
    ``np.ndarray[Any, np.dtype[ScalarT]]`` -- was implemented and measured before
    this was written down, and it does type-check: mypy accepts the covariant
    parameter, correctly rejects a ``complex128`` operator where
    ``LinearOperatorLike[np.float64]`` is asked for, and ``isinstance`` keeps
    working against the unparametrized form. It was not adopted, for three reasons
    found by running it:

    - Under ``disallow_any_generics``, which ``mypy --strict`` turns on for this
      package and for its users, a bare ``LinearOperatorLike`` annotation stops
      being valid and every use site has to spell a parameter. A PEP 696 default
      would keep the bare form working, but ``typing.TypeVar(default=...)`` needs
      Python 3.13 and this package supports 3.11, so it would mean taking
      ``typing_extensions`` as a runtime dependency for a three-member protocol.
    - Neither implementation the parameter would describe can carry it today.
      ``UlamMatrix.dtype`` is annotated ``np.dtype[Any]``, so the parameter is
      inferred as ``Any`` from the one real operator in the library, and the
      quantum models return :class:`scipy.sparse.linalg.LinearOperator`, which
      ships no type stubs and is therefore ``Any`` in its entirety -- declaring a
      parametrized return type for it produces ``no-any-return`` under strict.
    - No algorithm in v0.1 consumes this protocol. The eigensolvers take a dense
      array or a sparse matrix and reject matrix-free operators outright, so the
      parameter would buy no internal type safety, only annotation churn.

    Widening the result instead, to ``np.ndarray[Any, np.dtype[np.number[Any]]]``,
    type-checks across the package unchanged, but it weakens what every caller
    gets back without changing what any operator actually returns, so it was not
    adopted either.
    """

    @property
    def shape(self) -> OperatorShape:
        """Two-dimensional operator shape."""
        ...

    @property
    def dtype(self) -> np.dtype[Any]:
        """Operator element dtype, which is not the dtype :meth:`matvec` returns."""
        ...

    def matvec(self, vector: ArrayLike, /) -> ComplexArray:
        """Apply the operator to one vector, returning ``complex128`` always."""
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


@runtime_checkable
class SerializableResult(Protocol):
    """A result container that can be split into arrays and JSON metadata.

    The storage layer writes numerical arrays to NPZ and everything else to JSON.
    A container that implements this protocol can be persisted and reconstructed
    without the storage layer knowing which domain produced it.
    """

    @property
    def metadata(self) -> Any:
        """Reproducibility metadata for the computation that produced this."""
        ...

    def array_payload(self) -> dict[str, Any]:
        """Return independent writable copies of every stored array."""
        ...

    def metadata_payload(self) -> dict[str, object]:
        """Return the JSON-compatible descriptor, without any array contents."""
        ...


__all__ = [
    "ClassicalMap",
    "Flow",
    "LinearOperatorLike",
    "Partition",
    "QuantumMap",
    "SerializableResult",
]

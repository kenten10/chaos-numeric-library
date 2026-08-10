"""Quantum-state, basis, and boundary-phase conventions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from chaos_numerics.core import ValidationError
from chaos_numerics.core._validation import as_complex_array
from chaos_numerics.core.types import ArrayLike, ComplexArray


class QuantumBasis(StrEnum):
    """Built-in finite-dimensional basis labels."""

    POSITION = "position"
    MOMENTUM = "momentum"


@dataclass(frozen=True, slots=True)
class BoundaryPhases:
    """Canonical position/momentum boundary phases measured in turns modulo one."""

    position: float = 0.0
    momentum: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _phase(self.position, name="position phase"))
        object.__setattr__(self, "momentum", _phase(self.momentum, name="momentum phase"))

    def to_dict(self) -> dict[str, float]:
        return {"position": self.position, "momentum": self.momentum}


def basis_state(
    *, dimension: int, index: int, basis: QuantumBasis | str = QuantumBasis.POSITION
) -> ComplexArray:
    """Return one normalized computational-basis vector."""
    size = _positive_int(dimension, name="dimension")
    location = _nonnegative_int(index, name="index")
    if location >= size:
        raise ValidationError(f"index must lie in [0, {size}); got {location}")
    _basis(basis)
    state = np.zeros(size, dtype=np.complex128)
    state[location] = 1.0
    return state


def normalize_state(state: ArrayLike, *, dimension: int | None = None) -> ComplexArray:
    """Return a normalized complex copy along the trailing state axis."""
    array = quantum_state(state, dimension=dimension, normalized=False)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if bool(np.any(norms <= np.finfo(np.float64).tiny)):
        raise ValidationError("quantum state cannot have zero norm")
    return np.asarray(array / norms, dtype=np.complex128)


def quantum_state(
    state: ArrayLike,
    *,
    dimension: int | None = None,
    normalized: bool = True,
    tolerance: float = 1e-12,
) -> ComplexArray:
    """Validate scalar or batched states and return an owned ``complex128`` copy."""
    array = as_complex_array(state, name="quantum state", copy=True)
    if dimension is not None:
        size = _positive_int(dimension, name="dimension")
        if array.shape[-1] != size:
            raise ValidationError(
                f"quantum state must have trailing dimension {size}; got {array.shape}"
            )
    if not isinstance(normalized, bool):
        raise ValidationError(f"normalized must be a bool; got {normalized!r}")
    error_tolerance = _nonnegative_float(tolerance, name="tolerance")
    if normalized:
        norm_error = np.abs(np.linalg.norm(array, axis=-1) - 1.0)
        if bool(np.any(norm_error > error_tolerance)):
            raise ValidationError(
                f"quantum state norm must equal one within {error_tolerance}; "
                f"maximum error is {float(np.max(norm_error))}"
            )
    return array


def _parity_operator(dimension: int, *, antiperiodic: bool) -> ComplexArray:
    """Return the read-only position-space parity matrix for a matched grid.

    ``antiperiodic`` selects between the two twists whose position *and*
    momentum grids are both self-reflecting with the same convention:
    ``(alpha, beta) = (0, 0)`` reflects as ``j -> -j mod N`` and
    ``(1/2, 1/2)`` reflects as ``j -> N - 1 - j``. Both come out as plain 0/1
    permutations. :func:`_reflection_operator` covers the mixed twists as well.
    """
    phases = BoundaryPhases(0.5, 0.5) if antiperiodic else BoundaryPhases(0.0, 0.0)
    reflection = _reflection_operator(dimension, phases=phases)
    assert reflection is not None
    return reflection


def _reflection_operator(dimension: int, *, phases: BoundaryPhases) -> ComplexArray | None:
    """Return the exact ``q -> -q`` reflection on a twisted grid, or ``None``.

    On the grid ``q_j = (j + alpha) / N`` the point ``-q_j`` is another grid
    point only when ``2 alpha`` is an integer, and it generally lies one period
    away: ``-q_j = q_k - w`` with ``k = (-j - 2 alpha) mod N`` and integer
    winding ``w``. A momentum twist ``beta`` makes the wave function
    quasi-periodic, ``psi(q + 1) = exp(2 pi i beta) psi(q)``, so the reflected
    amplitude picks up ``exp(-2 pi i beta w)``:

    ``(S psi)_j = exp(-2 pi i beta w_j) psi_{k_j}``

    That factor is ``+/-1`` exactly when ``2 beta`` is an integer, which is also
    what makes ``S`` an involution, so the operator exists precisely for
    ``alpha, beta in {0, 1/2}`` and ``None`` is returned otherwise. Measured
    ``||[U, S]||_F / sqrt(N)`` for :class:`~chaos_numerics.quantum.KickedRotor`
    at ``K = 7.3`` is at most ``5.3e-14`` over ``N`` in ``{8, 16, 63, 64, 65,
    128}`` and all four twists, with the involution and Hermiticity defects
    identically zero.

    The overall sign is a convention, so it is normalized to ``+1`` at ``j = 0``.
    Without that the ``(1/2, 1/2)`` operator would come out as ``-P``, which
    carries the same sectors with the ``even``/``odd`` labels exchanged.
    """
    doubled_position = 2.0 * phases.position
    doubled_momentum = 2.0 * phases.momentum
    if not (doubled_position.is_integer() and doubled_momentum.is_integer()):
        return None
    shift = int(doubled_position)
    twisted = int(doubled_momentum) % 2
    indices = np.arange(dimension)
    targets = np.mod(-indices - shift, dimension)
    winding = (targets + indices + shift) // dimension
    signs = np.where(twisted * winding % 2 == 1, -1.0, 1.0)
    reflection = np.zeros((dimension, dimension), dtype=np.complex128)
    reflection[targets, indices] = signs * signs[0]
    reflection.setflags(write=False)
    return reflection


def _basis(value: QuantumBasis | str) -> QuantumBasis:
    try:
        return QuantumBasis(value)
    except ValueError as error:
        raise ValidationError(f"basis must be 'position' or 'momentum'; got {value!r}") from error


def _phase(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    return result % 1.0


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValidationError(f"{name} must be positive; got 0")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValidationError(f"{name} must be non-negative; got {result}")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite non-negative number; got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValidationError(f"{name} must be a finite non-negative number; got {value!r}")
    return result


__all__ = [
    "BoundaryPhases",
    "QuantumBasis",
    "basis_state",
    "normalize_state",
    "quantum_state",
]

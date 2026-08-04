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

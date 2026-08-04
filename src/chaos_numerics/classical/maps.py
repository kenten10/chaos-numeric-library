"""Two-dimensional classical maps on the half-open unit torus.

States use coordinate order ``(q, p)`` and shape ``(..., 2)``. Every step first
wraps both coordinates into ``[0, 1)`` and returns a canonical ``float64`` array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.exceptions import ValidationError
from chaos_numerics.core.types import ArrayLike, FloatArray

_BOUNDS = ((0.0, 1.0), (0.0, 1.0))
_PERIODIC = (True, True)
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True, slots=True)
class StandardMap:
    """Chirikov standard map in unit-torus coordinates.

    The update is ``p' = p + K/(2π) sin(2πq)`` followed by ``q' = q + p'``,
    with both results reduced modulo one. Its Jacobian is evaluated before the
    modulo operation, away from the discontinuous wrapping cuts.
    """

    kick_strength: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kick_strength",
            _finite_real(self.kick_strength, name="kick_strength"),
        )

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def is_periodic(self) -> tuple[bool, bool]:
        return _PERIODIC

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return _BOUNDS

    @property
    def parameters(self) -> dict[str, object]:
        """JSON-compatible model parameters for result metadata."""
        return {"kick_strength": self.kick_strength}

    def step(self, state: ArrayLike, /) -> FloatArray:
        points = _states(state)
        q = points[..., 0]
        p = points[..., 1]
        next_p = np.mod(p + (self.kick_strength / _TWO_PI) * np.sin(_TWO_PI * q), 1.0)
        next_q = np.mod(q + next_p, 1.0)
        return _coordinates(next_q, next_p)

    def jacobian(self, state: ArrayLike, /) -> FloatArray:
        points = _states(state)
        kick_derivative = self.kick_strength * np.cos(_TWO_PI * points[..., 0])
        result = np.empty((*points.shape[:-1], 2, 2), dtype=np.float64)
        result[..., 0, 0] = 1.0 + kick_derivative
        result[..., 0, 1] = 1.0
        result[..., 1, 0] = kick_derivative
        result[..., 1, 1] = 1.0
        return result


@dataclass(frozen=True, slots=True)
class CatMap:
    """Linear area-preserving torus automorphism from an ``SL(2, Z)`` matrix."""

    matrix: tuple[tuple[int, int], tuple[int, int]] = ((2, 1), (1, 1))

    def __post_init__(self) -> None:
        matrix = _integer_matrix(self.matrix)
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if determinant != 1:
            raise ValidationError(
                f"CatMap matrix must have exact integer determinant 1; got {determinant}"
            )
        object.__setattr__(self, "matrix", matrix)

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def is_periodic(self) -> tuple[bool, bool]:
        return _PERIODIC

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return _BOUNDS

    @property
    def parameters(self) -> dict[str, object]:
        """JSON-compatible model parameters for result metadata."""
        return {"matrix": self.matrix}

    def step(self, state: ArrayLike, /) -> FloatArray:
        points = _states(state)
        matrix = np.asarray(self.matrix, dtype=np.float64)
        return np.mod(points @ matrix.T, 1.0)

    def jacobian(self, state: ArrayLike, /) -> FloatArray:
        points = _states(state)
        matrix = np.asarray(self.matrix, dtype=np.float64)
        return np.broadcast_to(matrix, (*points.shape[:-1], 2, 2)).copy()


@dataclass(frozen=True, slots=True)
class BakerMap:
    """Area-preserving generalized baker map with a right-owned branch cut.

    For ``q < cut`` the left strip maps to the full horizontal interval and the
    lower vertical strip. ``q == cut`` belongs to the right branch. The default
    ``cut=0.5`` is the symmetric baker map.
    """

    cut: float = 0.5

    def __post_init__(self) -> None:
        cut = _finite_real(self.cut, name="cut")
        if not 0.0 < cut < 1.0:
            raise ValidationError(f"cut must satisfy 0 < cut < 1; got {cut}")
        object.__setattr__(self, "cut", cut)

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def is_periodic(self) -> tuple[bool, bool]:
        return _PERIODIC

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return _BOUNDS

    @property
    def parameters(self) -> dict[str, object]:
        """JSON-compatible model parameters for result metadata."""
        return {"cut": self.cut}

    def step(self, state: ArrayLike, /) -> FloatArray:
        points = _states(state)
        q = points[..., 0]
        p = points[..., 1]
        left = q < self.cut
        right_width = 1.0 - self.cut
        next_q = np.where(left, q / self.cut, (q - self.cut) / right_width)
        next_p = np.where(left, self.cut * p, self.cut + right_width * p)
        return _coordinates(next_q, next_p)

    def jacobian(self, state: ArrayLike, /) -> FloatArray:
        points = _states(state)
        left = points[..., 0] < self.cut
        q_scale = np.where(left, 1.0 / self.cut, 1.0 / (1.0 - self.cut))
        p_scale = np.where(left, self.cut, 1.0 - self.cut)
        result = np.zeros((*points.shape[:-1], 2, 2), dtype=np.float64)
        result[..., 0, 0] = q_scale
        result[..., 1, 1] = p_scale
        return result


def _states(state: ArrayLike) -> FloatArray:
    points = as_float_array(state, name="state", trailing_dim=2, copy=True)
    return np.mod(points, 1.0)


def _coordinates(q: FloatArray, p: FloatArray) -> FloatArray:
    result = np.empty((*q.shape, 2), dtype=np.float64)
    result[..., 0] = q
    result[..., 1] = p
    return result


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    return result


def _integer_matrix(value: object) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValidationError("CatMap matrix must be a regular 2 x 2 integer matrix") from error
    if raw.shape != (2, 2) or raw.dtype.kind not in "iu":
        raise ValidationError(
            f"CatMap matrix must be a 2 x 2 integer matrix; got shape {raw.shape}, dtype {raw.dtype}"
        )
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValidationError("CatMap matrix entries must fit in signed 64-bit integers")
    matrix = raw.astype(np.int64, copy=False)
    return (
        (int(matrix[0, 0]), int(matrix[0, 1])),
        (int(matrix[1, 0]), int(matrix[1, 1])),
    )


__all__ = ["BakerMap", "CatMap", "StandardMap"]

"""Classical maps on the half-open unit torus and the half-open unit interval.

Two-dimensional maps use coordinate order ``(q, p)`` and shape ``(..., 2)``; each
step wraps both coordinates into ``[0, 1)`` and returns a canonical ``float64``
array. The one-dimensional :class:`LogisticMap` uses shape ``(..., 1)`` on the
non-periodic interval ``[0, 1)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from chaos_numerics.core._validation import as_float_array, wrap_into_half_open
from chaos_numerics.core.exceptions import ValidationError
from chaos_numerics.core.types import ArrayLike, FloatArray

_BOUNDS = ((0.0, 1.0), (0.0, 1.0))
_PERIODIC = (True, True)
_TWO_PI = 2.0 * math.pi

_INTERVAL_BOUNDS = ((0.0, 1.0),)
_INTERVAL_PERIODIC = (False,)
_MAXIMUM_RATE = 4.0
# The largest float64 strictly below the excluded upper bound 1.0.
_BELOW_ONE = float(np.nextafter(1.0, 0.0))


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
        next_p = _wrap_unit_torus(p + (self.kick_strength / _TWO_PI) * np.sin(_TWO_PI * q))
        return _coordinates(q + next_p, next_p)

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
        return _wrap_unit_torus(points @ matrix.T)

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


@dataclass(frozen=True, slots=True)
class LogisticMap:
    """Logistic map ``x -> rate * x * (1 - x)`` on the half-open interval ``[0, 1)``.

    The default ``rate=4`` is the surjective (fully developed) case whose unique
    absolutely continuous invariant measure is the arcsine law with density
    ``1 / (pi * sqrt(x * (1 - x)))``; it is smoothly conjugate to the angle
    doubling map through ``x = sin(pi * theta / 2)**2``, so its Lyapunov exponent
    is exactly ``log 2``. Lowering ``rate`` walks the period-doubling cascade
    backwards, which makes this the standard bifurcation-diagram model.

    Notes
    -----
    ``rate=4`` sends the single point ``x = 1/2`` to exactly ``1.0``, the excluded
    endpoint of the ``bounds`` contract, so the raw image is pushed down to
    ``nextafter(1.0, 0.0)`` with :func:`numpy.minimum`. Every other map in this
    module keeps its images inside the documented half-open bounds, and code that
    consumes ``bounds`` -- partition location, Ulam matrices, trajectory
    normalization -- relies on that. Clamping is safe here because the affected
    preimages form a Lebesgue-null set: ``{1/2}`` for one step, and its finitely
    many preimages for any finite number of steps, so no invariant measure and no
    Monte Carlo average sees the change. Clamping is *not* a way to hide a
    genuinely escaping orbit: ``rate > 4`` maps a neighbourhood of ``1/2`` of
    positive measure outside ``[0, 1]``, which is why the constructor rejects it
    rather than clamping it.

    ``step`` and ``jacobian`` evaluate the polynomial for any finite input and only
    guarantee an image inside ``[0, 1)`` for a state inside ``bounds``. Rejecting
    out-of-domain states here would make the map unusable with the derivative-free
    root solver behind :func:`~chaos_numerics.classical.find_periodic_orbits`,
    which probes a rounding error either side of a fixed point at ``x = 0``. The
    domain is enforced where a state enters an experiment instead:
    :func:`~chaos_numerics.classical.iterate` rejects an initial state outside the
    non-periodic bounds, and partitions reject points outside their edges.
    """

    rate: float = _MAXIMUM_RATE

    def __post_init__(self) -> None:
        rate = _finite_real(self.rate, name="rate")
        if not 0.0 < rate <= _MAXIMUM_RATE:
            raise ValidationError(
                f"rate must satisfy 0 < rate <= {_MAXIMUM_RATE}; got {rate}. Rates above "
                f"{_MAXIMUM_RATE} send a positive-measure neighbourhood of x = 1/2 outside "
                "the unit interval, so the map no longer acts on its documented bounds"
            )
        object.__setattr__(self, "rate", rate)

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def is_periodic(self) -> tuple[bool]:
        return _INTERVAL_PERIODIC

    @property
    def bounds(self) -> tuple[tuple[float, float]]:
        return _INTERVAL_BOUNDS

    @property
    def parameters(self) -> dict[str, object]:
        """JSON-compatible model parameters for result metadata."""
        return {"rate": self.rate}

    def step(self, state: ArrayLike, /) -> FloatArray:
        points = _interval_states(state)
        image = self.rate * points * (1.0 - points)
        return np.minimum(image, _BELOW_ONE)

    def jacobian(self, state: ArrayLike, /) -> FloatArray:
        points = _interval_states(state)
        result = np.empty((*points.shape[:-1], 1, 1), dtype=np.float64)
        result[..., 0, 0] = self.rate * (1.0 - 2.0 * points[..., 0])
        return result


def _states(state: ArrayLike) -> FloatArray:
    """Validate ``(..., 2)`` torus states and reduce them into ``[0, 1)``.

    ``copy=False`` is safe even though ``as_float_array`` may then hand back the
    caller's own array: :func:`_wrap_unit_torus` allocates its result, and nothing
    downstream writes through this reference, so no input is ever mutated in place.
    Copying here cost one allocation per step of every trajectory.
    """
    return _wrap_unit_torus(as_float_array(state, name="state", trailing_dim=2))


def _interval_states(state: ArrayLike) -> FloatArray:
    """Validate the shape of ``(..., 1)`` states on the unit interval.

    The coordinate is non-periodic, so there is no reduction to apply the way the
    torus maps reduce theirs; the value itself is left alone. See
    :class:`LogisticMap` for why an out-of-domain value is evaluated rather than
    rejected.
    """
    return as_float_array(state, name="state", trailing_dim=1)


def _wrap_unit_torus(values: FloatArray) -> FloatArray:
    """Reduce torus coordinates into the documented half-open ``[0, 1)`` bounds.

    ``np.mod`` alone is not enough: ``np.mod(-1e-17, 1.0)`` rounds up to exactly
    ``1.0``, which is the excluded endpoint of the ``bounds`` contract.
    """
    return wrap_into_half_open(values, lower=0.0, upper=1.0)


def _coordinates(q: FloatArray, p: FloatArray) -> FloatArray:
    result = np.empty((*q.shape, 2), dtype=np.float64)
    result[..., 0] = _wrap_unit_torus(q)
    result[..., 1] = _wrap_unit_torus(p)
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


__all__ = ["BakerMap", "CatMap", "LogisticMap", "StandardMap"]

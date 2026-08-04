"""Low-period orbit search and linear stability diagnostics."""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root  # type: ignore[import-untyped]

from chaos_numerics.core import (
    ClassicalMap,
    ConvergenceError,
    ConvergenceInfo,
    ConvergenceWarning,
    Diagnostic,
    ExperimentMetadata,
    ValidationError,
)
from chaos_numerics.core._validation import as_complex_array, as_float_array
from chaos_numerics.core.types import ArrayLike, ComplexArray, FloatArray


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class PeriodicOrbitResult:
    """Immutable representative points, residuals, monodromy matrices, and multipliers."""

    period: int
    points: FloatArray
    residuals: FloatArray
    monodromy_matrices: FloatArray
    stability_multipliers: ComplexArray
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def __post_init__(self) -> None:
        period = _positive_int(self.period, name="period")
        points = _readonly_float(self.points, name="periodic points", ndim=2)
        count, dimension = points.shape
        residuals = _readonly_float(self.residuals, name="periodic residuals", ndim=1)
        monodromy = _readonly_float(
            self.monodromy_matrices,
            name="monodromy matrices",
            ndim=3,
        )
        multipliers = _readonly_complex(
            self.stability_multipliers,
            name="stability multipliers",
            ndim=2,
        )
        if residuals.shape != (count,):
            raise ValidationError(f"periodic residuals must have shape ({count},)")
        if monodromy.shape != (count, dimension, dimension):
            raise ValidationError(
                f"monodromy matrices must have shape ({count}, {dimension}, {dimension})"
            )
        if multipliers.shape != (count, dimension):
            raise ValidationError(f"stability multipliers must have shape ({count}, {dimension})")
        if bool(np.any(residuals < 0.0)):
            raise ValidationError("periodic residuals must be non-negative")
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "monodromy_matrices", monodromy)
        object.__setattr__(self, "stability_multipliers", multipliers)

    @property
    def count(self) -> int:
        return int(self.points.shape[0])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PeriodicOrbitResult):
            return NotImplemented
        return (
            self.period == other.period
            and np.array_equal(self.points, other.points)
            and np.array_equal(self.residuals, other.residuals)
            and np.array_equal(self.monodromy_matrices, other.monodromy_matrices)
            and np.array_equal(self.stability_multipliers, other.stability_multipliers)
            and self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        maximum = float(np.max(self.residuals)) if self.residuals.size else None
        return (
            f"PeriodicOrbitResult(period={self.period}, count={self.count}, "
            f"state_dim={self.points.shape[1]}, max_residual={maximum})"
        )


def find_periodic_orbits(
    model: ClassicalMap,
    guesses: ArrayLike,
    *,
    period: int,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    deduplication_tolerance: float = 1e-8,
    strict: bool = False,
) -> PeriodicOrbitResult:
    """Find representative roots of ``F**period(x) - x`` from supplied guesses."""
    orbit_period = _positive_int(period, name="period")
    maximum_iterations = _positive_int(max_iterations, name="max_iterations")
    residual_tolerance = _positive_float(tolerance, name="tolerance")
    deduplication = _positive_float(
        deduplication_tolerance,
        name="deduplication_tolerance",
    )
    if not isinstance(strict, bool):
        raise ValidationError(f"strict must be a bool; got {strict!r}")
    dimension = _dimension(model)
    starts = as_float_array(guesses, name="guesses", trailing_dim=dimension, copy=True)
    if starts.ndim == 1:
        starts = starts[None, :]
    if starts.ndim != 2 or starts.shape[0] == 0:
        raise ValidationError(
            f"guesses must have shape (state_dim,) or (count, state_dim); got {starts.shape}"
        )

    points: list[FloatArray] = []
    residuals: list[float] = []
    monodromy: list[FloatArray] = []
    multipliers: list[ComplexArray] = []
    failed = 0

    for guess in starts:
        solution = root(
            lambda candidate: _periodic_residual(model, candidate, orbit_period),
            guess,
            method="hybr",
            options={"maxfev": maximum_iterations},
        )
        point = _normalize(model, as_float_array(solution.x, name="root solution", ndim=1))
        residual = float(np.linalg.norm(_periodic_residual(model, point, orbit_period)))
        if not np.isfinite(residual) or residual > residual_tolerance:
            failed += 1
            continue
        if any(
            _same_orbit(model, point, existing, orbit_period, deduplication) for existing in points
        ):
            continue
        _, matrix = _iterate_with_monodromy(model, point, orbit_period)
        points.append(point)
        residuals.append(residual)
        monodromy.append(matrix)
        multipliers.append(np.asarray(np.linalg.eigvals(matrix), dtype=np.complex128))

    message: str | None = None
    if failed:
        message = f"{failed} of {starts.shape[0]} periodic-orbit guesses did not converge"
    if not points:
        message = "no periodic orbit converged from the supplied guesses"
    if message is not None and strict:
        raise ConvergenceError(message)
    diagnostics: tuple[Diagnostic, ...] = ()
    if message is not None:
        python_warnings.warn(message, ConvergenceWarning, stacklevel=2)
        diagnostics = (Diagnostic("periodic-orbit-partial", message, category="convergence"),)

    point_array = np.stack(points) if points else np.empty((0, dimension), dtype=np.float64)
    residual_array = np.asarray(residuals, dtype=np.float64)
    monodromy_array = (
        np.stack(monodromy) if monodromy else np.empty((0, dimension, dimension), dtype=np.float64)
    )
    multiplier_array = (
        np.stack(multipliers) if multipliers else np.empty((0, dimension), dtype=np.complex128)
    )
    convergence = ConvergenceInfo(
        converged=failed == 0 and bool(points),
        iterations=int(starts.shape[0]),
        residual=float(np.max(residual_array)) if residual_array.size else None,
        tolerance=residual_tolerance,
        history=residual_array if residual_array.size else None,
        reason="all guesses converged or deduplicated" if failed == 0 and points else message,
    )
    metadata = ExperimentMetadata(
        parameters={
            "model": type(model).__name__,
            "period": orbit_period,
            "guess_count": int(starts.shape[0]),
            "max_iterations": maximum_iterations,
            "deduplication_tolerance": deduplication,
        },
        warnings=diagnostics,
        convergence=convergence,
    )
    return PeriodicOrbitResult(
        orbit_period,
        point_array,
        residual_array,
        monodromy_array,
        multiplier_array,
        metadata,
    )


def _periodic_residual(model: ClassicalMap, point: ArrayLike, period: int) -> FloatArray:
    initial = as_float_array(point, name="periodic point", ndim=1, copy=True)
    final, _ = _iterate_with_monodromy(model, initial, period)
    difference = final - initial
    bounds = model.bounds
    periodic = model.is_periodic
    for coordinate in range(initial.size):
        if periodic[coordinate]:
            width = bounds[coordinate][1] - bounds[coordinate][0]
            difference[coordinate] = (
                np.mod(difference[coordinate] + 0.5 * width, width) - 0.5 * width
            )
    return difference


def _iterate_with_monodromy(
    model: ClassicalMap,
    point: FloatArray,
    period: int,
) -> tuple[FloatArray, FloatArray]:
    dimension = point.size
    state = point.copy()
    matrix = np.eye(dimension, dtype=np.float64)
    for _ in range(period):
        jacobian = as_float_array(model.jacobian(state), name="model jacobian", ndim=2)
        if jacobian.shape != (dimension, dimension):
            raise ValidationError(
                f"model jacobian must have shape ({dimension}, {dimension}); got {jacobian.shape}"
            )
        matrix = jacobian @ matrix
        state = as_float_array(
            model.step(state),
            name="model step result",
            ndim=1,
            trailing_dim=dimension,
        )
    return state, matrix


def _normalize(model: ClassicalMap, point: FloatArray) -> FloatArray:
    result = point.copy()
    for coordinate, ((lower, upper), periodic) in enumerate(
        zip(model.bounds, model.is_periodic, strict=True)
    ):
        if periodic:
            result[coordinate] = lower + np.mod(result[coordinate] - lower, upper - lower)
    return result


def _wrapped_norm(model: ClassicalMap, difference: FloatArray) -> float:
    adjusted = difference.copy()
    for coordinate, ((lower, upper), periodic) in enumerate(
        zip(model.bounds, model.is_periodic, strict=True)
    ):
        if periodic:
            width = upper - lower
            adjusted[coordinate] = np.mod(adjusted[coordinate] + 0.5 * width, width) - 0.5 * width
    return float(np.linalg.norm(adjusted))


def _same_orbit(
    model: ClassicalMap,
    candidate: FloatArray,
    representative: FloatArray,
    period: int,
    tolerance: float,
) -> bool:
    point = representative.copy()
    for _ in range(period):
        if _wrapped_norm(model, candidate - point) <= tolerance:
            return True
        point = as_float_array(model.step(point), name="model step result", ndim=1)
    return False


def _dimension(model: ClassicalMap) -> int:
    dimension = model.state_dim
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise ValidationError("model state_dim must be a positive integer")
    result = int(dimension)
    if result <= 0 or len(model.bounds) != result or len(model.is_periodic) != result:
        raise ValidationError("model state_dim, bounds, and is_periodic are inconsistent")
    return result


def _readonly_float(value: FloatArray, *, name: str, ndim: int) -> FloatArray:
    result = as_float_array(value, name=name, ndim=ndim, copy=True)
    result.setflags(write=False)
    return result


def _readonly_complex(value: ComplexArray, *, name: str, ndim: int) -> ComplexArray:
    result = as_complex_array(value, name=name, ndim=ndim, copy=True)
    result.setflags(write=False)
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a positive integer; got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValidationError(f"{name} must be positive; got {result}")
    return result


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite positive number; got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValidationError(f"{name} must be a finite positive number; got {value!r}")
    return result


__all__ = ["PeriodicOrbitResult", "find_periodic_orbits"]

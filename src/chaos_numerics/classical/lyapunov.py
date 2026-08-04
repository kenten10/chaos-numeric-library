"""Finite-time Lyapunov exponents for discrete classical maps."""

from __future__ import annotations

import warnings as python_warnings
from collections.abc import Mapping

import numpy as np

from chaos_numerics.core import (
    AnalysisResult,
    ClassicalMap,
    ConvergenceError,
    ConvergenceInfo,
    ConvergenceWarning,
    Diagnostic,
    ExperimentMetadata,
    NumericalError,
    ValidationError,
)
from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.types import ArrayLike, FloatArray

_MINIMUM_CONVERGENCE_STEPS = 32


def largest_lyapunov_exponent(
    model: ClassicalMap,
    initial_state: ArrayLike,
    *,
    steps: int,
    transient: int = 0,
    reorthogonalization_interval: int = 1,
    history_interval: int = 1,
    convergence_rtol: float = 1e-3,
    convergence_atol: float = 1e-8,
    seed: int | None = None,
    strict: bool = False,
) -> AnalysisResult:
    """Estimate the largest exponent by periodic tangent-vector normalization."""
    config = _validate_config(
        steps=steps,
        transient=transient,
        reorthogonalization_interval=reorthogonalization_interval,
        history_interval=history_interval,
        convergence_rtol=convergence_rtol,
        convergence_atol=convergence_atol,
        seed=seed,
        strict=strict,
    )
    state, dimension = _initial_state(model, initial_state)
    generator = np.random.default_rng(config.seed)
    tangent = (
        np.ones(dimension, dtype=np.float64)
        if config.seed is None
        else generator.standard_normal(dimension)
    )
    tangent /= _safe_norm(tangent, context="initial tangent vector")

    state, tangent, _ = _vector_transient(
        model,
        state,
        tangent,
        steps=config.transient,
        interval=config.interval,
    )
    values, history, window_difference = _vector_measurement(
        model,
        state,
        tangent,
        steps=config.steps,
        interval=config.interval,
        history_interval=config.history_interval,
    )
    return _result(
        name="largest_lyapunov_exponent",
        model=model,
        values=values,
        history=history,
        window_difference=window_difference,
        config=config,
        orthogonality_defect=None,
    )


def lyapunov_spectrum(
    model: ClassicalMap,
    initial_state: ArrayLike,
    *,
    steps: int,
    transient: int = 0,
    reorthogonalization_interval: int = 1,
    history_interval: int = 1,
    convergence_rtol: float = 1e-3,
    convergence_atol: float = 1e-8,
    seed: int | None = None,
    strict: bool = False,
) -> AnalysisResult:
    """Estimate the full spectrum with periodic QR reorthogonalization.

    Only one initial state with shape ``(state_dim,)`` is accepted in v0.1. The
    returned exponents are sorted in descending order. History has shape
    ``(records, state_dim)`` and its final row exactly equals ``values``.
    """
    config = _validate_config(
        steps=steps,
        transient=transient,
        reorthogonalization_interval=reorthogonalization_interval,
        history_interval=history_interval,
        convergence_rtol=convergence_rtol,
        convergence_atol=convergence_atol,
        seed=seed,
        strict=strict,
    )
    state, dimension = _initial_state(model, initial_state)
    if config.seed is None:
        basis = np.eye(dimension, dtype=np.float64)
    else:
        generator = np.random.default_rng(config.seed)
        basis, _ = np.linalg.qr(generator.standard_normal((dimension, dimension)))

    state, basis, transient_defect = _matrix_transient(
        model,
        state,
        basis,
        steps=config.transient,
        interval=config.interval,
    )
    values, history, window_difference, measurement_defect = _matrix_measurement(
        model,
        state,
        basis,
        steps=config.steps,
        interval=config.interval,
        history_interval=config.history_interval,
    )
    order = np.argsort(-values)
    values = values[order]
    history = history[:, order]
    window_difference = window_difference[order]
    return _result(
        name="lyapunov_spectrum",
        model=model,
        values=values,
        history=history,
        window_difference=window_difference,
        config=config,
        orthogonality_defect=max(transient_defect, measurement_defect),
    )


class _Config:
    def __init__(
        self,
        *,
        steps: int,
        transient: int,
        interval: int,
        history_interval: int,
        rtol: float,
        atol: float,
        seed: int | None,
        strict: bool,
    ) -> None:
        self.steps = steps
        self.transient = transient
        self.interval = interval
        self.history_interval = history_interval
        self.rtol = rtol
        self.atol = atol
        self.seed = seed
        self.strict = strict


def _validate_config(
    *,
    steps: object,
    transient: object,
    reorthogonalization_interval: object,
    history_interval: object,
    convergence_rtol: object,
    convergence_atol: object,
    seed: object,
    strict: object,
) -> _Config:
    measured_steps = _positive_int(steps, name="steps")
    if measured_steps < 2:
        raise ValidationError("steps must be at least 2 for two-window convergence diagnostics")
    transient_steps = _nonnegative_int(transient, name="transient")
    interval = _positive_int(
        reorthogonalization_interval,
        name="reorthogonalization_interval",
    )
    history_stride = _positive_int(history_interval, name="history_interval")
    rtol = _nonnegative_float(convergence_rtol, name="convergence_rtol")
    atol = _nonnegative_float(convergence_atol, name="convergence_atol")
    if seed is not None:
        seed = _nonnegative_int(seed, name="seed")
    if not isinstance(strict, bool):
        raise ValidationError(f"strict must be a bool; got {strict!r}")
    return _Config(
        steps=measured_steps,
        transient=transient_steps,
        interval=interval,
        history_interval=history_stride,
        rtol=rtol,
        atol=atol,
        seed=seed,
        strict=strict,
    )


def _initial_state(model: ClassicalMap, initial_state: ArrayLike) -> tuple[FloatArray, int]:
    dimension = model.state_dim
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise ValidationError(f"model state_dim must be a positive integer; got {dimension!r}")
    size = int(dimension)
    if size <= 0:
        raise ValidationError(f"model state_dim must be positive; got {size}")
    state = as_float_array(
        initial_state,
        name="initial_state",
        ndim=1,
        trailing_dim=size,
        copy=True,
    )
    return state, size


def _advance(model: ClassicalMap, state: FloatArray) -> tuple[FloatArray, FloatArray]:
    dimension = state.size
    jacobian = as_float_array(
        model.jacobian(state),
        name="model jacobian",
        ndim=2,
        copy=False,
    )
    if jacobian.shape != (dimension, dimension):
        raise ValidationError(
            f"model jacobian must have shape ({dimension}, {dimension}); got {jacobian.shape}"
        )
    next_state = as_float_array(
        model.step(state),
        name="model step result",
        ndim=1,
        trailing_dim=dimension,
        copy=False,
    )
    return next_state, jacobian


def _vector_transient(
    model: ClassicalMap,
    state: FloatArray,
    tangent: FloatArray,
    *,
    steps: int,
    interval: int,
) -> tuple[FloatArray, FloatArray, float]:
    for index in range(1, steps + 1):
        state, jacobian = _advance(model, state)
        tangent = jacobian @ tangent
        if index % interval == 0 or index == steps:
            tangent /= _safe_norm(tangent, context="transient tangent vector")
    return state, tangent, 0.0


def _matrix_transient(
    model: ClassicalMap,
    state: FloatArray,
    basis: FloatArray,
    *,
    steps: int,
    interval: int,
) -> tuple[FloatArray, FloatArray, float]:
    maximum_defect = 0.0
    for index in range(1, steps + 1):
        state, jacobian = _advance(model, state)
        basis = jacobian @ basis
        if index % interval == 0 or index == steps:
            basis, _ = np.linalg.qr(basis)
            maximum_defect = max(maximum_defect, _orthogonality_defect(basis))
    return state, basis, maximum_defect


def _vector_measurement(
    model: ClassicalMap,
    state: FloatArray,
    tangent: FloatArray,
    *,
    steps: int,
    interval: int,
    history_interval: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    cumulative = 0.0
    start_sum = 0.0
    split_sum = 0.0
    event_count = 0
    records: list[float] = []
    window_start, split = _window_boundaries(steps)
    for index in range(1, steps + 1):
        state, jacobian = _advance(model, state)
        tangent = jacobian @ tangent
        if _is_event(index, steps=steps, interval=interval, boundaries=(window_start, split)):
            norm = _safe_norm(tangent, context="tangent vector")
            cumulative += float(np.log(norm))
            tangent /= norm
            event_count += 1
            if index == window_start:
                start_sum = cumulative
            if index == split:
                split_sum = cumulative
            estimate = cumulative / index
            if event_count % history_interval == 0 or index == steps:
                records.append(estimate)
    window = steps // 2
    difference = abs((cumulative - split_sum) / window - (split_sum - start_sum) / window)
    return (
        np.asarray([cumulative / steps], dtype=np.float64),
        np.asarray(records, dtype=np.float64),
        np.asarray([difference], dtype=np.float64),
    )


def _matrix_measurement(
    model: ClassicalMap,
    state: FloatArray,
    basis: FloatArray,
    *,
    steps: int,
    interval: int,
    history_interval: int,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    dimension = state.size
    cumulative = np.zeros(dimension, dtype=np.float64)
    start_sum = cumulative.copy()
    split_sum = cumulative.copy()
    event_count = 0
    maximum_defect = 0.0
    records: list[FloatArray] = []
    window_start, split = _window_boundaries(steps)
    for index in range(1, steps + 1):
        state, jacobian = _advance(model, state)
        basis = jacobian @ basis
        if _is_event(index, steps=steps, interval=interval, boundaries=(window_start, split)):
            basis, triangular = np.linalg.qr(basis)
            diagonal = np.abs(np.diag(triangular))
            if bool(np.any(diagonal <= np.finfo(np.float64).tiny)):
                raise NumericalError("Lyapunov QR factor became singular")
            cumulative += np.log(diagonal)
            maximum_defect = max(maximum_defect, _orthogonality_defect(basis))
            event_count += 1
            if index == window_start:
                start_sum = cumulative.copy()
            if index == split:
                split_sum = cumulative.copy()
            estimate = cumulative / index
            if event_count % history_interval == 0 or index == steps:
                records.append(estimate.copy())
    window = steps // 2
    difference = np.abs((cumulative - split_sum) / window - (split_sum - start_sum) / window)
    return cumulative / steps, np.stack(records), difference, maximum_defect


def _result(
    *,
    name: str,
    model: ClassicalMap,
    values: FloatArray,
    history: FloatArray,
    window_difference: FloatArray,
    config: _Config,
    orthogonality_defect: float | None,
) -> AnalysisResult:
    thresholds = np.maximum(config.atol, config.rtol * np.maximum(np.abs(values), 1e-8))
    sufficiently_long = config.steps >= _MINIMUM_CONVERGENCE_STEPS
    converged = sufficiently_long and bool(np.all(window_difference <= thresholds))
    if not sufficiently_long:
        code = "insufficient-iterations"
        reason = f"at least {_MINIMUM_CONVERGENCE_STEPS} measured steps are required"
    elif not converged:
        code = "lyapunov-not-converged"
        reason = "final equal-length windows disagree beyond tolerance"
    else:
        code = "converged"
        reason = "final equal-length windows agree within tolerance"

    diagnostics: tuple[Diagnostic, ...] = ()
    if not converged:
        message = f"{name} did not converge: {reason}"
        if config.strict:
            raise ConvergenceError(message)
        python_warnings.warn(message, ConvergenceWarning, stacklevel=3)
        diagnostics = (Diagnostic(code=code, message=message, category="convergence"),)

    parameters: dict[str, object] = {
        "model": type(model).__name__,
        "model_parameters": _model_parameters(model),
        "steps": config.steps,
        "transient": config.transient,
        "reorthogonalization_interval": config.interval,
        "history_interval": config.history_interval,
        "convergence_rtol": config.rtol,
        "convergence_atol": config.atol,
        "minimum_convergence_steps": _MINIMUM_CONVERGENCE_STEPS,
    }
    if orthogonality_defect is not None:
        parameters["maximum_orthogonality_defect"] = orthogonality_defect
    convergence = ConvergenceInfo(
        converged=converged,
        iterations=config.steps,
        residual=float(np.max(window_difference)),
        tolerance=float(np.max(thresholds)),
        history=history,
        reason=reason,
    )
    metadata = ExperimentMetadata(
        parameters=parameters,
        precision="float64",
        seed=config.seed,
        warnings=diagnostics,
        convergence=convergence,
    )
    return AnalysisResult(
        name=name,
        values=values,
        uncertainty=window_difference,
        residuals=window_difference,
        metadata=metadata,
    )


def _model_parameters(model: ClassicalMap) -> dict[str, object]:
    parameters = getattr(model, "parameters", None)
    return dict(parameters) if isinstance(parameters, Mapping) else {}


def _window_boundaries(steps: int) -> tuple[int, int]:
    window = steps // 2
    return steps - 2 * window, steps - window


def _is_event(
    index: int,
    *,
    steps: int,
    interval: int,
    boundaries: tuple[int, int],
) -> bool:
    return index % interval == 0 or index == steps or index in boundaries


def _orthogonality_defect(basis: FloatArray) -> float:
    identity = np.eye(basis.shape[1], dtype=np.float64)
    return float(np.linalg.norm(basis.T @ basis - identity) / np.sqrt(basis.shape[1]))


def _safe_norm(vector: FloatArray, *, context: str) -> float:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        raise NumericalError(f"{context} has zero or non-finite norm")
    return norm


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
    if not np.isfinite(result) or result < 0.0:
        raise ValidationError(f"{name} must be a finite non-negative number; got {value!r}")
    return result


__all__ = ["largest_lyapunov_exponent", "lyapunov_spectrum"]

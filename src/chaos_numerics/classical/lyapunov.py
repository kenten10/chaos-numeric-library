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
    NumericalWarning,
    ValidationError,
)
from chaos_numerics.core._validation import as_float_array, validate_protocol
from chaos_numerics.core.types import ArrayLike, FloatArray

_MINIMUM_CONVERGENCE_STEPS = 32

# Benettin's criterion for QR-based spectra: the leading direction must not
# outgrow the trailing one by more than the float64 resolution between refreshes,
# i.e. (lambda_1 - lambda_d) * interval <= log(1/eps). Past the squared limit the
# trailing logarithms carry no significant digits and the result is refused.
_CONDITION_WARNING_LIMIT = 1.0 / np.finfo(np.float64).eps
_CONDITION_ERROR_LIMIT = _CONDITION_WARNING_LIMIT**2


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
    """Estimate the largest exponent by periodic tangent-vector normalization.

    ``reorthogonalization_interval`` must be short enough that the tangent vector
    stays inside the ``float64`` range between renormalizations; an interval that
    lets it overflow raises :class:`~chaos_numerics.core.NumericalError` naming
    the interval instead of leaking ``inf``/``nan`` into the result.

    The tangent vector starts at the all-ones direction when ``seed`` is ``None``
    and at a reproducible standard-normal draw otherwise, so a seed changes the
    finite-time estimate as well as making it repeatable.
    """
    validate_protocol(model, ClassicalMap, name="model")
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

    ``reorthogonalization_interval`` must be short enough that the propagated
    basis stays inside the ``float64`` range between QR refreshes; an interval
    that lets it overflow raises :class:`~chaos_numerics.core.NumericalError`
    naming the interval instead of leaking ``inf``/``nan`` into the result.

    A shorter interval than that is needed for the trailing exponents to mean
    anything. Between refreshes the leading direction outgrows the trailing one by
    ``exp((lambda_1 - lambda_d) * interval)``, and once that exceeds the
    ``float64`` resolution the trailing columns are rounding noise even though the
    QR basis remains perfectly orthogonal. The largest observed ratio is recorded
    as ``metadata.parameters["maximum_qr_condition"]``; exceeding the resolution
    emits :class:`~chaos_numerics.core.NumericalWarning` and exceeding its square
    raises :class:`~chaos_numerics.core.NumericalError`.
    """
    validate_protocol(model, ClassicalMap, name="model")
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

    state, basis, transient_defect, transient_condition = _matrix_transient(
        model,
        state,
        basis,
        steps=config.transient,
        interval=config.interval,
    )
    (
        values,
        history,
        window_difference,
        measurement_defect,
        measurement_condition,
    ) = _matrix_measurement(
        model,
        state,
        basis,
        steps=config.steps,
        interval=config.interval,
        history_interval=config.history_interval,
    )
    qr_condition = max(transient_condition, measurement_condition)
    if qr_condition > _CONDITION_WARNING_LIMIT:
        message = (
            f"Lyapunov QR condition reached {qr_condition:.3e}, above the float64 "
            f"resolution {_CONDITION_WARNING_LIMIT:.3e}: the trailing exponents may have "
            f"lost precision because the basis spread that far between refreshes. The "
            f"standard criterion is (lambda_1 - lambda_d) * reorthogonalization_interval "
            f"<= log(1/eps); reduce the interval (currently {config.interval})"
        )
        python_warnings.warn(message, NumericalWarning, stacklevel=2)
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
        qr_condition=qr_condition,
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
        tangent = _propagate(jacobian, tangent)
        if index % interval == 0 or index == steps:
            tangent /= _safe_norm(tangent, context="transient tangent vector", interval=interval)
    return state, tangent, 0.0


def _matrix_transient(
    model: ClassicalMap,
    state: FloatArray,
    basis: FloatArray,
    *,
    steps: int,
    interval: int,
) -> tuple[FloatArray, FloatArray, float, float]:
    maximum_defect = 0.0
    maximum_condition = 0.0
    for index in range(1, steps + 1):
        state, jacobian = _advance(model, state)
        basis = _propagate(jacobian, basis)
        if index % interval == 0 or index == steps:
            basis, _, condition = _reorthogonalize(basis, interval=interval, context="transient")
            maximum_defect = max(maximum_defect, _orthogonality_defect(basis))
            maximum_condition = max(maximum_condition, condition)
    return state, basis, maximum_defect, maximum_condition


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
        tangent = _propagate(jacobian, tangent)
        if _is_event(index, steps=steps, interval=interval, boundaries=(window_start, split)):
            norm = _safe_norm(tangent, context="tangent vector", interval=interval)
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
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    dimension = state.size
    cumulative = np.zeros(dimension, dtype=np.float64)
    start_sum = cumulative.copy()
    split_sum = cumulative.copy()
    event_count = 0
    maximum_defect = 0.0
    maximum_condition = 0.0
    records: list[FloatArray] = []
    window_start, split = _window_boundaries(steps)
    for index in range(1, steps + 1):
        state, jacobian = _advance(model, state)
        basis = _propagate(jacobian, basis)
        if _is_event(index, steps=steps, interval=interval, boundaries=(window_start, split)):
            basis, diagonal, condition = _reorthogonalize(
                basis, interval=interval, context="measurement"
            )
            cumulative += np.log(diagonal)
            maximum_defect = max(maximum_defect, _orthogonality_defect(basis))
            maximum_condition = max(maximum_condition, condition)
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
    return cumulative / steps, np.stack(records), difference, maximum_defect, maximum_condition


def _result(
    *,
    name: str,
    model: ClassicalMap,
    values: FloatArray,
    history: FloatArray,
    window_difference: FloatArray,
    config: _Config,
    orthogonality_defect: float | None,
    qr_condition: float | None = None,
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
    if qr_condition is not None:
        parameters["maximum_qr_condition"] = qr_condition
        parameters["qr_condition_warning_limit"] = float(_CONDITION_WARNING_LIMIT)
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


def _propagate(jacobian: FloatArray, tangent: FloatArray) -> FloatArray:
    """Advance a tangent vector or basis by one step without NumPy's overflow noise.

    A reorthogonalization interval that is too long overflows the tangent, which is
    detected and reported with an actionable :class:`NumericalError` a few lines
    later. NumPy would otherwise emit a bare ``RuntimeWarning: overflow encountered
    in matmul`` first, which says nothing about the remedy and, on NumPy 1.x, fires
    where NumPy 2.x stays quiet. Silencing it here keeps the reported failure the
    same on every supported version.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        return jacobian @ tangent


def _reorthogonalize(
    basis: FloatArray,
    *,
    interval: int,
    context: str,
) -> tuple[FloatArray, FloatArray, float]:
    """Return a QR-refreshed basis, the absolute ``R`` diagonal, and its condition.

    ``np.linalg.qr`` propagates an overflowed basis into ``inf``/``nan`` entries,
    and a ``nan`` passes every ``<=`` comparison, so the singularity guard alone
    used to let ``nan`` exponents escape into the reported result. Both the input
    basis and the resulting diagonal are therefore checked for finiteness.

    The returned condition number ``max(diag) / min(diag)`` is the diagnostic for
    the subtler failure. Between refreshes the leading direction outgrows the
    trailing one by ``exp((lambda_1 - lambda_d) * interval)``; once that ratio
    passes the float64 resolution, the trailing columns are rounding noise and
    their exponents are wrong while the basis stays perfectly orthogonal, so the
    orthogonality defect reports nothing. Measured on the cat map, an interval of
    40 returns ``+0.81`` for a trailing exponent whose exact value is ``-0.96``.
    Beyond ``eps**-2`` the trailing logarithms carry no significant digits at all
    and this raises rather than returning them.
    """
    if not bool(np.all(np.isfinite(basis))):
        raise NumericalError(
            f"Lyapunov {context} basis overflowed to a non-finite value before "
            f"reorthogonalization: the tangent directions grew beyond the float64 range "
            f"while propagating {interval} steps between QR refreshes. Reduce "
            f"reorthogonalization_interval (currently {interval})"
        )
    basis, triangular = np.linalg.qr(basis)
    diagonal = np.abs(np.diag(triangular))
    if not bool(np.all(np.isfinite(diagonal))):
        raise NumericalError(
            f"Lyapunov {context} QR factor has non-finite diagonal entries, so the "
            f"accumulated exponents would be nan. Reduce reorthogonalization_interval "
            f"(currently {interval})"
        )
    if bool(np.any(diagonal <= np.finfo(np.float64).tiny)):
        raise NumericalError(
            f"Lyapunov QR factor became singular during the {context} stage: a tangent "
            f"direction collapsed to zero length, so its exponent is undefined. If the "
            f"dynamics are not degenerate, reduce reorthogonalization_interval "
            f"(currently {interval})"
        )
    condition = float(diagonal.max() / diagonal.min())
    if condition > _CONDITION_ERROR_LIMIT:
        raise NumericalError(
            f"Lyapunov QR factor is too ill-conditioned during the {context} stage to "
            f"resolve the trailing exponents: the R diagonal spans a factor of "
            f"{condition:.3e}, beyond the float64 limit of {_CONDITION_ERROR_LIMIT:.3e}, "
            f"so the smaller exponents would carry no significant digits. Reduce "
            f"reorthogonalization_interval (currently {interval})"
        )
    return basis, diagonal, condition


def _safe_norm(vector: FloatArray, *, context: str, interval: int | None = None) -> float:
    """Return a Euclidean norm computed without squaring the largest component.

    ``np.linalg.norm`` sums squares, so it overflows once any component exceeds
    about ``1e154`` even though the norm itself is representable. Scaling by the
    largest absolute value first keeps the full exponent range available before
    the tangent vector genuinely exceeds ``float64``.
    """
    tiny = np.finfo(np.float64).tiny
    largest = float(np.max(np.abs(vector))) if vector.size else 0.0
    if not np.isfinite(largest):
        raise NumericalError(_overflow_message(context, interval))
    if largest <= tiny:
        raise NumericalError(
            f"{context} has zero or non-finite norm: the tangent collapsed to a zero "
            f"vector, so no growth direction remains"
        )
    norm = largest * float(np.linalg.norm(vector / largest))
    if not np.isfinite(norm) or norm <= tiny:
        raise NumericalError(_overflow_message(context, interval))
    return norm


def _overflow_message(context: str, interval: int | None) -> str:
    remedy = "Reduce reorthogonalization_interval" + (
        "" if interval is None else f" (currently {interval})"
    )
    return (
        f"{context} is no longer finite: it grew beyond the float64 range before being "
        f"renormalized, so its growth rate cannot be measured. {remedy}"
    )


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

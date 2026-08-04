"""Correlation and transport statistics with explicit averaging conventions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from chaos_numerics.core import AnalysisResult, ExperimentMetadata, Trajectory, ValidationError
from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.types import ArrayLike, FloatArray

Observable = Callable[[FloatArray], ArrayLike]


def autocorrelation(
    data: Trajectory | ArrayLike,
    *,
    observable: Observable | None = None,
    max_lag: int | None = None,
    demean: bool = True,
    normalize: bool = False,
) -> AnalysisResult:
    """Return direct time-origin-averaged autocorrelation for lags ``0..max_lag``."""
    series = _observable_series(data, observable=observable, name="autocorrelation")
    count = series.size
    lag_count = _max_lag(max_lag, count=count)
    centered = series - np.mean(series) if demean else series
    values = np.empty(lag_count + 1, dtype=np.float64)
    for lag in range(lag_count + 1):
        values[lag] = float(np.mean(centered[: count - lag] * centered[lag:]))
    if normalize:
        if values[0] <= np.finfo(np.float64).tiny:
            raise ValidationError("cannot normalize autocorrelation with zero lag-zero value")
        values /= values[0]
    metadata = ExperimentMetadata(
        parameters={
            "max_lag": lag_count,
            "demean": demean,
            "normalize": normalize,
            "observable": _observable_name(observable),
            "averaging": "all_time_origins",
        }
    )
    return AnalysisResult("autocorrelation", values, metadata=metadata)


def mean_square_displacement(
    positions: Trajectory | ArrayLike,
    *,
    observable: Observable | None = None,
    max_lag: int | None = None,
    time_origin_average: bool = True,
    unwrap: bool = False,
    periods: ArrayLike | None = None,
) -> AnalysisResult:
    """Return MSD for lags ``0..max_lag`` using coordinate-wise displacements."""
    points = _position_series(positions, observable=observable)
    count, dimension = points.shape
    lag_count = _max_lag(max_lag, count=count)
    if unwrap:
        points = _unwrap(points, periods=periods, dimension=dimension)
    elif periods is not None:
        raise ValidationError("periods may be supplied only when unwrap=True")

    values = np.empty(lag_count + 1, dtype=np.float64)
    uncertainty = np.empty(lag_count + 1, dtype=np.float64)
    for lag in range(lag_count + 1):
        if time_origin_average:
            displacements = points[lag:] - points[: count - lag]
        else:
            displacements = points[lag : lag + 1] - points[0]
        squared = np.sum(displacements * displacements, axis=1)
        values[lag] = float(np.mean(squared))
        uncertainty[lag] = (
            float(np.std(squared, ddof=1) / np.sqrt(squared.size)) if squared.size > 1 else 0.0
        )
    metadata = ExperimentMetadata(
        parameters={
            "max_lag": lag_count,
            "time_origin_average": time_origin_average,
            "unwrap": unwrap,
            "periods": None if periods is None else _periods(periods, dimension).tolist(),
            "observable": _observable_name(observable),
        }
    )
    return AnalysisResult(
        "mean_square_displacement", values, uncertainty=uncertainty, metadata=metadata
    )


def local_diffusion_exponent(
    msd: AnalysisResult | ArrayLike,
    *,
    fit_start: int,
    fit_stop: int,
    times: ArrayLike | None = None,
) -> AnalysisResult:
    """Fit ``MSD(t) = exp(intercept) * t**alpha`` on ``[fit_start, fit_stop)``."""
    values = msd.values if isinstance(msd, AnalysisResult) else as_float_array(msd, name="msd")
    if values.ndim != 1:
        raise ValidationError(f"msd must be one-dimensional; got shape {values.shape}")
    start = _nonnegative_int(fit_start, name="fit_start")
    stop = _nonnegative_int(fit_stop, name="fit_stop")
    if not 0 <= start < stop <= values.size:
        raise ValidationError(f"fit interval [{start}, {stop}) is outside {values.size} samples")
    time_values = (
        np.arange(values.size, dtype=np.float64)
        if times is None
        else as_float_array(times, name="times", ndim=1)
    )
    if time_values.shape != values.shape:
        raise ValidationError(f"times must have shape {values.shape}; got {time_values.shape}")
    selected_t = time_values[start:stop]
    selected_msd = values[start:stop]
    valid = (selected_t > 0.0) & (selected_msd > 0.0)
    if int(np.count_nonzero(valid)) < 3:
        raise ValidationError("diffusion fit requires at least three positive time/MSD points")
    x = np.log(selected_t[valid])
    y = np.log(selected_msd[valid])
    design = np.column_stack((x, np.ones_like(x)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2 or not bool(np.all(np.isfinite(coefficients))):
        raise ValidationError("diffusion fit is rank-deficient or non-finite")
    fitted = design @ coefficients
    errors = y - fitted
    rms = float(np.sqrt(np.mean(errors * errors)))
    degrees = x.size - 2
    slope_error = (
        float(np.sqrt(np.sum(errors * errors) / degrees / np.sum((x - np.mean(x)) ** 2)))
        if degrees > 0
        else 0.0
    )
    metadata = ExperimentMetadata(
        parameters={
            "fit_start": start,
            "fit_stop": stop,
            "sample_count": int(x.size),
            "intercept": float(coefficients[1]),
            "fit_model": "log(msd) = alpha * log(time) + intercept",
        }
    )
    return AnalysisResult(
        "local_diffusion_exponent",
        np.asarray([coefficients[0]], dtype=np.float64),
        uncertainty=np.asarray([slope_error], dtype=np.float64),
        residuals=np.asarray([rms], dtype=np.float64),
        metadata=metadata,
    )


def _observable_series(
    data: Trajectory | ArrayLike,
    *,
    observable: Observable | None,
    name: str,
) -> FloatArray:
    raw = data.states if isinstance(data, Trajectory) else as_float_array(data, name=name)
    transformed = observable(raw) if observable is not None else raw
    series = as_float_array(transformed, name=f"{name} observable")
    if series.ndim != 1:
        raise ValidationError(
            f"{name} observable must produce a one-dimensional time series; got {series.shape}"
        )
    if series.size < 2:
        raise ValidationError(f"{name} requires at least two time samples")
    return series


def _position_series(data: Trajectory | ArrayLike, *, observable: Observable | None) -> FloatArray:
    raw = data.states if isinstance(data, Trajectory) else as_float_array(data, name="positions")
    transformed = observable(raw) if observable is not None else raw
    points = as_float_array(transformed, name="position observable")
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValidationError(
            "position observable must produce shape (time,) or (time, dimension) with at least two times"
        )
    return points


def _unwrap(points: FloatArray, *, periods: ArrayLike | None, dimension: int) -> FloatArray:
    if periods is None:
        raise ValidationError("periods are required when unwrap=True")
    widths = _periods(periods, dimension)
    differences = np.diff(points, axis=0)
    differences -= widths * np.floor(differences / widths + 0.5)
    result = np.empty_like(points)
    result[0] = points[0]
    result[1:] = points[0] + np.cumsum(differences, axis=0)
    return result


def _periods(value: ArrayLike, dimension: int) -> FloatArray:
    widths = as_float_array(value, name="periods")
    if widths.ndim == 0:
        widths = np.full(dimension, float(widths), dtype=np.float64)
    if widths.shape != (dimension,) or bool(np.any(widths <= 0.0)):
        raise ValidationError(
            f"periods must be positive with shape ({dimension},); got {widths.shape}"
        )
    return widths


def _max_lag(value: int | None, *, count: int) -> int:
    result = count - 1 if value is None else _nonnegative_int(value, name="max_lag")
    if result >= count:
        raise ValidationError(f"max_lag must be less than sample count {count}; got {result}")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValidationError(f"{name} must be non-negative; got {result}")
    return result


def _observable_name(observable: Observable | None) -> str | None:
    return (
        None if observable is None else getattr(observable, "__name__", type(observable).__name__)
    )


__all__ = ["autocorrelation", "local_diffusion_exponent", "mean_square_displacement"]

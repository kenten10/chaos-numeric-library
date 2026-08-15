"""Correlation and transport statistics with explicit averaging conventions.

Shape conventions
-----------------
:func:`autocorrelation` consumes a scalar series whose **last** axis is time:
``(time,)`` or ``(*batch, time)``. :func:`mean_square_displacement` consumes
coordinates whose last axis is the coordinate index and whose second-to-last axis
is time: ``(time,)`` (promoted to one coordinate), ``(time, dimension)``, or
``(*batch, time, dimension)``. A two-dimensional input to the latter is therefore
always ``(time, dimension)``, never a batch of scalar series; add a leading axis
to make a batch explicit.

Every leading axis is flattened into a single ensemble axis of size
``batch_size = prod(batch)``, so ``(2, 3, time, dimension)`` is an ensemble of six
trajectories. The reported curve is the ensemble average of the per-trajectory
curves, each of which is itself averaged over time origins when requested:
``value(lag) = (1 / batch_size) * sum_b curve_b(lag)``. A batch of one is exactly
the non-batched computation, bit for bit.

Uncertainty
-----------
:func:`autocorrelation` and :func:`mean_square_displacement` report their error
bar under one shared rule: ``uncertainty`` is the standard error of the mean over
the independent trajectories in the batch,
``std(curves, ddof=1) / sqrt(batch_size)``, which assumes only that the
trajectories are independent. **A single trajectory therefore gets
``uncertainty=None``** together with a
:class:`~chaos_numerics.core.NumericalWarning`, and every result records which of
the two cases applied in ``metadata.parameters["error_semantics"]``.
:func:`local_diffusion_exponent` follows the same rule one level up, over curves
rather than trajectories.

:func:`mean_square_displacement` used to fall back to the scatter over time
origins when it was given one trajectory. That estimator was measured against
synthetic data whose answer is known exactly -- 400 unit random walks of 2000
steps, whose MSD is ``<dx**2> = t`` -- and it is not calibrated. The ratio of the
true spread of the estimator to the reported error grew from ``1.05`` at lag 1
through ``2.64`` at lag 10 and ``8.76`` at lag 100 to ``20.9`` at lag 500, and the
fraction of realizations more than two reported sigma from the truth grew from
``4.8%`` to ``90.0%`` where ``5%`` was expected. The time origins of one
trajectory are strongly correlated, which is the same failure that makes an
ordinary-least-squares error on a log-log MSD fit about thirty times too small in
:func:`local_diffusion_exponent`. The batch standard error passes the same test on
the same data: with 16 trajectories the ratio stayed within ``0.98``-``1.13`` and
the tail within ``4.5%``-``9.5%`` over lags 1 to 500, inside the section 4.3
criterion of ``docs/design/numerical-standards.md`` (ratio in ``[0.5, 2]``, at most
``20%`` beyond two sigma) that the single-trajectory estimator failed.
"""

from __future__ import annotations

import warnings as python_warnings
from collections.abc import Callable
from typing import Literal, TypeAlias, cast

import numpy as np

from chaos_numerics.core import (
    AnalysisResult,
    ExperimentMetadata,
    NumericalError,
    NumericalWarning,
    Trajectory,
    ValidationError,
)
from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.types import ArrayLike, FloatArray

Observable: TypeAlias = Callable[[FloatArray], ArrayLike]
"""Reduction from raw states to the series an analysis should consume.

An observable receives the whole ``float64`` array that was passed in -- for a
:class:`~chaos_numerics.core.Trajectory` that is ``states``, with shape
``(*batch, time, state_dim)`` -- and returns the series to analyse. Write it with
trailing-axis indexing so that it works batched and unbatched alike, for example
``lambda states: states[..., 1]`` to correlate the momentum coordinate. Its
``__name__`` is recorded in ``metadata.parameters["observable"]``.
"""

Method: TypeAlias = Literal["auto", "direct", "fft"]
"""Algorithm selector shared by :func:`autocorrelation` and
:func:`mean_square_displacement`."""

_METHODS: tuple[str, ...] = ("auto", "direct", "fft")

# Reconstructed one-step displacements within this fraction of half a period are
# indistinguishable from aliased ones, so :func:`_unwrap` refuses to guess.
_ALIASING_MARGIN = 0.01

# Crossover used by ``method="auto"``, in units of (time samples) * (lags).
#
# The direct loop costs one length-``time`` pass per lag; a real FFT pair costs
# O(time log time) regardless of the number of lags. Measured on an Apple M-series
# CPU with NumPy 2.x, one autocorrelation of a float64 series takes:
#
#     time    lags    direct     fft
#      2000      50    0.20 ms   0.05 ms
#      2000     200    0.74 ms   0.05 ms
#     20000      10    0.16 ms   0.75 ms
#     20000   19999  139.95 ms   0.65 ms
#     40000   39999  429.32 ms   1.48 ms
#
# ``time * lags = 250_000`` sits inside the band where the two are within a small
# factor of each other: picking the slower branch there costs well under a
# millisecond, while the same rule saves 0.43 s at ``time = 40_000`` and turns the
# extrapolated 210 s of a ``time = 10**6`` full-lag correlation into milliseconds.
_FFT_CROSSOVER = 250_000


def autocorrelation(
    data: Trajectory | ArrayLike,
    *,
    observable: Observable | None = None,
    max_lag: int | None = None,
    demean: bool = True,
    normalize: bool = False,
    method: Method = "auto",
) -> AnalysisResult:
    """Return the time-origin-averaged autocorrelation for lags ``0..max_lag``.

    The series is ``(time,)`` or ``(*batch, time)``; see the module docstring for
    the shape and ensemble-averaging conventions. ``demean`` subtracts each
    trajectory's own time mean, the ensemble curve is the mean of the
    per-trajectory curves, and ``normalize`` divides that curve by its lag-zero
    value.

    Uncertainty
    -----------
    ``uncertainty`` is the standard error of the mean over the independent
    trajectories in the batch, ``std(curves, ddof=1) / sqrt(batch_size)``, and it
    assumes only that the trajectories are independent. **A single trajectory
    therefore gets ``uncertainty=None``** and a
    :class:`~chaos_numerics.core.NumericalWarning`, because one trajectory carries
    no ensemble spread and its scatter over time origins is not a calibrated
    substitute; the module docstring records that measurement.
    :func:`mean_square_displacement` reports its error under exactly this rule, so
    the two agree, and ``metadata.parameters["error_semantics"]`` states which case
    produced the returned ``uncertainty``.

    ``method`` selects the algorithm and never the answer. ``"direct"`` is the
    O(time * lags) reference loop, ``"fft"`` is the Wiener-Khinchin
    O(time log time) route, and ``"auto"`` -- the default -- takes ``"fft"`` once
    ``time * max_lag`` exceeds 250 000. The method actually used is recorded in
    ``metadata.parameters["method"]``.
    """
    series, batch_size = _observable_series(data, observable=observable, name="autocorrelation")
    count = series.shape[-1]
    lag_count = _max_lag(max_lag, count=count)
    resolved = _resolve_method(method, count=count, lag_count=lag_count)
    with np.errstate(over="ignore", invalid="ignore"):
        centered = series - np.mean(series, axis=-1, keepdims=True) if demean else series
        curves = (
            _fft_autocorrelation(centered, lag_count=lag_count)
            if resolved == "fft"
            else _direct_autocorrelation(centered, lag_count=lag_count)
        )
    _require_finite_transport(
        curves,
        name="autocorrelation",
        quantity="the lag products x(t) * x(t + lag)",
        scale=_magnitude(series),
        remedy=(
            "Rescale the series before calling: the autocorrelation scales as its square, so "
            "dividing the series by c divides every value by c**2"
        ),
    )
    values = np.mean(curves, axis=0)
    # Normalization is settled before the error bar so that a rejected normalize
    # request raises instead of first emitting the single-trajectory warning.
    if normalize:
        if values[0] <= np.finfo(np.float64).tiny:
            raise ValidationError("cannot normalize autocorrelation with zero lag-zero value")
        values = values / values[0]
        # Each trajectory is normalized by *its own* lag-zero value before the
        # spread is taken. Dividing the unnormalized standard error by the ensemble
        # lag-zero value instead ignores that the denominator is itself estimated
        # and varies from trajectory to trajectory, which made the reported error
        # wrong in both directions at once: measured on an AR(1) series with
        # phi = 0.9 and batch = 16, the true spread was only 0.11, 0.20 and 0.44 of
        # what was reported at lags 1, 2 and 5, while at lag zero the normalized
        # value is identically 1 -- zero spread by construction -- and a nonzero
        # error of about 0.024 was reported for it anyway.
        flat = curves.reshape(-1, curves.shape[-1])
        usable = flat[:, 0] > np.finfo(np.float64).tiny
        dropped = int(flat.shape[0] - int(np.count_nonzero(usable)))
        if dropped:
            # A trajectory that never moves -- one launched exactly on a fixed
            # point, say -- has a zero lag-zero value once demeaned, so it has no
            # normalized curve of its own. The ensemble curve is still perfectly
            # well defined, because the ensemble lag-zero value is not zero; only
            # that member cannot contribute to the spread.
            python_warnings.warn(
                f"autocorrelation dropped {dropped} of {flat.shape[0]} trajectories "
                "from the normalized error bar because their lag-zero value is zero, "
                "so they have no normalized curve. The reported values still average "
                "over every trajectory; only the error bar is narrowed to the ones "
                "that carry a curve",
                NumericalWarning,
                stacklevel=2,
            )
        selected = flat[usable]
        curves = selected / selected[:, :1]
        batch_size = int(selected.shape[0])
    uncertainty, semantics = _trajectory_uncertainty(
        curves,
        batch_size=batch_size,
        name="autocorrelation",
        shape="(*batch, time)",
    )
    metadata = ExperimentMetadata(
        parameters={
            "max_lag": lag_count,
            "demean": demean,
            "normalize": normalize,
            "observable": _observable_name(observable),
            "averaging": "all_time_origins",
            "method": resolved,
            "batch_size": batch_size,
            "error_semantics": _NORMALIZED_ENSEMBLE_SEMANTICS
            if normalize and uncertainty is not None
            else semantics,
        }
    )
    return AnalysisResult("autocorrelation", values, uncertainty=uncertainty, metadata=metadata)


def mean_square_displacement(
    positions: Trajectory | ArrayLike,
    *,
    observable: Observable | None = None,
    max_lag: int | None = None,
    time_origin_average: bool = True,
    unwrap: bool = False,
    periods: ArrayLike | None = None,
    method: Method = "auto",
    per_trajectory: bool = False,
) -> AnalysisResult:
    """Return MSD for lags ``0..max_lag`` using coordinate-wise displacements.

    Positions are ``(time,)``, ``(time, dimension)``, or
    ``(*batch, time, dimension)``; see the module docstring for the shape and
    ensemble-averaging conventions. A batched input is the standard transport
    measurement: each trajectory contributes
    ``curve_b(lag) = mean_t |x_b(t + lag) - x_b(t)|**2`` and the reported curve is
    their ensemble mean.

    Uncertainty
    -----------
    ``uncertainty`` is the standard error of the mean over the independent
    trajectories in the batch, ``std(curves, ddof=1) / sqrt(batch_size)``, and it
    assumes only that the trajectories are independent. **A single trajectory
    therefore gets ``uncertainty=None``** and a
    :class:`~chaos_numerics.core.NumericalWarning`. Earlier versions returned the
    scatter over time origins there; the module docstring records the measurement
    that retired it, which found the reported error up to twenty-one times too
    small with 90% of realizations beyond two of its sigma. :func:`autocorrelation`
    reports its error under exactly this rule, so the two agree, and
    ``metadata.parameters["error_semantics"]`` states which case produced the
    returned ``uncertainty``.

    ``per_trajectory=True`` skips the ensemble average and returns the individual
    curves instead: ``values`` has shape ``(batch_size, max_lag + 1)`` and
    ``uncertainty`` is ``None`` without a warning, because the spread is now in the
    rows rather than in an error bar. That is the input
    :func:`local_diffusion_exponent` needs in order to report a calibrated error
    on the fitted exponent, so the two compose directly::

        curves = mean_square_displacement(trajectory, per_trajectory=True)
        alpha = local_diffusion_exponent(curves, fit_start=10, fit_stop=200)

    With ``unwrap=True`` the periodic coordinates are reconstructed by the
    minimum-image rule, which requires ``periods``. ``periods`` accepts either
    one width per coordinate or a single scalar that applies to every coordinate,
    so ``periods=1.0`` is equivalent to ``periods=[1.0] * dimension``.

    Minimum-image unwrapping is only unique while every one-step displacement
    stays below half its period, judged over the whole ensemble. Sampling too
    coarse for that limit is rejected with
    :class:`~chaos_numerics.core.ValidationError` instead of silently returning
    an aliased, far too small displacement.

    ``method`` selects the algorithm and never the answer: ``"direct"`` is the
    O(time * lags) reference loop and ``"fft"`` evaluates the identity
    ``MSD(lag) = (S1(lag) - 2 * C(lag)) / (time - lag)`` -- with ``S1`` from
    cumulative sums of squared norms and ``C`` the unnormalized cross-correlation
    from :func:`numpy.fft.rfft` -- in O(time log time). ``"auto"`` takes ``"fft"``
    once ``time * max_lag`` exceeds 250 000. With ``time_origin_average=False``
    there is only one algorithm and the recorded method is ``"single_origin"``.
    """
    if not isinstance(per_trajectory, bool):
        raise ValidationError(f"per_trajectory must be a bool; got {per_trajectory!r}")
    points, batch_size = _position_series(positions, observable=observable)
    count, dimension = points.shape[-2], points.shape[-1]
    lag_count = _max_lag(max_lag, count=count)
    if unwrap:
        points = _unwrap(points, periods=periods, dimension=dimension)
    elif periods is not None:
        raise ValidationError("periods may be supplied only when unwrap=True")

    with np.errstate(over="ignore", invalid="ignore"):
        if not time_origin_average:
            recorded_method = "single_origin"
            curves = _single_origin_displacement(points, lag_count=lag_count)
        else:
            recorded_method = _resolve_method(method, count=count, lag_count=lag_count)
            compute = _fft_msd if recorded_method == "fft" else _direct_msd
            curves = compute(points, lag_count=lag_count)
    _require_finite_transport(
        curves,
        name="mean_square_displacement",
        quantity="the squared displacements |x(t + lag) - x(t)|**2",
        scale=_magnitude(points),
        remedy=_MSD_REMEDY,
    )

    values: FloatArray
    uncertainty: FloatArray | None
    if per_trajectory:
        values = curves
        uncertainty, semantics = None, _PER_TRAJECTORY_SEMANTICS
    else:
        values = np.mean(curves, axis=0)
        uncertainty, semantics = _trajectory_uncertainty(
            curves,
            batch_size=batch_size,
            name="mean_square_displacement",
            shape="(*batch, time, dimension)",
        )
    metadata = ExperimentMetadata(
        parameters={
            "max_lag": lag_count,
            "time_origin_average": time_origin_average,
            "unwrap": unwrap,
            "periods": None if periods is None else _periods(periods, dimension).tolist(),
            "observable": _observable_name(observable),
            "method": recorded_method,
            "batch_size": batch_size,
            "per_trajectory": per_trajectory,
            "error_semantics": semantics,
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
    """Fit ``MSD(t) = exp(intercept) * t**alpha`` on ``[fit_start, fit_stop)``.

    ``msd`` is either one MSD curve -- a one-dimensional array or an
    :class:`~chaos_numerics.core.AnalysisResult` from
    :func:`mean_square_displacement` -- or an ensemble of them with shape
    ``(curve_count, time)``, which is what
    ``mean_square_displacement(..., per_trajectory=True)`` returns. ``values``
    always holds the single exponent fitted to the ensemble-*mean* curve, which is
    bit-identical to fitting the collapsed curve of the same batch.

    Uncertainty
    -----------
    ``uncertainty`` is the standard error of the mean over the per-curve fits,
    ``std(alpha_b, ddof=1) / sqrt(curve_count)``, and it assumes only that the
    trajectories behind the curves are independent. That is the only assumption
    the data supports, so it is the only error bar this function reports.

    **A single curve therefore gets ``uncertainty=None``** and a
    :class:`~chaos_numerics.core.NumericalWarning`. The obvious alternative -- the
    ordinary-least-squares standard error of the log-log fit -- assumes the fit
    residuals are independent draws, and a time-origin-averaged MSD violates that
    badly: it is a smooth curve whose neighbouring lags are strongly correlated,
    so the residuals almost cancel and the error bar comes out roughly 30 times
    too small. Measured over 200 realizations of unit random walks of 2000 steps
    fitted over lags 10..200, the OLS error averaged 0.0048 against a true
    estimator spread of 0.142 for one trajectory and 0.00066 against 0.0178 for a
    batch of 64, which puts 93% and 96% of realizations more than two nominal sigma
    away from the true exponent ``alpha = 1``. Batching the *input* to
    :func:`mean_square_displacement` does not help, because the collapsed curve is
    just as smooth.

    The recommended procedure is therefore to keep the trajectories separate::

        curves = mean_square_displacement(ensemble, per_trajectory=True)
        alpha = local_diffusion_exponent(curves, fit_start=10, fit_stop=200)

    which reports 0.0183 against a true spread of 0.0174 for the same 64-trajectory
    random walk, with 3.5% of realizations beyond two sigma; the same measurement on
    fractional Brownian motion with ``H = 0.75``, whose MSD is exactly ``t**1.5``,
    gives 0.0201 against 0.0198 and 3.0%. ``residuals`` stays the RMS
    of the log-log fit residuals of the mean curve: it measures how well a single
    power law describes the window, not how well ``alpha`` is determined.
    ``metadata.parameters["error_semantics"]`` records which of the two situations
    produced the returned ``uncertainty``.
    """
    values = msd.values if isinstance(msd, AnalysisResult) else as_float_array(msd, name="msd")
    if values.ndim == 1:
        curves = values[None, :]
    elif values.ndim == 2:
        curves = values
    else:
        raise ValidationError(
            "msd must be one-dimensional (a single curve) or two-dimensional "
            f"(curve_count, time); got shape {values.shape}"
        )
    curve_count, sample_count = int(curves.shape[0]), int(curves.shape[1])
    if curve_count == 0:
        raise ValidationError("msd must contain at least one curve")
    start = _nonnegative_int(fit_start, name="fit_start")
    stop = _nonnegative_int(fit_stop, name="fit_stop")
    if not 0 <= start < stop <= sample_count:
        raise ValidationError(f"fit interval [{start}, {stop}) is outside {sample_count} samples")
    time_values = (
        np.arange(sample_count, dtype=np.float64)
        if times is None
        else as_float_array(times, name="times", ndim=1)
    )
    if time_values.shape != (sample_count,):
        raise ValidationError(f"times must have shape ({sample_count},); got {time_values.shape}")
    selected_t = time_values[start:stop]
    mean_curve = curves[0] if curve_count == 1 else np.mean(curves, axis=0)
    slope, intercept, rms, used = _log_log_fit(selected_t, mean_curve[start:stop], context="")
    uncertainty, semantics = _exponent_uncertainty(curves, selected_t, start=start, stop=stop)
    metadata = ExperimentMetadata(
        parameters={
            "fit_start": start,
            "fit_stop": stop,
            "sample_count": used,
            "curve_count": curve_count,
            "intercept": intercept,
            "fit_model": "log(msd) = alpha * log(time) + intercept",
            "error_semantics": semantics,
        }
    )
    return AnalysisResult(
        "local_diffusion_exponent",
        np.asarray([slope], dtype=np.float64),
        uncertainty=uncertainty,
        residuals=np.asarray([rms], dtype=np.float64),
        metadata=metadata,
    )


_SINGLE_CURVE_SEMANTICS = "none: a single MSD curve cannot calibrate the error on alpha"
_ENSEMBLE_SEMANTICS = "standard error of the mean over per-curve fits"

# The one convention shared by autocorrelation and mean_square_displacement.
_TRAJECTORY_ENSEMBLE_SEMANTICS = "standard error of the mean over trajectories"
_NORMALIZED_ENSEMBLE_SEMANTICS = (
    "standard error of the mean over trajectories, each normalized by its own lag-zero value"
)
_SINGLE_TRAJECTORY_SEMANTICS = (
    "none: a single trajectory cannot calibrate the error on a time-origin-averaged curve"
)
_PER_TRAJECTORY_SEMANTICS = (
    "none: per_trajectory=True returns the individual curves, whose spread is the error"
)

_SINGLE_TRAJECTORY_WARNING = (
    "{name} received a single trajectory, so it reports uncertainty=None. The only "
    "calibrated error bar here is the spread across independent trajectories. The "
    "obvious alternative -- the scatter over the time origins of one trajectory -- "
    "assumes the origins are independent draws, and the origins of a "
    "time-origin-averaged curve are strongly correlated: measured over 400 unit "
    "random walks of 2000 steps, whose MSD is exactly <dx**2> = t, it understated "
    "the true spread by factors of 1.0, 2.6, 8.8 and 20.9 at lags 1, 10, 100 and "
    "500, leaving 4.8%, 46%, 82% and 90% of realizations more than two of its sigma "
    "from the truth where 5% was expected. Pass an ensemble of shape {shape} to get "
    "the standard error of the mean over trajectories instead"
)


def _trajectory_uncertainty(
    curves: FloatArray,
    *,
    batch_size: int,
    name: str,
    shape: str,
) -> tuple[FloatArray | None, str]:
    """Return the across-trajectory standard error, or ``None`` for one trajectory."""
    if batch_size < 2:
        python_warnings.warn(
            _SINGLE_TRAJECTORY_WARNING.format(name=name, shape=shape),
            NumericalWarning,
            stacklevel=3,
        )
        return None, _SINGLE_TRAJECTORY_SEMANTICS
    return _batch_standard_error(curves), _TRAJECTORY_ENSEMBLE_SEMANTICS


_SINGLE_CURVE_WARNING = (
    "local_diffusion_exponent received a single MSD curve, so it reports "
    "uncertainty=None. A trustworthy error bar on the exponent needs an ensemble: "
    "the least-squares standard error of a log-log fit assumes independent "
    "residuals, but a time-origin-averaged MSD is a smooth curve whose lags are "
    "strongly correlated, and using it understates the spread of alpha by about a "
    "factor of 30 (measured on 2000-step random walks fitted over lags 10..200). "
    "Pass mean_square_displacement(ensemble, per_trajectory=True), or any "
    "(curve_count, time) array of independent MSD curves, to get the standard "
    "error of the mean over the per-curve fits instead"
)


def _exponent_uncertainty(
    curves: FloatArray,
    selected_t: FloatArray,
    *,
    start: int,
    stop: int,
) -> tuple[FloatArray | None, str]:
    """Return the across-curve standard error, or ``None`` for a single curve."""
    curve_count = int(curves.shape[0])
    if curve_count < 2:
        python_warnings.warn(_SINGLE_CURVE_WARNING, NumericalWarning, stacklevel=3)
        return None, _SINGLE_CURVE_SEMANTICS
    slopes = np.asarray(
        [
            _log_log_fit(selected_t, curves[index, start:stop], context=f" (curve {index})")[0]
            for index in range(curve_count)
        ],
        dtype=np.float64,
    )
    error = float(np.std(slopes, ddof=1) / np.sqrt(float(curve_count)))
    return np.asarray([error], dtype=np.float64), _ENSEMBLE_SEMANTICS


def _log_log_fit(
    selected_t: FloatArray,
    selected_msd: FloatArray,
    *,
    context: str,
) -> tuple[float, float, float, int]:
    """Return ``(slope, intercept, residual_rms, points_used)`` of one log-log fit."""
    valid = (selected_t > 0.0) & (selected_msd > 0.0)
    if int(np.count_nonzero(valid)) < 3:
        raise ValidationError(
            f"diffusion fit requires at least three positive time/MSD points{context}"
        )
    x = np.log(selected_t[valid])
    y = np.log(selected_msd[valid])
    design = np.column_stack((x, np.ones_like(x)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2 or not bool(np.all(np.isfinite(coefficients))):
        raise ValidationError(f"diffusion fit is rank-deficient or non-finite{context}")
    errors = y - design @ coefficients
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(np.sqrt(np.mean(errors * errors))),
        int(x.size),
    )


def _direct_autocorrelation(centered: FloatArray, *, lag_count: int) -> FloatArray:
    """Reference O(batch * time * lags) loop, one dot product per lag."""
    count = centered.shape[-1]
    curves = np.empty((centered.shape[0], lag_count + 1), dtype=np.float64)
    for lag in range(lag_count + 1):
        curves[:, lag] = np.mean(centered[:, : count - lag] * centered[:, lag:], axis=-1)
    return curves


def _fft_autocorrelation(centered: FloatArray, *, lag_count: int) -> FloatArray:
    """Wiener-Khinchin autocorrelation with the same ``time - lag`` divisor."""
    count = centered.shape[-1]
    correlation = _correlate(centered, count=count, lag_count=lag_count)
    return correlation / _origin_counts(count, lag_count)


def _single_origin_displacement(points: FloatArray, *, lag_count: int) -> FloatArray:
    displacements = points[:, : lag_count + 1, :] - points[:, :1, :]
    return np.sum(displacements * displacements, axis=-1)


def _direct_msd(points: FloatArray, *, lag_count: int) -> FloatArray:
    """Reference O(batch * time * lags * dimension) loop over lags."""
    count = points.shape[-2]
    curves = np.empty((points.shape[0], lag_count + 1), dtype=np.float64)
    for lag in range(lag_count + 1):
        displacements = points[:, lag:, :] - points[:, : count - lag, :]
        squared = np.sum(displacements * displacements, axis=-1)
        curves[:, lag] = np.mean(squared, axis=-1)
    return curves


def _fft_msd(points: FloatArray, *, lag_count: int) -> FloatArray:
    """Evaluate ``MSD(lag) = (S1(lag) - 2 C(lag)) / (time - lag)`` by FFT.

    ``S1`` collects the squared norms of both endpoints and follows from prefix
    sums; ``C`` is the unnormalized cross-correlation of each coordinate with
    itself. Both are sums of ``|x|**2`` while their difference is a squared
    *displacement*, so the identity cancels badly for coordinates far from the
    origin -- exactly the situation ``unwrap=True`` creates. Displacements are
    invariant under a constant shift, so each trajectory is recentred on its own
    time mean first, which keeps the cancelling terms the same order as the answer.
    Lag zero is set to its exact value instead of the cancelled one, and the
    remaining lags are projected onto the non-negativity that MSD has by
    definition.
    """
    count = points.shape[-2]
    dimension = points.shape[-1]
    divisor = _origin_counts(count, lag_count)
    points = points - np.mean(points, axis=-2, keepdims=True)
    square_norms = np.sum(points * points, axis=-1)
    endpoint_sum = _endpoint_sums(square_norms, lag_count=lag_count)
    correlation = np.zeros((points.shape[0], lag_count + 1), dtype=np.float64)
    for coordinate in range(dimension):
        column = points[..., coordinate]
        correlation += _correlate(column, count=count, lag_count=lag_count)
    curves = np.maximum((endpoint_sum - 2.0 * correlation) / divisor, 0.0)
    curves[:, 0] = 0.0
    return curves


def _endpoint_sums(values: FloatArray, *, lag_count: int) -> FloatArray:
    """Return ``sum_t (v[t] + v[t + lag])`` over the ``time - lag`` valid origins."""
    count = values.shape[-1]
    lags = np.arange(lag_count + 1)
    prefix = np.concatenate(
        (np.zeros((values.shape[0], 1), dtype=np.float64), np.cumsum(values, axis=-1)),
        axis=-1,
    )
    return prefix[:, count - lags] + prefix[:, -1:] - prefix[:, lags]


def _correlate(values: FloatArray, *, count: int, lag_count: int) -> FloatArray:
    """Return ``sum_t v[t] * v[t + lag]`` for ``lag`` in ``0..lag_count``.

    Zero padding to at least ``2 * count - 1`` samples turns the circular
    correlation of the FFT into the linear one, so no lag wraps around. Only the
    self-correlation is needed, so the spectrum is the real power spectrum rather
    than a general cross-spectrum.
    """
    size = 1 << max(2 * count - 1, 1).bit_length()
    forward = np.fft.rfft(values, size, axis=-1)
    spectrum = forward.real * forward.real + forward.imag * forward.imag
    transformed = cast(FloatArray, np.fft.irfft(spectrum, size, axis=-1))
    return transformed[..., : lag_count + 1]


def _origin_counts(count: int, lag_count: int) -> FloatArray:
    """Return the ``time - lag`` divisor shared by the direct and FFT routes."""
    counts: FloatArray = (count - np.arange(lag_count + 1)).astype(np.float64)
    return counts


# Squaring a float64 larger than this overflows, which is where the transport
# estimators stop losing precision and start losing the answer to ``inf``.
_SQUARING_LIMIT = float(np.sqrt(np.finfo(np.float64).max))

_MSD_REMEDY = (
    "Rescale the coordinates before calling: the MSD scales as their square, so dividing "
    "them by c divides every value by c**2. If unwrap=True is accumulating an unbounded "
    "drift, shorten the trajectory or analyse the wrapped coordinates instead"
)


def _magnitude(values: FloatArray) -> float:
    largest = float(np.max(np.abs(values))) if values.size else 0.0
    return largest


def _require_finite_transport(
    values: FloatArray,
    *,
    name: str,
    quantity: str,
    scale: float,
    remedy: str,
) -> None:
    """Name an overflow instead of leaking ``inf``/``nan`` into a result container.

    Without this the squared displacements of coordinates near ``1e160`` emit a
    bare ``RuntimeWarning: overflow encountered in multiply`` -- silenced by the
    surrounding :func:`numpy.errstate`, because it names neither the cause nor a
    remedy -- and the caller only learns of the failure from the generic
    "must contain only finite values" check inside
    :class:`~chaos_numerics.core.AnalysisResult`. This matches instead what
    ``classical.lyapunov`` does for its own overflow: an actionable
    :class:`~chaos_numerics.core.NumericalError`.
    """
    if bool(np.all(np.isfinite(values))):
        return
    raise NumericalError(
        f"{name} overflowed to a non-finite value while forming {quantity}: the input reaches "
        f"a magnitude of {scale:.3e}, and this route leaves the float64 range above "
        f"{_SQUARING_LIMIT:.3e}. {remedy}"
    )


def _batch_standard_error(curves: FloatArray) -> FloatArray:
    size = curves.shape[0]
    return cast(FloatArray, np.std(curves, axis=0, ddof=1) / np.sqrt(float(size)))


def _resolve_method(
    value: object,
    *,
    count: int,
    lag_count: int,
) -> Literal["direct", "fft"]:
    if not isinstance(value, str) or value not in _METHODS:
        raise ValidationError(
            f"method must be one of {', '.join(repr(name) for name in _METHODS)}; got {value!r}"
        )
    if value != "auto":
        return cast(Literal["direct", "fft"], value)
    return "fft" if count * lag_count > _FFT_CROSSOVER else "direct"


def _observable_series(
    data: Trajectory | ArrayLike,
    *,
    observable: Observable | None,
    name: str,
) -> tuple[FloatArray, int]:
    """Return ``(batch_size, time)`` scalar series plus the ensemble size."""
    raw = data.states if isinstance(data, Trajectory) else as_float_array(data, name=name)
    if isinstance(data, Trajectory) and observable is None:
        if data.state_dim != 1:
            raise ValidationError(
                f"{name} needs a scalar time series, but the trajectory stores "
                f"{data.state_dim} coordinates per state; pass observable=lambda states: "
                "states[..., 0] to choose one, or any other reduction of the trailing axis"
            )
        raw = raw[..., 0]
    transformed = observable(raw) if observable is not None else raw
    series = as_float_array(transformed, name=f"{name} observable")
    if series.shape[-1] < 2:
        raise ValidationError(f"{name} requires at least two time samples")
    if any(extent == 0 for extent in series.shape[:-1]):
        raise ValidationError(f"{name} observable must not have an empty batch axis")
    return series.reshape(-1, series.shape[-1]), int(np.prod(series.shape[:-1], dtype=np.intp))


def _position_series(
    data: Trajectory | ArrayLike,
    *,
    observable: Observable | None,
) -> tuple[FloatArray, int]:
    """Return ``(batch_size, time, dimension)`` positions plus the ensemble size."""
    raw = data.states if isinstance(data, Trajectory) else as_float_array(data, name="positions")
    transformed = observable(raw) if observable is not None else raw
    points = as_float_array(transformed, name="position observable")
    if points.ndim == 1:
        points = points[:, None]
    if points.shape[-2] < 2 or points.shape[-1] == 0:
        raise ValidationError(
            "position observable must produce shape (time,), (time, dimension) or "
            f"(*batch, time, dimension) with at least two times; got {points.shape}"
        )
    if any(extent == 0 for extent in points.shape[:-2]):
        raise ValidationError("position observable must not have an empty batch axis")
    batch_size = int(np.prod(points.shape[:-2], dtype=np.intp))
    return points.reshape(batch_size, *points.shape[-2:]), batch_size


def _unwrap(points: FloatArray, *, periods: ArrayLike | None, dimension: int) -> FloatArray:
    if periods is None:
        raise ValidationError("periods are required when unwrap=True")
    widths = _periods(periods, dimension)
    differences = np.diff(points, axis=-2)
    differences -= widths * np.floor(differences / widths + 0.5)
    _validate_unwrap_resolution(differences, widths=widths)
    result = np.empty_like(points)
    result[..., 0, :] = points[..., 0, :]
    result[..., 1:, :] = points[..., :1, :] + np.cumsum(differences, axis=-2)
    return result


def _validate_unwrap_resolution(differences: FloatArray, *, widths: FloatArray) -> None:
    """Reject sampling too coarse for the minimum-image branch to be unique.

    The minimum-image reconstruction is exact only while every true one-step
    displacement stays strictly inside ``(-period/2, period/2)``; a displacement
    of exactly ``period/2`` is already ambiguous and anything larger is aliased
    back to a wrong, silently smaller value. Wrapped samples carry no record of
    the true displacement, so the only observable symptom is reconstructed
    displacements piling up against the aliasing limit: the guard rejects a
    coordinate whose largest reconstructed displacement -- over every time step of
    every trajectory in the ensemble, because one aliased trajectory poisons the
    ensemble average -- comes within :data:`_ALIASING_MARGIN` of half its period.
    """
    if differences.size == 0:
        return
    half_widths = 0.5 * widths
    largest = np.max(np.abs(differences), axis=tuple(range(differences.ndim - 1)))
    unsafe = np.flatnonzero(largest >= (1.0 - _ALIASING_MARGIN) * half_widths)
    if unsafe.size == 0:
        return
    coordinate = int(unsafe[0])
    raise ValidationError(
        f"unwrap cannot resolve coordinate {coordinate}: the largest one-step displacement "
        f"is {float(largest[coordinate])!r}, which reaches the aliasing limit of half its "
        f"period {float(half_widths[coordinate])!r} (period {float(widths[coordinate])!r}); "
        "minimum-image unwrapping is only unique below that limit. Sample more finely until "
        "every one-step displacement is below half the period, or pass already-unwrapped "
        "coordinates with unwrap=False"
    )


def _periods(value: ArrayLike, dimension: int) -> FloatArray:
    """Return one period per coordinate, broadcasting a scalar to every coordinate."""
    if np.ndim(value) == 0:
        scalar = as_float_array(np.reshape(value, (1,)), name="periods", ndim=1)
        widths = np.full(dimension, float(scalar[0]), dtype=np.float64)
    else:
        widths = as_float_array(value, name="periods")
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


__all__ = [
    "Method",
    "Observable",
    "autocorrelation",
    "local_diffusion_exponent",
    "mean_square_displacement",
]

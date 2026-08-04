"""Long-range spectral statistics and random-matrix reference curves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.integrate import quad  # type: ignore[import-untyped]
from scipy.special import sici  # type: ignore[import-untyped]

from chaos_numerics.core import ExperimentMetadata, ValidationError
from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.types import ArrayLike, FloatArray
from chaos_numerics.spectral.levels import PreparedEigenphases, UnfoldedSpectrum

Window = Literal["none", "hann"]
Ensemble = Literal["poisson", "goe", "gue", "cue"]
Statistic = Literal["spectral_form_factor", "number_variance"]


@dataclass(frozen=True, slots=True, eq=False)
class SpectralCurve:
    """Plot-ready x/value curve with optional estimator uncertainty and variance."""

    x: FloatArray
    values: FloatArray
    uncertainty: FloatArray | None = None
    variance: FloatArray | None = None
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def __post_init__(self) -> None:
        x = _readonly_1d(self.x, name="curve x")
        values = _readonly_1d(self.values, name="curve values")
        if x.shape != values.shape:
            raise ValidationError(
                f"curve x and values must have equal shape; got {x.shape} and {values.shape}"
            )
        uncertainty = _optional_curve(self.uncertainty, values=values, name="uncertainty")
        variance = _optional_curve(self.variance, values=values, name="variance")
        if not isinstance(self.metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "variance", variance)


def spectral_form_factor(
    spectrum: PreparedEigenphases | UnfoldedSpectrum,
    times: ArrayLike,
    *,
    window: Window = "none",
    connected: bool = True,
    bootstrap: int = 0,
    seed: int | None = None,
) -> SpectralCurve:
    """Return ``|sum w_n exp(2*pi*i*tau*x_n)|^2 / sum(w_n^2)``.

    ``tau`` is time divided by the Heisenberg time. The connected estimator
    subtracts the same window evaluated on a uniform unit-spacing spectrum.
    """
    levels, sector = _unfolded_levels(spectrum)
    tau = _nonnegative_1d(times, name="times")
    if window not in {"none", "hann"}:
        raise ValidationError(f"window must be 'none' or 'hann'; got {window!r}")
    if not isinstance(connected, bool):
        raise ValidationError("connected must be a bool")
    resamples = _nonnegative_int(bootstrap, name="bootstrap")
    random_seed = _seed(seed, required=resamples > 0)
    count = levels.size
    if count == 0:
        zeros = np.zeros_like(tau)
        return SpectralCurve(
            tau,
            zeros,
            uncertainty=zeros if resamples else None,
            variance=zeros if resamples else None,
            metadata=_curve_metadata(
                "spectral_form_factor",
                sector=sector,
                parameters={
                    "window": window,
                    "connected": connected,
                    "bootstrap": resamples,
                    "seed": random_seed,
                    "level_count": 0,
                    "time_normalization": "tau=t/t_H",
                },
            ),
        )
    weights = _window_weights(count, window)
    normalization = float(np.sum(weights**2))
    centered = levels - float(np.mean(levels))
    phases = np.exp(2j * np.pi * np.outer(tau, centered))
    amplitude = phases @ weights
    smooth = (
        _smooth_amplitude(tau, count=count, window=window, weights=weights) if connected else 0.0
    )
    values = np.asarray(np.abs(amplitude - smooth) ** 2 / normalization, dtype=np.float64)

    uncertainty: FloatArray | None = None
    variance: FloatArray | None = None
    if resamples:
        rng = np.random.default_rng(random_seed)
        replicates = np.empty((resamples, tau.size), dtype=np.float64)
        for index in range(resamples):
            selection = rng.integers(0, count, size=count)
            sampled_amplitude = phases[:, selection] @ weights
            replicates[index] = np.abs(sampled_amplitude - smooth) ** 2 / normalization
        variance = np.var(replicates, axis=0, ddof=1 if resamples > 1 else 0)
        uncertainty = np.sqrt(variance)
    metadata = _curve_metadata(
        "spectral_form_factor",
        sector=sector,
        parameters={
            "window": window,
            "connected": connected,
            "bootstrap": resamples,
            "seed": random_seed,
            "level_count": int(count),
            "normalization": "sum(w^2); plateau=1",
            "time_normalization": "tau=t/t_H",
            "finite_size_background": "continuous uniform-density subtraction"
            if connected
            else "retained",
        },
    )
    return SpectralCurve(tau, values, uncertainty, variance, metadata)


def number_variance(
    spectrum: PreparedEigenphases | UnfoldedSpectrum,
    lengths: ArrayLike,
    *,
    samples: int = 2048,
    bootstrap: int = 0,
    seed: int | None = None,
    finite_size_correction: bool = False,
) -> SpectralCurve:
    """Return circular-window ``mean((n(origin,L)-L)^2)`` for unfolded levels."""
    levels, sector = _unfolded_levels(spectrum)
    windows = _nonnegative_1d(lengths, name="lengths")
    origin_count = _positive_int(samples, name="samples")
    resamples = _nonnegative_int(bootstrap, name="bootstrap")
    random_seed = _seed(seed, required=resamples > 0)
    if not isinstance(finite_size_correction, bool):
        raise ValidationError("finite_size_correction must be a bool")
    dimension = levels.size
    if dimension == 0:
        if windows.size:
            raise ValidationError("number variance requires a non-empty spectrum")
        return SpectralCurve(windows, windows.copy(), windows.copy(), windows.copy())
    if bool(np.any(windows > dimension)):
        raise ValidationError(f"lengths must not exceed the circular period {dimension}")
    if finite_size_correction and bool(np.any(windows >= dimension)):
        raise ValidationError("finite-size correction requires every length to be less than N")

    canonical = np.sort(np.mod(levels, dimension))
    doubled = np.concatenate((canonical, canonical + dimension))
    origins = np.linspace(0.0, dimension, origin_count, endpoint=False)
    starts = np.searchsorted(canonical, origins, side="left")
    values = np.empty(windows.size, dtype=np.float64)
    uncertainty = np.empty(windows.size, dtype=np.float64)
    estimator_variance = np.empty(windows.size, dtype=np.float64)
    rng = np.random.default_rng(random_seed) if resamples else None
    for index, length in enumerate(windows):
        ends = np.searchsorted(doubled, origins + length, side="left")
        counts = ends - starts
        squared = (counts - length) ** 2
        estimate = float(np.mean(squared))
        correction = 1.0 / (1.0 - length / dimension) if finite_size_correction else 1.0
        values[index] = estimate * correction
        if resamples:
            assert rng is not None
            replicates = np.empty(resamples, dtype=np.float64)
            for sample in range(resamples):
                selection = rng.integers(0, origin_count, size=origin_count)
                replicates[sample] = float(np.mean(squared[selection])) * correction
            estimator_variance[index] = float(np.var(replicates, ddof=1 if resamples > 1 else 0))
        else:
            estimator_variance[index] = (
                float(np.var(squared, ddof=1 if origin_count > 1 else 0) / origin_count)
                * correction**2
            )
        uncertainty[index] = np.sqrt(estimator_variance[index])
    metadata = _curve_metadata(
        "number_variance",
        sector=sector,
        parameters={
            "samples": origin_count,
            "bootstrap": resamples,
            "seed": random_seed,
            "level_count": int(dimension),
            "window_origin": "uniform circular grid",
            "counting_interval": "[origin, origin+L)",
            "finite_size_correction": finite_size_correction,
            "finite_size_factor": "1/(1-L/N)" if finite_size_correction else "none",
        },
    )
    return SpectralCurve(windows, values, uncertainty, estimator_variance, metadata)


def rmt_reference(
    statistic: Statistic,
    ensemble: Ensemble,
    x: ArrayLike,
    *,
    dimension: int | None = None,
) -> SpectralCurve:
    """Return analytic bulk Poisson/GOE/GUE/CUE comparison curves."""
    if statistic not in {"spectral_form_factor", "number_variance"}:
        raise ValidationError(f"unsupported RMT statistic {statistic!r}")
    if ensemble not in {"poisson", "goe", "gue", "cue"}:
        raise ValidationError(f"unsupported RMT ensemble {ensemble!r}")
    points = _nonnegative_1d(x, name="x")
    size = _positive_int(dimension, name="dimension") if dimension is not None else None
    if statistic == "spectral_form_factor":
        values = _rmt_form_factor(ensemble, points)
        finite_size = "CUE min(t,N)/N at tau=t/N" if ensemble == "cue" and size else "bulk"
    else:
        if size is not None and bool(np.any(points > size)):
            raise ValidationError("number-variance reference lengths must not exceed dimension")
        values = _rmt_number_variance(ensemble, points, dimension=size)
        finite_size = "circular kernel" if ensemble == "cue" and size else "bulk"
    zeros = np.zeros_like(values)
    metadata = ExperimentMetadata(
        parameters={
            "statistic": statistic,
            "ensemble": ensemble.upper(),
            "dimension": size,
            "reference_type": "analytic",
            "finite_size": finite_size,
        },
        precision="float64",
    )
    return SpectralCurve(points, values, zeros, zeros, metadata)


def _rmt_form_factor(ensemble: Ensemble, tau: FloatArray) -> FloatArray:
    absolute = np.abs(tau)
    if ensemble == "poisson":
        values = np.ones_like(absolute)
        values[absolute == 0.0] = 0.0
        return values
    if ensemble in {"gue", "cue"}:
        return np.minimum(absolute, 1.0)
    values = np.empty_like(absolute)
    below = absolute <= 1.0
    values[below] = 2.0 * absolute[below] - absolute[below] * np.log1p(2.0 * absolute[below])
    above_values = absolute[~below]
    values[~below] = 2.0 - above_values * np.log(
        (2.0 * above_values + 1.0) / (2.0 * above_values - 1.0)
    )
    return values


def _rmt_number_variance(
    ensemble: Ensemble,
    lengths: FloatArray,
    *,
    dimension: int | None,
) -> FloatArray:
    if ensemble == "poisson":
        if dimension is None:
            return lengths.copy()
        return np.asarray(lengths * (1.0 - lengths / dimension), dtype=np.float64)
    result = np.empty_like(lengths)
    for index, length in enumerate(lengths):
        if length == 0.0:
            result[index] = 0.0
            continue
        integral = quad(
            _weighted_cluster,
            0.0,
            float(length),
            args=(float(length), ensemble, dimension),
            epsabs=1e-11,
            epsrel=1e-11,
        )[0]
        result[index] = max(0.0, float(length) - 2.0 * integral)
    return result


def _weighted_cluster(
    separation: float,
    length: float,
    ensemble: Ensemble,
    dimension: int | None,
) -> float:
    if ensemble == "goe":
        cluster = _goe_cluster(separation)
    elif ensemble == "cue" and dimension is not None:
        cluster = _cue_cluster(separation, dimension)
    else:
        cluster = float(np.sinc(separation) ** 2)
    return (length - separation) * cluster


def _goe_cluster(separation: float) -> float:
    if separation == 0.0:
        return 1.0
    sine = np.sin(np.pi * separation)
    sinc = sine / (np.pi * separation)
    derivative = (np.pi * separation * np.cos(np.pi * separation) - sine) / (np.pi * separation**2)
    tail = 0.5 - float(sici(np.pi * separation)[0]) / np.pi
    return float(sinc**2 + derivative * tail)


def _cue_cluster(separation: float, dimension: int) -> float:
    denominator = dimension * np.sin(np.pi * separation / dimension)
    if abs(denominator) <= np.finfo(np.float64).tiny:
        return 1.0
    return float((np.sin(np.pi * separation) / denominator) ** 2)


def _unfolded_levels(
    spectrum: PreparedEigenphases | UnfoldedSpectrum,
) -> tuple[FloatArray, str | None]:
    if isinstance(spectrum, UnfoldedSpectrum):
        return spectrum.values, spectrum.symmetry_sector
    if isinstance(spectrum, PreparedEigenphases):
        if spectrum.count == 0:
            return np.empty(0, dtype=np.float64), spectrum.symmetry_sector
        mean_spacing = 2.0 * np.pi / spectrum.count
        return np.asarray(
            (spectrum.phases - spectrum.phases[0]) / mean_spacing
        ), spectrum.symmetry_sector
    raise ValidationError("spectrum must be prepared or unfolded eigenphase data")


def _window_weights(count: int, window: Window) -> FloatArray:
    if window == "none":
        return np.ones(count, dtype=np.float64)
    if count < 3:
        raise ValidationError("hann window requires at least three levels")
    return np.asarray(np.hanning(count), dtype=np.float64)


def _smooth_amplitude(
    tau: FloatArray,
    *,
    count: int,
    window: Window,
    weights: FloatArray,
) -> FloatArray:
    scaled_time = count * tau
    if window == "none":
        return np.asarray(count * np.sinc(scaled_time), dtype=np.float64)
    continuous = 0.5 * count * np.sinc(scaled_time) + 0.25 * count * (
        np.sinc(scaled_time + 1.0) + np.sinc(scaled_time - 1.0)
    )
    return np.asarray(continuous * (np.sum(weights) / (0.5 * count)), dtype=np.float64)


def _curve_metadata(
    statistic: str,
    *,
    sector: str | None,
    parameters: dict[str, object],
) -> ExperimentMetadata:
    return ExperimentMetadata(
        parameters={"statistic": statistic, "symmetry_sector": sector, **parameters},
        precision="float64",
    )


def _readonly_1d(value: ArrayLike, *, name: str) -> FloatArray:
    array = as_float_array(value, name=name, ndim=1, copy=True)
    array.setflags(write=False)
    return array


def _optional_curve(value: ArrayLike | None, *, values: FloatArray, name: str) -> FloatArray | None:
    if value is None:
        return None
    array = _readonly_1d(value, name=name)
    if array.shape != values.shape:
        raise ValidationError(f"{name} must have shape {values.shape}; got {array.shape}")
    if np.any(array < 0.0):
        raise ValidationError(f"{name} must be non-negative")
    return array


def _nonnegative_1d(value: ArrayLike, *, name: str) -> FloatArray:
    array = as_float_array(value, name=name, ndim=1, copy=True)
    if np.any(array < 0.0):
        raise ValidationError(f"{name} must be non-negative")
    return array


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


def _seed(value: object, *, required: bool) -> int | None:
    if value is None:
        if required:
            raise ValidationError("seed is required when bootstrap is positive")
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"seed must be an integer or None; got {value!r}")
    return int(value)


__all__ = ["SpectralCurve", "number_variance", "rmt_reference", "spectral_form_factor"]

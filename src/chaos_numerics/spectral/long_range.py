"""Long-range spectral statistics and random-matrix reference curves.

The three ``Literal`` aliases below appear in the public signatures of
:func:`spectral_form_factor` and :func:`rmt_reference` and are therefore exported
from :mod:`chaos_numerics.spectral` so that callers can annotate their own
wrappers. See that package's ``__init__`` for what each one means.
"""

from __future__ import annotations

import warnings as python_warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np
from scipy.special import sici  # type: ignore[import-untyped]

from chaos_numerics.core import (
    Diagnostic,
    ExperimentMetadata,
    NumericalWarning,
    ValidationError,
)
from chaos_numerics.core._payload import (
    ArrayPayload,
    add_convergence_history,
    metadata_payload,
)
from chaos_numerics.core._validation import as_float_array, wrap_into_half_open
from chaos_numerics.core.types import ArrayLike, FloatArray, IndexArray
from chaos_numerics.spectral._ensembles import (
    CanonicalEnsemble,
    Ensemble,
    canonical_ensemble,
    quoted,
)
from chaos_numerics.spectral.levels import PreparedEigenphases, UnfoldedSpectrum

Window = Literal["none", "hann"]
Statistic = Literal[
    "spectral_form_factor",
    "number_variance",
    "spacing_distribution",
    "spectral_rigidity",
    "gap_ratio_distribution",
]

#: Statistics whose reference curve carries no finite-``N`` parameter, with the
#: reason that ``dimension`` is rejected rather than silently ignored.
_DIMENSIONLESS_REFERENCES: dict[str, str] = {
    "spacing_distribution": "the Wigner surmise is a 2x2 result with no finite-N correction",
    "gap_ratio_distribution": "the Atas 3x3 surmise carries no finite-N parameter",
    "spectral_rigidity": (
        "the Mehta relation this reference integrates is a bulk identity, so there is no "
        "finite-N kernel to substitute"
    ),
}

#: Dyson index and reciprocal normalization ``1/Z_beta`` of the Atas et al. (2013)
#: 3x3 gap-ratio surmise
#: ``P_beta(r) = (r+r^2)^beta / (Z_beta * (1+r+r^2)^(1+3*beta/2))``.
#: The closed forms ``Z_1 = 8/27``, ``Z_2 = 4*pi/(81*sqrt(3))`` and
#: ``Z_4 = 4*pi/(729*sqrt(3))`` were each confirmed against adaptive quadrature of
#: the unnormalized density over ``(0, inf)`` to better than 4e-16 relative. A test
#: re-integrates every curve rather than trusting the constants.
_GAP_RATIO_SURMISE: dict[CanonicalEnsemble, tuple[float, float]] = {
    "goe": (1.0, 27.0 / 8.0),
    "gue": (2.0, 81.0 * np.sqrt(3.0) / (4.0 * np.pi)),
    "cue": (2.0, 81.0 * np.sqrt(3.0) / (4.0 * np.pi)),
    "gse": (4.0, 729.0 * np.sqrt(3.0) / (4.0 * np.pi)),
}

#: Dyson index, prefactor and Gaussian rate of the Wigner surmise
#: ``P_beta(s) = a_beta * s^beta * exp(-b_beta * s^2)``. See
#: :func:`_rmt_spacing_distribution` for the two normalization conditions that fix
#: ``a`` and ``b`` jointly. ``a_4 = 2**18 / (3**6 * pi**3)`` and
#: ``b_4 = 64 / (9*pi)``.
_WIGNER_SURMISE: dict[CanonicalEnsemble, tuple[float, float, float]] = {
    "goe": (1.0, 0.5 * np.pi, 0.25 * np.pi),
    "gue": (2.0, 32.0 / np.pi**2, 4.0 / np.pi),
    "cue": (2.0, 32.0 / np.pi**2, 4.0 / np.pi),
    "gse": (4.0, 2.0**18 / (3.0**6 * np.pi**3), 64.0 / (9.0 * np.pi)),
}

#: Gauss-Legendre rule applied to each unit-length panel of the cluster-function
#: integrals behind the ``"number_variance"`` and ``"spectral_rigidity"``
#: references. The integrand is a low-degree polynomial times ``Y_2``, whose
#: oscillation period is one for beta=1,2 and one half for beta=4, so one panel per
#: unit resolves both. 24 nodes are already at rounding: doubling to 48, or
#: quartering the panels, moves ``Delta_3`` by 1e-15 at ``L=5`` and 3e-13 at
#: ``L=3000``, for beta=4 as well as for beta=1,2.
#:
#: A fixed panel rule rather than adaptive quadrature, because ``scipy``'s ``quad``
#: silently loses the oscillatory tail once the interval holds a few hundred
#: periods: at ``L=1024`` it returned ``Sigma^2_GUE = 0.9078`` against the correct
#: ``1.0483`` and ``Sigma^2_GSE = -65.4`` against ``0.6218``, in both cases after
#: an ``IntegrationWarning`` that a caller filtering warnings would never see. On
#: the finite-``N`` CUE kernel at ``N=64`` the two agree to 4e-16.
_MEHTA_NODES, _MEHTA_WEIGHTS = np.polynomial.legendre.leggauss(24)

#: Panels evaluated per array allocation, so that a large ``L`` costs time rather
#: than memory.
_MEHTA_PANEL_CHUNK = 4096

#: What the ``uncertainty``/``variance`` arrays of :func:`spectral_form_factor`
#: actually are. ``K(tau)`` is a coherent sum that does not self-average, and
#: resampling levels with replacement destroys the level correlations that fix
#: its value, so the bootstrap spread is *not* an estimate of the scatter over
#: spectra. Calibration against 150 Haar-CUE spectra at ``N=128``: the bootstrap
#: standard deviation is 3.7x the true realization scatter at ``tau=0.3``, 1.5x
#: at ``tau=1.0`` and 1.6x at ``tau=1.5``.
_SFF_ERROR_SEMANTICS: dict[bool, str] = {
    False: "none; no uncertainty is reported without bootstrap",
    True: (
        "level-resampling bootstrap spread of a single spectrum; NOT realization-to-"
        "realization scatter, which it overstates by 1.5-3.7x on Haar CUE (N=128). "
        "K(tau) does not self-average: average over an ensemble for a real error bar"
    ),
}


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

    def __reduce__(self) -> tuple[type[SpectralCurve], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only.

        NumPy drops ``writeable=False`` when an array is pickled, and
        ``copy.deepcopy`` goes through the same protocol, so without this the
        documented read-only contract survived neither round trip.
        """
        return (
            self.__class__,
            (self.x, self.values, self.uncertainty, self.variance, self.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpectralCurve):
            return NotImplemented
        return (
            np.array_equal(self.x, other.x)
            and np.array_equal(self.values, other.values)
            and _optional_array_equal(self.uncertainty, other.uncertainty)
            and _optional_array_equal(self.variance, other.variance)
            and self.metadata == other.metadata
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable copies of every stored array."""
        payload: ArrayPayload = {"x": self.x.copy(), "values": self.values.copy()}
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty.copy()
        if self.variance is not None:
            payload["variance"] = self.variance.copy()
        add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return the JSON descriptor, with array shapes but no array contents."""
        arrays: ArrayPayload = {"x": self.x, "values": self.values}
        if self.uncertainty is not None:
            arrays["uncertainty"] = self.uncertainty
        if self.variance is not None:
            arrays["variance"] = self.variance
        add_convergence_history(arrays, self.metadata, copy=False)
        return metadata_payload("spectral_curve", self.metadata, arrays)


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

    **``bootstrap`` does not measure the error on ``K(tau)``.** ``K`` is a
    coherent sum over the whole spectrum and does not self-average: its
    realization-to-realization relative fluctuation stays of order one however
    large ``N`` gets, and the only way to reduce it is to average ``K`` over an
    ensemble of spectra. Resampling levels with replacement destroys exactly the
    level-level correlations that set the value of ``K``, so the returned
    ``uncertainty`` is a resampling diagnostic of one spectrum and nothing more.

    Measured against the true scatter of 150 Haar-CUE spectra at ``N=128``, the
    bootstrap standard deviation is **3.7x too large at ``tau=0.3``** (1.222
    against a true 0.326), **1.5x at ``tau=1.0``** (1.658 against 1.110) and
    **1.6x at ``tau=1.5``** (1.658 against 1.053). It errs high, so it will not
    manufacture a spurious detection, but it is wide enough to hide a real
    departure from RMT and must not be quoted as an error bar on the physics.
    ``metadata.parameters["error_semantics"]`` says the same thing in machine-
    readable form. Average over an ensemble and take the scatter of the members
    if you need a real uncertainty on ``K(tau)``.
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
                    "error_semantics": _SFF_ERROR_SEMANTICS[bool(resamples)],
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
            "error_semantics": _SFF_ERROR_SEMANTICS[bool(resamples)],
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
    """Return circular-window ``mean((n(origin,L)-L)^2)`` for unfolded levels.

    Window origins sit on the *half-shifted* uniform circular grid
    ``(k + 0.5) * N / samples``. A grid starting at zero coincides with the
    levels themselves whenever ``samples`` is a multiple of ``N`` (the common
    case, both defaults being powers of two), and ``searchsorted(..., "left")``
    then resolves those ties by excluding the level at the left edge, biasing
    the counts low. On an equally spaced spectrum this turned the exact
    ``Sigma^2(L) = frac(L) * (1 - frac(L))`` into a visibly wrong value at
    integer ``L``; the half shift keeps every origin off the level positions.

    **What ``uncertainty`` means.** Origins are dense on one spectrum, so the
    ``samples`` windows are not independent measurements: only about ``N / L``
    windows of length ``L`` fit side by side on ``N`` levels, and everything
    beyond that is the same data read again. The reported error is therefore a
    *batch-means* standard error over ``effective_samples =
    min(samples, floor(N / L))`` contiguous arcs of the circle, not
    ``sqrt(var / samples)``. ``metadata.parameters`` records both
    ``effective_samples`` (one entry per requested length) and
    ``error_semantics``.

    Dividing by ``samples`` -- what this function used to do -- understated the
    realization-to-realization scatter by a factor of 2 to 4 and, worse, kept
    shrinking as ``samples`` grew on a *fixed* spectrum, so a converged estimate
    could be made to look arbitrarily many sigma away from an RMT reference.
    Calibrated against the scatter of Haar-CUE ensembles, the reported error is
    now 0.70 to 0.87 times the true realization standard deviation (90 members
    at ``N=256``: 0.82 at ``L=2``, 0.87 at ``L=5``, 0.77 at ``L=10``; the old
    estimator gave 0.47, 0.38 and 0.27 on the same data). It is an estimate of
    the scatter of *this* statistic over spectra drawn from the same ensemble,
    and it is mildly conservative on the low side because neighbouring arcs of a
    rigid spectrum are anticorrelated.

    ``bootstrap`` resamples the same arcs, not individual origins. Resampling
    origins treats overlapping windows as independent draws and reproduces the
    ``sqrt(var / samples)`` understatement exactly; the block bootstrap agrees
    with the batch-means error to within a few percent.

    When ``L`` covers more than half the spectrum a single window is all that
    fits, no scatter can be resolved from one spectrum, and the function warns
    with :class:`NumericalWarning` and falls back to the spread of that one
    window -- an honest upper bound, not a standard error.
    """
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
        return SpectralCurve(
            windows,
            windows.copy(),
            windows.copy(),
            windows.copy(),
            _number_variance_metadata(
                sector=sector,
                origin_count=origin_count,
                resamples=resamples,
                random_seed=random_seed,
                dimension=0,
                finite_size_correction=finite_size_correction,
                effective=(),
                diagnostics=[],
            ),
        )
    if bool(np.any(windows > dimension)):
        raise ValidationError(f"lengths must not exceed the circular period {dimension}")
    if finite_size_correction and bool(np.any(windows >= dimension)):
        raise ValidationError("finite-size correction requires every length to be less than N")

    canonical = np.sort(wrap_into_half_open(levels, lower=0.0, upper=float(dimension)))
    doubled = np.concatenate((canonical, canonical + dimension))
    origins = (np.arange(origin_count, dtype=np.float64) + 0.5) * dimension / origin_count
    starts = np.searchsorted(canonical, origins, side="left")
    values = np.empty(windows.size, dtype=np.float64)
    uncertainty = np.empty(windows.size, dtype=np.float64)
    estimator_variance = np.empty(windows.size, dtype=np.float64)
    rng = np.random.default_rng(random_seed) if resamples else None
    effective: list[int] = []
    unresolved: list[float] = []
    for index, length in enumerate(windows):
        ends = np.searchsorted(doubled, origins + length, side="left")
        counts = ends - starts
        squared = (counts - length) ** 2
        estimate = float(np.mean(squared))
        correction = 1.0 / (1.0 - length / dimension) if finite_size_correction else 1.0
        values[index] = estimate * correction
        blocks = _independent_windows(
            dimension=dimension, length=float(length), origin_count=origin_count
        )
        effective.append(blocks)
        if blocks < 2:
            unresolved.append(float(length))
        estimator_variance[index] = _batch_means_variance(
            squared,
            blocks=blocks,
            origin_count=origin_count,
            resamples=resamples,
            rng=rng,
            correction=correction,
        )
        uncertainty[index] = np.sqrt(estimator_variance[index])
    diagnostics: list[Diagnostic] = []
    if unresolved:
        message = (
            f"{len(unresolved)} window length(s) up to {max(unresolved)} admit fewer than two "
            f"independent windows on {dimension} levels; the reported uncertainty is the "
            "spread of a single window, not an estimate of realization scatter"
        )
        diagnostics.append(
            Diagnostic("number-variance-unresolved-error", message, category="spectral")
        )
        python_warnings.warn(message, NumericalWarning, stacklevel=2)
    metadata = _number_variance_metadata(
        sector=sector,
        origin_count=origin_count,
        resamples=resamples,
        random_seed=random_seed,
        dimension=dimension,
        finite_size_correction=finite_size_correction,
        effective=tuple(effective),
        diagnostics=diagnostics,
    )
    return SpectralCurve(windows, values, uncertainty, estimator_variance, metadata)


def _batch_means_variance(
    per_window: FloatArray,
    *,
    blocks: int,
    origin_count: int,
    resamples: int,
    rng: np.random.Generator | None,
    correction: float = 1.0,
) -> float:
    """Return the estimator variance of ``mean(per_window)`` over independent arcs.

    Shared by :func:`number_variance` and :func:`spectral_rigidity` so that the two
    curves cannot drift into reporting different things under the same name. When
    fewer than two arcs fit, one window of this length is all the spectrum holds,
    the scatter of the estimator cannot be resolved from a single realization at
    all, and the spread of that one window is returned instead -- an upper bound
    rather than an understatement.
    """
    if blocks < 2:
        spread = float(np.var(per_window, ddof=1 if origin_count > 1 else 0))
        return spread * correction**2
    block_means = np.asarray(
        [float(np.mean(part)) for part in np.array_split(per_window, blocks)],
        dtype=np.float64,
    )
    if resamples:
        assert rng is not None
        replicates = np.empty(resamples, dtype=np.float64)
        for sample in range(resamples):
            selection = rng.integers(0, blocks, size=blocks)
            replicates[sample] = float(np.mean(block_means[selection])) * correction
        return float(np.var(replicates, ddof=1 if resamples > 1 else 0))
    return float(np.var(block_means, ddof=1) / blocks) * correction**2


def spectral_rigidity(
    spectrum: PreparedEigenphases | UnfoldedSpectrum,
    lengths: ArrayLike,
    *,
    samples: int = 2048,
    bootstrap: int = 0,
    seed: int | None = None,
) -> SpectralCurve:
    """Return the Dyson-Mehta spectral rigidity ``Delta_3(L)`` for unfolded levels.

    ``Delta_3(L)`` is the mean-square residual of the best straight-line fit to the
    staircase counting function inside a window of length ``L``::

        Delta_3(L) = < min_{A,B} (1/L) * int_{E0}^{E0+L} (N(E) - A*E - B)^2 dE >_{E0}

    averaged over window origins. Conventions are deliberately identical to
    :func:`number_variance`: levels come from a ``PreparedEigenphases`` or
    ``UnfoldedSpectrum``, windows are circular with period ``N``, origins sit on
    the half-shifted grid ``(k + 0.5) * N / samples`` so that no origin can land on
    a level, and the error bar is a batch-means standard error over
    ``effective_samples = min(samples, floor(N / L))`` independent arcs.

    **The window integral is evaluated in closed form, not by quadrature.**
    ``N(E)`` is a staircase, so the least-squares residual is a finite expression in
    the ``n`` in-window level positions ``x_k`` measured from the origin. Writing
    ``s_k = x_k / L``::

        Delta_3 = n^2 - sum_k (2k-1) s_k - (n - sum_k s_k)^2 - 3 * (sum_k s_k(1-s_k))^2

    which is what the implementation computes from prefix sums over the doubled
    level array. Checked against an independent implementation that integrates each
    staircase segment exactly and solves the 2x2 normal equations directly: the
    per-window algebra agrees to 1.5e-15 absolute, and this vectorized function
    agrees to 5.0e-13 (worst case over ``L`` from 0.7 to 11 on a 24-level
    spectrum), the difference being the cancellation inside the prefix sums. A
    two-million-point Riemann sum of the same residual agrees to 2.5e-7, which is
    the Riemann sum's own error rather than a disagreement.

    An equally spaced spectrum has a closed-form answer that pins the whole
    pipeline: for integer ``L``, ``Delta_3(L) = 1/12 - 1/(60*L^2)`` exactly, so
    ``Delta_3(1) = 1/15`` and ``Delta_3(L) -> 1/12``. The half-shifted origin grid
    is a midpoint rule for the average over the fractional offset, so the measured
    value approaches that closed form as ``samples/N`` grows, with an error of about
    ``1/(12*(samples/N)^2*L^2)``: at ``N=64`` this is 2.0e-5 at ``samples=4096``,
    8.0e-8 at ``samples=65536`` and 5.0e-9 at ``samples=262144`` for ``L=1``.

    **What ``uncertainty`` means.** Exactly what it means for
    :func:`number_variance`, and the calibration was redone for this statistic
    rather than assumed: against the true scatter of 80 Haar-CUE spectra at
    ``N=128`` (``samples=1024``) the reported error is 1.00 (``L=2``), 0.85
    (``L=5``) and 0.79 (``L=10``) times the realization standard deviation, and
    over four independent ensembles of 80 the ratios span 0.74 to 1.00. Those are
    inside the 0.5-2 band this library requires of an error bar, and where they err
    they err low, for the same reason as the number variance: adjacent arcs of a
    rigid spectrum are anticorrelated, which batch means ignores. ``bootstrap``
    resamples the arcs, not the origins.

    When ``L`` exceeds ``N/2`` a single window is all that fits, no scatter can be
    resolved from one spectrum, and the function warns with
    :class:`NumericalWarning` and falls back to the spread of that one window.
    """
    levels, sector = _unfolded_levels(spectrum)
    windows = _nonnegative_1d(lengths, name="lengths")
    origin_count = _positive_int(samples, name="samples")
    resamples = _nonnegative_int(bootstrap, name="bootstrap")
    random_seed = _seed(seed, required=resamples > 0)
    dimension = levels.size
    if dimension == 0:
        if windows.size:
            raise ValidationError("spectral rigidity requires a non-empty spectrum")
        return SpectralCurve(
            windows,
            windows.copy(),
            windows.copy(),
            windows.copy(),
            _rigidity_metadata(
                sector=sector,
                origin_count=origin_count,
                resamples=resamples,
                random_seed=random_seed,
                dimension=0,
                effective=(),
                diagnostics=[],
            ),
        )
    if bool(np.any(windows > dimension)):
        raise ValidationError(f"lengths must not exceed the circular period {dimension}")

    canonical = np.sort(wrap_into_half_open(levels, lower=0.0, upper=float(dimension)))
    doubled = np.concatenate((canonical, canonical + dimension))
    position = np.arange(doubled.size, dtype=np.float64)
    prefix_level = np.concatenate(([0.0], np.cumsum(doubled)))
    prefix_square = np.concatenate(([0.0], np.cumsum(doubled**2)))
    prefix_moment = np.concatenate(([0.0], np.cumsum(position * doubled)))
    origins = (np.arange(origin_count, dtype=np.float64) + 0.5) * dimension / origin_count
    starts = np.searchsorted(canonical, origins, side="left")
    values = np.empty(windows.size, dtype=np.float64)
    uncertainty = np.empty(windows.size, dtype=np.float64)
    estimator_variance = np.empty(windows.size, dtype=np.float64)
    rng = np.random.default_rng(random_seed) if resamples else None
    effective: list[int] = []
    unresolved: list[float] = []
    for index, length in enumerate(windows):
        per_window = _rigidity_per_window(
            length=float(length),
            origins=origins,
            starts=starts,
            doubled=doubled,
            prefix_level=prefix_level,
            prefix_square=prefix_square,
            prefix_moment=prefix_moment,
        )
        values[index] = float(np.mean(per_window))
        blocks = _independent_windows(
            dimension=dimension, length=float(length), origin_count=origin_count
        )
        effective.append(blocks)
        if blocks < 2:
            unresolved.append(float(length))
        estimator_variance[index] = _batch_means_variance(
            per_window,
            blocks=blocks,
            origin_count=origin_count,
            resamples=resamples,
            rng=rng,
        )
        uncertainty[index] = np.sqrt(estimator_variance[index])
    diagnostics: list[Diagnostic] = []
    if unresolved:
        message = (
            f"{len(unresolved)} window length(s) up to {max(unresolved)} admit fewer than two "
            f"independent windows on {dimension} levels; the reported uncertainty is the "
            "spread of a single window, not an estimate of realization scatter"
        )
        diagnostics.append(
            Diagnostic("spectral-rigidity-unresolved-error", message, category="spectral")
        )
        python_warnings.warn(message, NumericalWarning, stacklevel=2)
    metadata = _rigidity_metadata(
        sector=sector,
        origin_count=origin_count,
        resamples=resamples,
        random_seed=random_seed,
        dimension=dimension,
        effective=tuple(effective),
        diagnostics=diagnostics,
    )
    return SpectralCurve(windows, values, uncertainty, estimator_variance, metadata)


def _rigidity_per_window(
    *,
    length: float,
    origins: FloatArray,
    starts: IndexArray,
    doubled: FloatArray,
    prefix_level: FloatArray,
    prefix_square: FloatArray,
    prefix_moment: FloatArray,
) -> FloatArray:
    """Return the closed-form ``Delta_3`` of every window of one length.

    ``Delta_3`` is invariant under adding any straight line to the staircase, so
    the absolute counting offset never enters and only the in-window level
    positions matter. With ``x_k`` the ``k``-th of the ``n`` in-window levels
    measured from the origin and ``s_k = x_k / L``, the orthogonal projection of
    the staircase onto ``span{1, x}`` gives the expression in the docstring of
    :func:`spectral_rigidity`; the three sums it needs are differences of prefix
    sums of ``x``, ``x^2`` and ``k*x`` over the doubled level array.
    """
    if length == 0.0:
        return np.zeros_like(origins)
    ends = np.searchsorted(doubled, origins + length, side="left")
    count = (ends - starts).astype(np.float64)
    level_sum = prefix_level[ends] - prefix_level[starts]
    square_sum = prefix_square[ends] - prefix_square[starts]
    moment_sum = prefix_moment[ends] - prefix_moment[starts]
    first = level_sum - count * origins
    second = square_sum - 2.0 * origins * level_sum + count * origins**2
    # sum_k k*x_k with k the one-based in-window rank of each level.
    ranked = moment_sum + (1.0 - starts) * level_sum - origins * count * (count + 1.0) / 2.0
    weighted = 2.0 * ranked - first
    scaled_first = first / length
    scaled_second = second / length**2
    return np.asarray(
        count**2
        - weighted / length
        - (count - scaled_first) ** 2
        - 3.0 * (scaled_first - scaled_second) ** 2,
        dtype=np.float64,
    )


def _rigidity_metadata(
    *,
    sector: str | None,
    origin_count: int,
    resamples: int,
    random_seed: int | None,
    dimension: int,
    effective: tuple[int, ...],
    diagnostics: list[Diagnostic],
) -> ExperimentMetadata:
    semantics = (
        "block-bootstrap standard deviation over effective_samples independent circular arcs"
        if resamples
        else "batch-means standard error over effective_samples independent circular arcs"
    )
    return _curve_metadata(
        "spectral_rigidity",
        sector=sector,
        parameters={
            "samples": origin_count,
            "bootstrap": resamples,
            "seed": random_seed,
            "level_count": int(dimension),
            "window_origin": "half-shifted uniform circular grid: (k+0.5)*N/samples",
            "counting_interval": "[origin, origin+L)",
            "window_fit": "least-squares straight line, evaluated in closed form",
            "effective_samples": effective,
            "effective_samples_rule": "min(samples, floor(N/L)); samples at L=0",
            "error_semantics": (
                f"{semantics}; approximates realization-to-realization scatter "
                "(0.74-1.00x the true CUE ensemble standard deviation), NOT sqrt(var/samples)"
            ),
        },
        warnings=tuple(diagnostics),
    )


def _independent_windows(*, dimension: int, length: float, origin_count: int) -> int:
    """Return how many windows of length ``length`` fit side by side on the circle.

    ``L = 0`` counts nothing and has exactly zero variance, so every origin is
    trivially independent there and the origin count is returned unchanged.
    """
    if length <= 0.0:
        return origin_count
    return min(origin_count, max(1, int(dimension // length)))


def _number_variance_metadata(
    *,
    sector: str | None,
    origin_count: int,
    resamples: int,
    random_seed: int | None,
    dimension: int,
    finite_size_correction: bool,
    effective: tuple[int, ...],
    diagnostics: list[Diagnostic],
) -> ExperimentMetadata:
    semantics = (
        "block-bootstrap standard deviation over effective_samples independent circular arcs"
        if resamples
        else "batch-means standard error over effective_samples independent circular arcs"
    )
    return _curve_metadata(
        "number_variance",
        sector=sector,
        parameters={
            "samples": origin_count,
            "bootstrap": resamples,
            "seed": random_seed,
            "level_count": int(dimension),
            "window_origin": "half-shifted uniform circular grid: (k+0.5)*N/samples",
            "counting_interval": "[origin, origin+L)",
            "finite_size_correction": finite_size_correction,
            "finite_size_factor": "1/(1-L/N)" if finite_size_correction else "none",
            "effective_samples": effective,
            "effective_samples_rule": "min(samples, floor(N/L)); samples at L=0",
            "error_semantics": (
                f"{semantics}; approximates realization-to-realization scatter "
                "(0.70-0.87x the true CUE ensemble standard deviation), NOT sqrt(var/samples)"
            ),
        },
        warnings=tuple(diagnostics),
    )


def rmt_reference(
    statistic: Statistic,
    ensemble: Ensemble,
    x: ArrayLike,
    *,
    dimension: int | None = None,
    folded: bool | None = None,
) -> SpectralCurve:
    """Return analytic bulk Poisson/GOE/COE/GUE/CUE/GSE/CSE comparison curves.

    ``"coe"`` is an accepted alias of ``"goe"`` and ``"cse"`` of ``"gse"``: bulk
    spectral statistics only depend on the Dyson index, so COE and GOE (beta=1)
    share every curve here, as do CSE and GSE (beta=4), and CUE and GUE (beta=2)
    apart from the optional finite-``N`` CUE kernel.

    **No model in this library produces GSE statistics.** beta=4 needs an
    antiunitary symmetry that squares to ``-1`` -- half-integer spin with time
    reversal -- and every map that ships here (cat, baker, kicked rotor) is
    spinless, so it lands in beta=1 or beta=2. The GSE curves are here because
    :func:`rmt_reference` compares against a spectrum the *caller* supplies, and
    that is independent of what the library itself can simulate; the beta=4 tests
    generate Haar circular-symplectic matrices of their own. If you are comparing
    a library model against ``"gse"``, you have almost certainly picked the wrong
    ensemble.

    A GSE spectrum is Kramers degenerate, and every curve here is written for the
    *distinct* levels: feed it one eigenvalue per doublet. Leaving the doublets in
    place does not merely add noise, it replaces the statistic -- half the spacings
    become zero, so the measured ``<r>`` collapses from 0.6744 to 1.3e-14.

    ``x`` is the statistic's own abscissa and must be non-negative: ``tau=t/t_H``
    for ``"spectral_form_factor"``, a window length ``L`` for
    ``"number_variance"`` and ``"spectral_rigidity"``, a spacing ``s`` in units of
    the mean spacing for ``"spacing_distribution"``, and a gap ratio ``r`` for
    ``"gap_ratio_distribution"``.

    ``statistic="spacing_distribution"`` returns the **Wigner surmise**, not the
    exact large-``N`` nearest-neighbour density: it is the closed-form spacing
    density of a 2x2 matrix, extended to the bulk by the usual abuse. It is
    normalized (``integral P ds = 1``, ``<s> = 1``) and is what every textbook
    plots, but it deviates from the exact Gaudin-Mehta distribution by up to
    about 1-2% of the peak, so it is not a tool for measuring small departures
    from Wigner-Dyson statistics.

    ``statistic="spectral_rigidity"`` returns the Dyson-Mehta ``Delta_3(L)``,
    obtained from the exact Mehta relation to the number variance

    ``Delta_3(L) = (2/L^4) * int_0^L (L^3 - 2 L^2 r + r^3) * Sigma^2(r) dr``

    and therefore built from the *same* two-level cluster function that
    ``"number_variance"`` uses. See :func:`_rmt_spectral_rigidity` for the
    reduction that turns the nested integral into one quadrature over ``Y_2``,
    for the exact Poisson result, and for the numerically determined large-``L``
    asymptotics.

    ``statistic="gap_ratio_distribution"`` returns the density of the adjacent gap
    ratio. By default (``folded=True``) it is the density of
    ``r_tilde = min(r, 1/r)`` on ``[0, 1]``, which is exactly what
    :func:`~chaos_numerics.spectral.adjacent_gap_ratios` measures and what
    :func:`~chaos_numerics.spectral.mean_gap_ratio_reference` publishes the mean of;
    it is zero above ``r = 1``. ``folded=False`` returns the unfolded density on
    ``(0, inf)``, which is half of the folded one below ``r = 1`` and mirrors it
    through ``P(1/r)/r^2 = P(r)`` above. See
    :func:`_rmt_gap_ratio_distribution` for the surmise, the normalization, and why
    Poisson is *not* the ``beta -> 0`` limit of it.

    ``dimension`` is rejected -- not ignored -- for every statistic that has no
    finite-``N`` parameter, and for ``"gse"``/``"cse"`` on every statistic, since
    the only finite-``N`` kernel implemented here is the circular unitary one and
    there is no symplectic counterpart to substitute. ``folded`` is rejected for
    every statistic other than ``"gap_ratio_distribution"``. Silently accepting
    either would suggest a correction or a convention had been applied when none
    was.
    """
    if statistic not in set(get_args(Statistic)):
        raise ValidationError(
            f"unsupported RMT statistic {statistic!r}; "
            f"expected one of {quoted(get_args(Statistic))}"
        )
    canonical = canonical_ensemble(ensemble)
    if dimension is not None and statistic in _DIMENSIONLESS_REFERENCES:
        raise ValidationError(
            f"dimension does not apply to the {statistic} reference; "
            f"{_DIMENSIONLESS_REFERENCES[statistic]}"
        )
    if dimension is not None and canonical == "gse":
        raise ValidationError(
            "dimension does not apply to the gse reference; the only finite-N kernel "
            "implemented here is the circular unitary one and there is no symplectic "
            "counterpart, so a dimension would be accepted and ignored"
        )
    if folded is not None and statistic != "gap_ratio_distribution":
        raise ValidationError(
            f"folded does not apply to the {statistic} reference; it selects between the "
            "min(r,1/r) and the r-in-(0,inf) convention of gap_ratio_distribution"
        )
    if folded is not None and not isinstance(folded, bool):
        raise ValidationError(f"folded must be a bool or None; got {folded!r}")
    points = _nonnegative_1d(x, name="x")
    size = _positive_int(dimension, name="dimension") if dimension is not None else None
    fold = True if folded is None else folded
    if statistic == "spectral_form_factor":
        values = _rmt_form_factor(canonical, points)
        finite_size = "CUE min(t,N)/N at tau=t/N" if canonical == "cue" and size else "bulk"
        reference_type = "analytic"
    elif statistic == "number_variance":
        if size is not None and bool(np.any(points > size)):
            raise ValidationError("number-variance reference lengths must not exceed dimension")
        values = _rmt_number_variance(canonical, points, dimension=size)
        finite_size = "circular kernel" if canonical == "cue" and size else "bulk"
        reference_type = "analytic"
    elif statistic == "spectral_rigidity":
        values = _rmt_spectral_rigidity(canonical, points)
        finite_size = "not applicable"
        reference_type = "mehta_integral_of_number_variance"
    elif statistic == "gap_ratio_distribution":
        values = _rmt_gap_ratio_distribution(canonical, points, folded=fold)
        finite_size = "not applicable"
        reference_type = (
            "exact_poisson_ratio_density" if canonical == "poisson" else "atas_surmise_3x3"
        )
    else:
        values = _rmt_spacing_distribution(canonical, points)
        finite_size = "not applicable"
        reference_type = "wigner_surmise_2x2"
    zeros = np.zeros_like(values)
    metadata = ExperimentMetadata(
        parameters={
            "statistic": statistic,
            "ensemble": canonical.upper(),
            "requested_ensemble": ensemble.upper(),
            "ensemble_alias_applied": canonical != ensemble,
            "dimension": size,
            "reference_type": reference_type,
            "finite_size": finite_size,
            "folded": fold if statistic == "gap_ratio_distribution" else None,
        },
        precision="float64",
    )
    return SpectralCurve(points, values, zeros, zeros, metadata)


def _rmt_form_factor(ensemble: CanonicalEnsemble, tau: FloatArray) -> FloatArray:
    """Connected form factors ``K(tau) = 1 - b_2(tau)``, ``tau = t / t_H``.

    Each was checked here against the Fourier transform of the ``Y_2`` that
    :func:`_cluster_function` implements, ``K(tau) = 1 - 2 int_0^inf Y_2(r)
    cos(2 pi tau r) dr`` by oscillatory quadrature, and agreed to eight digits at
    every ``tau`` tried::

        beta=1  2*tau - tau*ln(1+2*tau)            (tau<=1); 2 - tau*ln((2tau+1)/(2tau-1))
        beta=2  min(tau, 1)
        beta=4  tau/2 - (tau/4)*ln|1-tau|          (tau<=2); 1 above

    The beta=4 curve is the odd one out in three ways: it reaches the plateau at
    ``tau=2`` rather than ``tau=1``, it *overshoots* the plateau on the way
    (``K(1.1) = 1.183``), and it has a genuine logarithmic divergence at
    ``tau=1`` -- the Kramers-degeneracy peak at the Heisenberg time of the
    doublet-resolved spectrum. ``tau=1`` is therefore rejected rather than
    returned as an infinity that the container would refuse anyway.
    """
    absolute = np.abs(tau)
    if ensemble == "poisson":
        values = np.ones_like(absolute)
        values[absolute == 0.0] = 0.0
        return values
    if ensemble in {"gue", "cue"}:
        return np.minimum(absolute, 1.0)
    if ensemble == "gse":
        if bool(np.any(absolute == 1.0)):
            raise ValidationError(
                "the GSE form factor diverges logarithmically at tau=1, where the "
                "Kramers doublets pile up at the Heisenberg time; evaluate the "
                "reference on a grid that steps over tau=1"
            )
        values = np.ones_like(absolute)
        inside = absolute < 2.0
        below = absolute[inside]
        values[inside] = 0.5 * below - 0.25 * below * np.log(np.abs(1.0 - below))
        return values
    values = np.empty_like(absolute)
    below = absolute <= 1.0
    values[below] = 2.0 * absolute[below] - absolute[below] * np.log1p(2.0 * absolute[below])
    above_values = absolute[~below]
    values[~below] = 2.0 - above_values * np.log(
        (2.0 * above_values + 1.0) / (2.0 * above_values - 1.0)
    )
    return values


def _rmt_spacing_distribution(ensemble: CanonicalEnsemble, spacings: FloatArray) -> FloatArray:
    """Wigner surmise densities, each normalized to unit mass and unit mean.

    ``P_Poisson(s) = exp(-s)`` and ``P_beta(s) = a_beta * s^beta *
    exp(-b_beta*s^2)`` for the Wigner-Dyson classes, with the pair
    ``(a_beta, b_beta)`` in :data:`_WIGNER_SURMISE`. Unit mean forces
    ``sqrt(b) = Gamma((beta+2)/2) / Gamma((beta+1)/2)`` and unit mass then forces
    ``a = 2*b^((beta+1)/2) / Gamma((beta+1)/2)``, so the rate and the prefactor are
    not independent: changing one without the other breaks both normalizations at
    once. The closed forms in the table were confirmed against those two gamma
    expressions to 1e-15 relative.
    """
    if ensemble == "poisson":
        return np.asarray(np.exp(-spacings), dtype=np.float64)
    index, prefactor, rate = _WIGNER_SURMISE[ensemble]
    return np.asarray(prefactor * spacings**index * np.exp(-rate * spacings**2), dtype=np.float64)


def _rmt_number_variance(
    ensemble: CanonicalEnsemble,
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
        integral = _cluster_integral(
            ensemble,
            length=float(length),
            dimension=dimension,
            weight=_number_variance_kernel,
        )
        result[index] = max(0.0, float(length) - 2.0 * integral)
    return result


def _rmt_spectral_rigidity(ensemble: CanonicalEnsemble, lengths: FloatArray) -> FloatArray:
    """Dyson-Mehta ``Delta_3(L)`` from the exact Mehta relation to ``Sigma^2``.

    Substituting ``Sigma^2(r) = r - 2*int_0^r (r-s) Y_2(s) ds`` into

    ``Delta_3(L) = (2/L^4) int_0^L (L^3 - 2 L^2 r + r^3) Sigma^2(r) dr``

    and doing the ``r`` integral analytically collapses the nested integral onto a
    single quadrature over the *same* two-level cluster function that
    :func:`_rmt_number_variance` uses::

        Delta_3(L) = L/15 - (4/L^4) * int_0^L Y_2(s) * G(s, L) ds
        G(s, L) = int_s^L (L^3 - 2 L^2 r + r^3)(r - s) dr
                = L^5/30 - L^4 s/4 + L^3 s^2/2 - L^2 s^3/3 + s^5/20

    ``G(L, L) = 0`` and ``G(0, L) = L^5/30``, and ``int_0^L G(s, L) ds = 0``
    identically, which is why ``Delta_3 -> L/15`` as ``L -> 0`` for every ensemble.
    Poisson has ``Y_2 = 0``, so ``Delta_3(L) = L/15`` **exactly** -- no quadrature
    is performed for it, and it is the sharpest available check on the kernel above.
    Against a direct nested integration that calls the public ``"number_variance"``
    reference under the Mehta weight, this reduction agrees to 1.8e-15 relative
    (worst case) for GOE and GUE at ``L`` from 0.5 to 20.

    Large-``L`` asymptotics were determined here rather than quoted. A least-squares
    fit of the curve against ``ln L`` over 25 logarithmically spaced ``L`` in
    ``[512, 8192]`` gives slopes ``0.10126081`` (beta=1), ``0.05066059`` (beta=2) and
    ``0.02533787`` (beta=4), against ``1/(beta*pi^2)`` of ``0.10132118``,
    ``0.05066059`` and ``0.02533030`` -- ratios 0.99940, 1.0000000 and 1.00030. The
    residual beta=1 and beta=4 offsets are ``1/L`` corrections, not wrong slopes:
    adding a ``c/L`` term to the same fit brings the ratios to 0.9999988 and
    0.9999997. **The series is ``1/(beta*pi^2)``, one over beta and not one over
    two beta**: ``1/pi^2``, ``1/(2*pi^2)``, ``1/(4*pi^2)``. Pinning the slope to the
    analytic value and reading off the constant gives

    ``Delta_3_GOE(L) -> (1/pi^2) * ln L - 0.006950``
    ``Delta_3_GUE(L) -> (1/(2 pi^2)) * ln L + 0.0590243``
    ``Delta_3_GSE(L) -> (1/(4 pi^2)) * ln L + 0.0783197``

    where the GUE constant is stable to eight digits from ``L=512`` upwards, and the
    GOE and GSE ones are ``1/L`` Richardson extrapolations of the values at
    ``L=4096`` (-0.0069267, 0.0783167) and ``L=8192`` (-0.0069391, 0.0783182). The
    ratio ``Delta_3_GOE/Delta_3_GUE`` approaches 2 only logarithmically -- 1.342 at
    ``L=10``, 1.576 at ``L=100``, 1.695 at ``L=1000``, 1.758 at ``L=8192`` -- so
    the ratios of the slopes, which are exactly 2 and 4, are the statements worth
    testing.

    The beta=4 curve *crosses* the other two rather than simply lying below them:
    at ``L=0.5`` it is the largest of the three (0.033197 against 0.032822 for
    beta=2 and 0.032422 for beta=1), because every ensemble starts from the common
    ``Delta_3 -> L/15``, and the ordering only settles by ``L=2``.
    """
    result = np.empty_like(lengths)
    for index, length in enumerate(lengths):
        result[index] = _rigidity_at(ensemble, float(length))
    return result


def _rigidity_at(ensemble: CanonicalEnsemble, length: float) -> float:
    if length == 0.0:
        return 0.0
    if ensemble == "poisson":
        return length / 15.0
    total = _cluster_integral(ensemble, length=length, dimension=None, weight=_mehta_kernel)
    return max(0.0, length / 15.0 - 4.0 * total / length**4)


def _cluster_integral(
    ensemble: CanonicalEnsemble,
    *,
    length: float,
    dimension: int | None,
    weight: Callable[[FloatArray, float], FloatArray],
) -> float:
    """Integrate ``Y_2(s) * weight(s, L)`` over ``[0, L]`` on unit-length panels.

    Shared by the ``"number_variance"`` and ``"spectral_rigidity"`` references so
    that the two curves are the same quadrature of the same ``Y_2`` and can be
    cross-checked against each other through the Mehta relation. Panels are
    processed ``_MEHTA_PANEL_CHUNK`` at a time so that a large ``L`` costs time
    rather than memory.
    """
    panels = int(np.ceil(length))
    total = 0.0
    for start in range(0, panels, _MEHTA_PANEL_CHUNK):
        lower = np.arange(start, min(start + _MEHTA_PANEL_CHUNK, panels), dtype=np.float64)
        upper = np.minimum(lower + 1.0, length)
        middle = 0.5 * (lower + upper)
        half = 0.5 * (upper - lower)
        nodes = middle[:, None] + half[:, None] * _MEHTA_NODES[None, :]
        cluster = _cluster_function(np.ravel(nodes), ensemble, dimension).reshape(nodes.shape)
        total += float(
            np.sum(half[:, None] * _MEHTA_WEIGHTS[None, :] * cluster * weight(nodes, length))
        )
    return total


def _number_variance_kernel(separations: FloatArray, length: float) -> FloatArray:
    """``L - s``, the weight of ``Sigma^2(L) = L - 2 int_0^L (L - s) Y_2(s) ds``."""
    return np.asarray(length - separations, dtype=np.float64)


def _mehta_kernel(separations: FloatArray, length: float) -> FloatArray:
    """``G(s, L) = int_s^L (L^3 - 2 L^2 r + r^3)(r - s) dr``, the reduced Mehta weight."""
    return np.asarray(
        length**5 / 30.0
        - length**4 * separations / 4.0
        + length**3 * separations**2 / 2.0
        - length**2 * separations**3 / 3.0
        + separations**5 / 20.0,
        dtype=np.float64,
    )


def _rmt_gap_ratio_distribution(
    ensemble: CanonicalEnsemble, ratios: FloatArray, *, folded: bool
) -> FloatArray:
    """Adjacent-gap-ratio densities: the Atas 3x3 surmise, plus exact Poisson.

    The surmise of Atas, Bogomolny, Giraud and Roux, Phys. Rev. Lett. 110, 084101
    (2013) is::

        P_beta(r) = (r + r^2)^beta / (Z_beta * (1 + r + r^2)^(1 + 3*beta/2))

    with ``beta = 1, 2, 4`` for GOE/GUE/GSE. The normalizations ``Z_1 = 8/27``,
    ``Z_2 = 4*pi/(81*sqrt(3))`` and ``Z_4 = 4*pi/(729*sqrt(3))`` were each confirmed
    against adaptive quadrature of the unnormalized density over ``(0, inf)``.

    **Poisson is not the ``beta -> 0`` limit of that formula.** Setting ``beta=0``
    leaves ``1/(Z_0 (1 + r + r^2))`` with ``Z_0 = 2*pi/(3*sqrt(3)) = 1.2092``, whose
    folded mean is 0.40855 -- not the exact Poisson ``2*ln(2) - 1 = 0.38629``. The
    ratio density of independent exponential gaps is exactly
    ``P(r) = 1/(1+r)^2``, and that is what this function returns for
    ``ensemble="poisson"``.

    ``folded=True`` (the default) returns the density of
    ``r_tilde = min(r, 1/r)`` on ``[0, 1]``, which is what
    :func:`~chaos_numerics.spectral.adjacent_gap_ratios` measures. Every density
    above satisfies ``P(1/r)/r^2 = P(r)``, so the two branches of the fold
    coincide and the folded density is simply ``2*P(r)`` for ``r <= 1`` and zero
    above. That identity is what makes the folded curve integrate to one, and
    makes ``int_0^1 r*P_folded(r) dr`` equal the surmise column of
    :func:`~chaos_numerics.spectral.mean_gap_ratio_reference`: 0.3862944 (Poisson,
    exact), 0.5358984 against the published 0.5359 (GOE), 0.6026578 against 0.6027
    (GUE), and 0.6761683 against 0.6762 (GSE).
    """
    if not folded:
        return _gap_ratio_density(ensemble, ratios)
    density = np.zeros_like(ratios)
    inside = ratios <= 1.0
    density[inside] = 2.0 * _gap_ratio_density(ensemble, ratios[inside])
    return density


def _gap_ratio_density(ensemble: CanonicalEnsemble, ratios: FloatArray) -> FloatArray:
    """Unfolded density on ``(0, inf)``, written so that large ``r`` cannot overflow."""
    if ensemble == "poisson":
        return np.asarray(1.0 / (1.0 + ratios) ** 2, dtype=np.float64)
    index, inverse_normalization = _GAP_RATIO_SURMISE[ensemble]
    product = ratios + ratios**2
    total = 1.0 + product
    # (r+r^2)^beta / (1+r+r^2)^(1+3*beta/2) regrouped as ((r+r^2)/(1+r+r^2)^1.5)^beta
    # / (1+r+r^2), whose base decays like 1/r instead of growing like r^(2*beta).
    return np.asarray(
        inverse_normalization * (product / total**1.5) ** index / total, dtype=np.float64
    )


def _cluster_function(
    separations: FloatArray, ensemble: CanonicalEnsemble, dimension: int | None
) -> FloatArray:
    """Two-level cluster function ``Y_2(s)``, shared by ``Sigma^2`` and ``Delta_3``.

    One implementation rather than two, because the Mehta relation that produces the
    ``Delta_3`` reference is only a valid cross-check of ``Sigma^2`` if both curves
    are built from the same ``Y_2``. Writing ``s(x) = sin(pi x)/(pi x)``::

        beta=2  Y_2 = s(r)^2
        beta=1  Y_2 = s(r)^2 + s'(r) * int_r^inf s
        beta=4  Y_2 = s(2r)^2 - (d/dr) s(2r) * int_0^r s(2r') dr'

    ``s`` is the mean spacing of the *distinct* levels in every case. For beta=4 that
    means one level per Kramers doublet: the doubled argument is exactly the
    statement that the symplectic kernel is built on half as many levels as the
    matrix has eigenvalues, and feeding it a spectrum with the doublets still in
    place compares against the wrong curve entirely.

    All three satisfy ``Y_2(0) = 1`` and ``int_0^inf Y_2 = 1/2``, and reproduce the
    level repulsion ``1 - Y_2(r) ~ r^beta``: measured here as ``pi^2 r/6`` (beta=1),
    ``pi^2 r^2/3`` (beta=2) and ``(2 pi r)^4/135`` (beta=4), the last agreeing with
    the closed form to 2e-4 relative at ``r=1e-2``. Note that the beta=4 tail decays
    only like ``1/r`` -- beta=1 and beta=2 both decay like ``1/r^2`` -- which is why
    the integrals over it are evaluated on fixed panels rather than adaptively.
    """
    values = np.ones_like(separations)
    nonzero = separations != 0.0
    scaled = separations[nonzero]
    if ensemble == "cue" and dimension is not None:
        denominator = dimension * np.sin(np.pi * scaled / dimension)
        degenerate = np.abs(denominator) <= np.finfo(np.float64).tiny
        safe = np.where(degenerate, 1.0, denominator)
        values[nonzero] = np.where(degenerate, 1.0, (np.sin(np.pi * scaled) / safe) ** 2)
        return values
    if ensemble == "goe":
        sine = np.sin(np.pi * scaled)
        sinc = sine / (np.pi * scaled)
        derivative = (np.pi * scaled * np.cos(np.pi * scaled) - sine) / (np.pi * scaled**2)
        tail = 0.5 - sici(np.pi * scaled)[0] / np.pi
        values[nonzero] = sinc**2 + derivative * tail
        return values
    if ensemble == "gse":
        # Same shape as beta=1 but with the sine kernel at doubled argument, the
        # complementary sine integral, and the opposite sign:
        # Y_2 = sbar^2 - sbar' * int_0^s sbar, with sbar(s) = sin(2 pi s)/(2 pi s).
        doubled = 2.0 * np.pi * scaled
        sine = np.sin(doubled)
        sinc = sine / doubled
        derivative = (doubled * np.cos(doubled) - sine) / (2.0 * np.pi * scaled**2)
        head = sici(doubled)[0] / (2.0 * np.pi)
        values[nonzero] = sinc**2 - derivative * head
        return values
    values[nonzero] = np.sinc(scaled) ** 2
    return values


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
    warnings: tuple[Diagnostic, ...] = (),
) -> ExperimentMetadata:
    return ExperimentMetadata(
        parameters={"statistic": statistic, "symmetry_sector": sector, **parameters},
        precision="float64",
        warnings=warnings,
    )


def _optional_array_equal(left: FloatArray | None, right: FloatArray | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(left, right))


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


__all__ = [
    "SpectralCurve",
    "Statistic",
    "Window",
    "number_variance",
    "rmt_reference",
    "spectral_form_factor",
    "spectral_rigidity",
]

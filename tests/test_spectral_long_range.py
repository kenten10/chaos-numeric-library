from __future__ import annotations

import contextlib
import copy
import itertools
import pickle
import warnings
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from scipy.integrate import quad, trapezoid  # type: ignore[import-untyped]

from chaos_numerics.core import (
    ConvergenceInfo,
    ExperimentMetadata,
    NumericalWarning,
    ValidationError,
)
from chaos_numerics.spectral import (
    Ensemble,
    PreparedEigenphases,
    SpacingDistributionResult,
    SpectralCurve,
    Statistic,
    UnfoldedSpectrum,
    number_variance,
    prepare_eigenphases,
    rmt_reference,
    spacing_distribution,
    spectral_form_factor,
    spectral_rigidity,
    unfold,
)

_EULER_MASCHERONI = 0.5772156649015329


@contextlib.contextmanager
def coarse_arcs_expected() -> Iterator[None]:
    """Accept the coarse-arc warning in a test that is about values, not error bars.

    ``number_variance`` and ``spectral_rigidity`` warn once a window length leaves
    too few side-by-side arcs for their batch-means error bar to be calibrated --
    eight and four respectively, both measured. Many tests in this file check the
    *values* against a closed form on a deliberately tiny synthetic spectrum, where
    a handful of arcs is unavoidable and the warning is simply true. Swallowing it
    here keeps ``filterwarnings = ["error"]`` meaningful everywhere else, and using
    a named helper rather than a bare filter keeps it visible which tests are in
    that position.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*leave fewer than \d+ independent windows.*",
            category=NumericalWarning,
        )
        yield


def _equally_spaced(count: int) -> PreparedEigenphases:
    return prepare_eigenphases(
        np.arange(count) * (2.0 * np.pi / count), symmetry_sector="equally-spaced"
    )


def _haar_unitary(
    rng: np.random.Generator, dimension: int
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    """Draw one Haar-distributed unitary (CUE) via a phase-fixed QR decomposition."""
    gaussian = rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal(
        (dimension, dimension)
    )
    unitary, triangular = np.linalg.qr(gaussian)
    diagonal = np.diag(triangular)
    return np.asarray(unitary * (diagonal / np.abs(diagonal)).conj())


def _symplectic_dual(
    matrix: np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    """Return the quaternion dual ``M^D = J M^T J^-1`` of a ``2n x 2n`` matrix."""
    pairs = matrix.shape[0] // 2
    symplectic = np.zeros_like(matrix)
    symplectic[:pairs, pairs:] = np.eye(pairs)
    symplectic[pairs:, :pairs] = -np.eye(pairs)
    return np.asarray(symplectic @ matrix.T @ (-symplectic))


def _haar_symplectic_eigenphases(
    rng: np.random.Generator, pairs: int
) -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
]:
    """Draw one Haar CSE spectrum and return it with and without its doublets.

    No model in this library has beta=4 statistics, so the only way to test the
    GSE references against data is to generate the ensemble here. ``U = V^D V``
    with ``V`` Haar on ``U(2n)`` is Dyson's circular symplectic ensemble: the dual
    satisfies ``(XY)^D = Y^D X^D`` and ``(X^D)^D = X``, so ``U^D = V^D (V^D)^D =
    U`` is self-dual, and a self-dual unitary has Kramers-degenerate eigenvalues.

    Both members of a doublet land on the same phase to about 1e-14, so sorting
    puts each pair at positions ``(2k, 2k+1)`` and ``phases[0::2]`` is one level
    per doublet. Every caller asserts that the within-pair gap really is at
    rounding before relying on that.
    """
    unitary = _haar_unitary(rng, 2 * pairs)
    self_dual = _symplectic_dual(unitary) @ unitary
    phases = np.sort(np.mod(np.angle(np.linalg.eigvals(self_dual)), 2.0 * np.pi))
    assert float(np.max(phases[1::2] - phases[0::2])) < 1e-10
    return phases, np.asarray(phases[0::2])


# Levels at the integers give an exactly computable circular number variance:
# a window of length L starting at a non-integer origin with fractional part phi
# holds floor(L) + [phi + frac(L) >= 1] levels, so averaging over uniform phi
# gives frac(L) * (1 - frac(L)) exactly, independent of the level count.
_EXACT_LENGTHS = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.25, 4.75, 8.5])
_EXACT_VARIANCE = (_EXACT_LENGTHS % 1.0) * (1.0 - _EXACT_LENGTHS % 1.0)


def test_spectral_form_factor_matches_direct_complex_sum() -> None:
    prepared = prepare_eigenphases([0.0, 0.7, 2.2, 5.0], symmetry_sector="all")
    unfolded = unfold(prepared, method="mean")
    times = np.asarray([0.0, 0.125, 0.5, 1.0])
    centered = unfolded.values - np.mean(unfolded.values)
    expected = np.abs(np.exp(2j * np.pi * np.outer(times, centered)).sum(axis=1)) ** 2 / 4

    result = spectral_form_factor(unfolded, times, connected=False)

    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(result.x, times)
    assert result.metadata.parameters["normalization"] == "sum(w^2); plateau=1"
    assert not result.values.flags.writeable


def test_connected_sff_bootstrap_is_seeded_and_returns_plot_ready_uncertainty() -> None:
    prepared = prepare_eigenphases(np.linspace(0.0, 2.0 * np.pi, 65)[:-1], symmetry_sector="all")
    times = np.linspace(0.0, 1.5, 12)

    # Every bootstrap call warns that its spread is a resampling diagnostic rather
    # than an error bar; the point here is that the numbers are seeded, not silent.
    with pytest.warns(NumericalWarning, match="resampling diagnostic"):
        first = spectral_form_factor(prepared, times, window="hann", bootstrap=32, seed=7)
    with pytest.warns(NumericalWarning, match="resampling diagnostic"):
        second = spectral_form_factor(prepared, times, window="hann", bootstrap=32, seed=7)

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.uncertainty, second.uncertainty)
    np.testing.assert_array_equal(first.variance, second.variance)
    assert first.values[0] == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(ValidationError, match="seed is required"):
        spectral_form_factor(prepared, times, bootstrap=2)


def test_window_normalization_is_pinned_by_exact_values_at_the_heisenberg_time() -> None:
    """Pin ``sum(w^2)`` normalization, the Hann shape, and the connected subtraction.

    For 64 levels at the integers and ``tau=1`` every ``sinc`` in the continuous
    background lands on a non-zero integer, so the connected subtraction is
    exactly zero and ``K(1) = (sum w)^2 / sum(w^2)``.

    ``sum(np.hanning(64)) = 31.5`` and ``sum(np.hanning(64)**2) = 23.625``
    exactly (the ``n=63`` sample repeats the ``n=0`` phase, contributing the
    leftover halves), giving ``31.5**2 / 23.625 = 42`` for the Hann window and
    ``64**2 / 64 = 64`` for the flat one. Normalizing by ``sum(w)`` instead would
    return 31.5, and a periodic instead of symmetric Hann shape would also move
    the value.
    """
    prepared = _equally_spaced(64)
    heisenberg = np.asarray([1.0])

    flat = spectral_form_factor(prepared, heisenberg, window="none", connected=True)
    hann = spectral_form_factor(prepared, heisenberg, window="hann", connected=True)

    np.testing.assert_allclose(flat.values, [64.0], rtol=1e-12)
    np.testing.assert_allclose(hann.values, [42.0], rtol=1e-12)
    assert np.sum(np.hanning(64)) == pytest.approx(31.5, rel=1e-15)
    assert np.sum(np.hanning(64) ** 2) == pytest.approx(23.625, rel=1e-15)


def test_number_variance_matches_direct_circular_window_counting() -> None:
    prepared = prepare_eigenphases([0.1, 1.0, 2.0, 4.0, 5.5], symmetry_sector="all")
    unfolded = unfold(prepared, method="mean")
    lengths = np.asarray([0.0, 0.75, 2.0, 4.0])
    samples = 256
    # L=4 on five levels leaves room for a single independent window, so the
    # scatter of the estimator cannot be resolved from one spectrum and the
    # function says so.
    # Two diagnostics fire together here: L=4 leaves a single window (unresolved)
    # and L=2 leaves four, below the eight the batch-means bar needs (coarse).
    with pytest.warns(NumericalWarning) as caught:
        result = number_variance(unfolded, lengths, samples=samples)
    messages = [str(item.message) for item in caught]
    assert any("fewer than two independent windows" in text for text in messages)

    levels = np.sort(np.mod(unfolded.values, unfolded.count))
    origins = (np.arange(samples) + 0.5) * unfolded.count / samples
    doubled = np.concatenate((levels, levels + unfolded.count))
    starts = np.searchsorted(levels, origins, side="left")
    expected = []
    for length in lengths:
        counts = np.searchsorted(doubled, origins + length, side="left") - starts
        expected.append(np.mean((counts - length) ** 2))

    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-14)
    assert result.uncertainty is not None
    assert result.variance is not None


@pytest.mark.parametrize("samples", [128, 256, 2048])
def test_number_variance_matches_the_exact_equally_spaced_result(samples: int) -> None:
    """Origins must not land on the levels themselves.

    With an origin grid starting at zero, ``samples`` being a multiple of the
    level count put every origin on a level, and the half-open
    ``searchsorted(..., "left")`` lookup then dropped the level at the left edge.
    On this spectrum that turned the exact ``0.0`` at integer ``L`` into
    ``0.0625``. The half-shifted grid reproduces the analytic value for every
    ``samples`` whose origin phases resolve the chosen fractional lengths.
    """
    prepared = _equally_spaced(32)

    with coarse_arcs_expected():
        result = number_variance(prepared, _EXACT_LENGTHS, samples=samples)

    np.testing.assert_allclose(result.values, _EXACT_VARIANCE, atol=1e-12)


def test_number_variance_finite_size_correction_rescales_the_exact_result() -> None:
    prepared = _equally_spaced(32)
    lengths = _EXACT_LENGTHS[:-1]
    expected = _EXACT_VARIANCE[:-1] / (1.0 - lengths / 32.0)

    with coarse_arcs_expected():
        corrected = number_variance(prepared, lengths, samples=128, finite_size_correction=True)

    np.testing.assert_allclose(corrected.values, expected, atol=1e-12)
    assert corrected.metadata.parameters["finite_size_factor"] == "1/(1-L/N)"
    with pytest.raises(ValidationError, match="less than N"):
        number_variance(prepared, [32.0], samples=128, finite_size_correction=True)


def test_number_variance_bootstrap_is_seeded_and_non_negative() -> None:
    prepared = _equally_spaced(32)
    first = number_variance(prepared, [1.5, 4.0], samples=128, bootstrap=24, seed=9)
    second = number_variance(prepared, [1.5, 4.0], samples=128, bootstrap=24, seed=9)

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.uncertainty, second.uncertainty)
    assert first.uncertainty is not None
    assert bool(np.all(first.uncertainty >= 0.0))
    with pytest.raises(ValidationError, match="seed is required"):
        number_variance(prepared, [1.5], samples=128, bootstrap=4)


def test_number_variance_counts_a_half_open_window() -> None:
    """``[origin, origin+L)`` excludes a level sitting exactly on the right edge.

    Four levels at the integers with ``samples=2*N`` puts the origins at
    ``0.25 + 0.5*k``, so every second window closes exactly on a level:
    ``[0.75, 1.0)``, ``[1.75, 2.0)``, and so on. Under the half-open convention
    those windows are empty and every count is ``0``, giving
    ``Sigma^2 = (0 - 0.25)^2 = 0.0625`` exactly. Closing the interval instead --
    ``searchsorted(..., "right")`` at the window end -- admits one level in half
    of the windows and returns ``0.3125``.

    The number is deliberately *not* the continuum answer
    ``frac(L)*(1-frac(L)) = 0.1875``; this grid is degenerate on purpose, which
    is exactly why it resolves the convention.
    """
    prepared = _equally_spaced(4)

    result = number_variance(prepared, [0.25], samples=8)

    assert float(result.values[0]) == pytest.approx(0.0625, abs=1e-15)
    assert result.metadata.parameters["counting_interval"] == "[origin, origin+L)"


def test_number_variance_uncertainty_is_calibrated_against_cue_realization_scatter() -> None:
    """The reported error must estimate the scatter over spectra, within a factor of two.

    This is the acceptance criterion for the estimator, measured rather than
    argued: 80 Haar-CUE spectra at ``N=128`` give a genuine standard deviation of
    ``Sigma^2(L)`` across realizations, and the number the function prints next to
    each value has to be that quantity to within a factor of two.

    The batch-means error over ``min(samples, floor(N/L))`` independent arcs comes
    out at ``0.85`` (``L=2``), ``0.71`` (``L=5``) and ``0.70`` (``L=10``) times the
    true scatter. The ``sqrt(var/samples)`` estimator that this replaced -- the
    naive column below, recomputed here from the same windows -- gives ``0.49``,
    ``0.32`` and ``0.25``, i.e. it understates the error by two to four times, and
    the second assertion fails if anyone puts it back. That bound is ``0.6`` rather
    than the measured ``0.49`` so that it does not sit on a boundary: the two
    estimators are a clear factor apart everywhere between ``N=64`` and ``N=256``.
    """
    rng = np.random.default_rng(4242)
    dimension = 128
    samples = 1024
    lengths = np.asarray([2.0, 5.0, 10.0])
    estimates = np.empty((80, lengths.size))
    reported = np.empty_like(estimates)
    naive = np.empty_like(estimates)
    for index in range(80):
        phases = np.angle(np.linalg.eigvals(_haar_unitary(rng, dimension)))
        prepared = prepare_eigenphases(phases, symmetry_sector="cue", degeneracy_tolerance=0.0)
        curve = number_variance(prepared, lengths, samples=samples)
        assert curve.uncertainty is not None
        estimates[index] = np.asarray(curve.values)
        reported[index] = np.asarray(curve.uncertainty)
        naive[index] = _naive_origin_standard_error(prepared, lengths, samples=samples)

    true_scatter = estimates.std(axis=0, ddof=1)
    ratio = reported.mean(axis=0) / true_scatter
    naive_ratio = naive.mean(axis=0) / true_scatter

    assert bool(np.all((ratio >= 0.5) & (ratio <= 2.0))), ratio
    assert bool(np.all(naive_ratio < 0.6)), naive_ratio


def _naive_origin_standard_error(
    prepared: PreparedEigenphases, lengths: np.ndarray, *, samples: int
) -> np.ndarray:
    """Recompute the discredited ``sqrt(var/samples)`` error bar over origins."""
    dimension = prepared.count
    levels = (np.asarray(prepared.phases) - prepared.phases[0]) / (2.0 * np.pi / dimension)
    canonical = np.sort(np.mod(levels, dimension))
    doubled = np.concatenate((canonical, canonical + dimension))
    origins = (np.arange(samples, dtype=np.float64) + 0.5) * dimension / samples
    starts = np.searchsorted(canonical, origins, side="left")
    out = np.empty(lengths.size)
    for index, length in enumerate(lengths):
        ends = np.searchsorted(doubled, origins + length, side="left")
        squared = (ends - starts - length) ** 2.0
        out[index] = np.sqrt(np.var(squared, ddof=1) / samples)
    return out


def test_number_variance_error_bar_stops_shrinking_once_the_windows_run_out() -> None:
    """More origins on one spectrum must not buy more precision without limit.

    This was the sharpest symptom of the old estimator: on a single fixed
    spectrum the value settled while ``sqrt(var/samples)`` kept falling, so at
    ``N=256, L=10`` a converged ``Sigma^2`` of 0.544 sat an apparent 16 sigma from
    the RMT reference 0.5793 purely because ``samples`` had been raised to 131072.
    ``effective_samples`` is a property of ``N`` and ``L`` alone, so it -- and
    with it the error bar -- is now flat in ``samples`` above the point where the
    windows stop fitting.
    """
    rng = np.random.default_rng(20260808)
    prepared = prepare_eigenphases(
        np.angle(np.linalg.eigvals(_haar_unitary(rng, 128))),
        symmetry_sector="cue",
        degeneracy_tolerance=0.0,
    )

    coarse = number_variance(prepared, [10.0], samples=512)
    fine = number_variance(prepared, [10.0], samples=32768)

    assert coarse.metadata.parameters["effective_samples"] == (12,)
    assert fine.metadata.parameters["effective_samples"] == (12,)
    assert coarse.uncertainty is not None
    assert fine.uncertainty is not None
    ratio = float(fine.uncertainty[0]) / float(coarse.uncertainty[0])
    assert 0.5 <= ratio <= 2.0, ratio
    # The naive estimator would have divided the error by sqrt(64) = 8 here.
    assert ratio > 0.5


def test_number_variance_metadata_publishes_effective_samples_and_error_semantics() -> None:
    prepared = _equally_spaced(64)
    lengths = np.asarray([0.0, 1.0, 4.0, 10.0])

    with coarse_arcs_expected():
        result = number_variance(prepared, lengths, samples=256)

    # min(samples, floor(N/L)), and the full origin count where L counts nothing.
    assert result.metadata.parameters["effective_samples"] == (256, 64, 16, 6)
    assert (
        result.metadata.parameters["effective_samples_rule"]
        == "min(samples, floor(N/L)); samples at L=0"
    )
    semantics = result.metadata.parameters["error_semantics"]
    assert isinstance(semantics, str)
    assert "batch-means" in semantics
    assert "NOT sqrt(var/samples)" in semantics

    with coarse_arcs_expected():
        booted = number_variance(prepared, lengths, samples=256, bootstrap=8, seed=3)
    boot_semantics = booted.metadata.parameters["error_semantics"]
    assert isinstance(boot_semantics, str)
    assert "block-bootstrap" in boot_semantics


def test_number_variance_block_bootstrap_agrees_with_the_batch_means_error() -> None:
    """The two error paths must describe the same quantity.

    Resampling origins instead of arcs is what made the bootstrap path reproduce
    the ``sqrt(var/samples)`` understatement; resampling the arcs makes the
    bootstrap standard deviation converge on ``sqrt(var(arc means)/arcs)``, so the
    two options no longer disagree by a factor of three.
    """
    prepared = _equally_spaced(64)
    rng = np.random.default_rng(17)
    phases = np.sort(rng.uniform(0.0, 2.0 * np.pi, 64))
    noisy = prepare_eigenphases(phases, symmetry_sector="uniform", degeneracy_tolerance=0.0)
    lengths = np.asarray([2.0, 5.0])

    for spectrum in (prepared, noisy):
        analytic = number_variance(spectrum, lengths, samples=512)
        booted = number_variance(spectrum, lengths, samples=512, bootstrap=400, seed=11)
        assert analytic.uncertainty is not None
        assert booted.uncertainty is not None
        np.testing.assert_allclose(booted.uncertainty, analytic.uncertainty, rtol=0.2, atol=1e-12)


def test_number_variance_bootstrap_variance_is_the_unbiased_two_sample_estimate() -> None:
    """``bootstrap=2`` pins ``ddof=1``: the variance is ``(x1-x2)^2/2``, not ``/4``.

    The two replicates are not observable from the outside, so they are rebuilt
    here from the same arcs and the same seeded generator. That makes this the one
    deliberately white-box test in the file; it exists because ``ddof=0`` halves
    every bootstrap variance in the library and nothing else notices.
    """
    dimension = 32
    samples = 64
    length = 2.0
    seed = 1
    rng = np.random.default_rng(seed)
    prepared = prepare_eigenphases(
        np.sort(rng.uniform(0.0, 2.0 * np.pi, dimension)),
        symmetry_sector="uniform",
        degeneracy_tolerance=0.0,
    )

    result = number_variance(prepared, [length], samples=samples, bootstrap=2, seed=seed)

    levels = (np.asarray(prepared.phases) - prepared.phases[0]) / (2.0 * np.pi / dimension)
    canonical = np.sort(np.mod(levels, dimension))
    doubled = np.concatenate((canonical, canonical + dimension))
    origins = (np.arange(samples, dtype=np.float64) + 0.5) * dimension / samples
    starts = np.searchsorted(canonical, origins, side="left")
    ends = np.searchsorted(doubled, origins + length, side="left")
    squared = (ends - starts - length) ** 2.0
    arcs = min(samples, int(dimension // length))
    arc_means = np.asarray([float(part.mean()) for part in np.array_split(squared, arcs)])
    replay = np.random.default_rng(seed)
    replicates = np.asarray(
        [float(arc_means[replay.integers(0, arcs, size=arcs)].mean()) for _ in range(2)]
    )

    assert arcs == 16
    assert result.metadata.parameters["effective_samples"] == (16,)
    assert result.variance is not None
    unbiased = float((replicates[0] - replicates[1]) ** 2 / 2.0)
    assert unbiased > 0.0
    assert float(result.variance[0]) == pytest.approx(unbiased, rel=1e-12)


def _naive_window_rigidity(levels: np.ndarray, origin: float, length: float) -> float:
    """Independent ``Delta_3`` of one window: exact segment integrals + normal equations.

    Nothing here is shared with the implementation. The staircase is constant on
    each segment between consecutive levels, so ``int f``, ``int x f`` and
    ``int f^2`` are elementary; the best straight line then comes from solving the
    2x2 least-squares system directly instead of projecting onto an orthogonal
    basis, which is what the closed form in ``spectral_rigidity`` does.
    """
    inside = np.sort(levels[(levels >= origin) & (levels < origin + length)] - origin)
    edges = np.concatenate(([0.0], inside, [length]))
    heights = np.arange(edges.size - 1, dtype=np.float64)
    constant = float(np.sum(heights * np.diff(edges)))
    linear = float(np.sum(heights * 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)))
    square = float(np.sum(heights**2 * np.diff(edges)))
    system = np.asarray([[length**3 / 3.0, 0.5 * length**2], [0.5 * length**2, length]])
    slope, offset = np.linalg.solve(system, np.asarray([linear, constant]))
    return float((square - slope * linear - offset * constant) / length)


def test_spectral_rigidity_matches_an_independent_window_integration() -> None:
    """The closed form must reproduce a naive exact integration window by window.

    ``N(E)`` is a staircase, so the least-squares residual inside a window has a
    finite closed form and the implementation evaluates it from prefix sums over the
    doubled level array. The reference here integrates each staircase segment
    exactly and solves the normal equations directly, sharing no algebra with it.
    Worst absolute disagreement over these five lengths is 5.0e-13, all of it the
    cancellation inside the prefix sums; the per-window formula on its own agrees to
    1.5e-15.
    """
    rng = np.random.default_rng(5)
    dimension = 24
    samples = 64
    prepared = prepare_eigenphases(
        np.sort(rng.uniform(0.0, 2.0 * np.pi, dimension)),
        symmetry_sector="uniform",
        degeneracy_tolerance=0.0,
    )
    levels = (np.asarray(prepared.phases) - prepared.phases[0]) / (2.0 * np.pi / dimension)
    circular = np.sort(np.mod(levels, dimension))
    # Three copies so that a window starting near the end of the circle is complete.
    tiled = np.concatenate((circular, circular + dimension, circular + 2 * dimension))
    origins = (np.arange(samples, dtype=np.float64) + 0.5) * dimension / samples

    for length in (0.7, 1.0, 2.5, 6.0, 11.0):
        with coarse_arcs_expected():
            curve = spectral_rigidity(prepared, [length], samples=samples)
        expected = float(
            np.mean([_naive_window_rigidity(tiled, float(origin), length) for origin in origins])
        )
        assert float(curve.values[0]) == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize(("samples", "tolerance"), [(4096, 2.1e-5), (65536, 8.0e-8)])
def test_spectral_rigidity_matches_the_exact_equally_spaced_result(
    samples: int, tolerance: float
) -> None:
    """Levels at the integers give ``Delta_3(L) = 1/12 - 1/(60*L^2)`` for integer ``L``.

    Derivation: against the best straight line the staircase of a unit lattice is
    the sawtooth ``-frac(E)``, whose mean square about its own mean is ``1/12``. For
    integer ``L`` the fitted slope is not zero -- it is set by
    ``c(phi) = 1/12 + (phi^2-phi)/2`` with ``phi`` the fractional window origin --
    and removing it subtracts ``12*c^2/L^2``. Averaging ``c^2`` over uniform ``phi``
    gives ``1/720``, hence ``1/12 - 1/(60*L^2)``: ``1/15`` at ``L=1``, ``19/240`` at
    ``L=2``, and ``1/12`` in the limit.

    The half-shifted origin grid is a midpoint rule for that average over ``phi``,
    so the measured value converges as ``1/(samples/N)^2``: raising ``samples`` from
    4096 to 65536 at ``N=64`` divides the error by ``16^2 = 256``, which the last
    assertion checks so that the agreement cannot be passed off as coincidence.
    """
    prepared = _equally_spaced(64)
    lengths = np.asarray([1.0, 2.0, 5.0, 10.0])
    exact = 1.0 / 12.0 - 1.0 / (60.0 * lengths**2)
    assert exact[0] == pytest.approx(1.0 / 15.0, rel=1e-15)
    assert exact[1] == pytest.approx(19.0 / 240.0, rel=1e-15)

    result = spectral_rigidity(prepared, lengths, samples=samples)

    np.testing.assert_allclose(result.values, exact, atol=tolerance)
    coarse = spectral_rigidity(prepared, lengths, samples=4096)
    fine = spectral_rigidity(prepared, lengths, samples=65536)
    ratio = np.abs(np.asarray(coarse.values) - exact) / np.abs(np.asarray(fine.values) - exact)
    np.testing.assert_allclose(ratio, 256.0, rtol=0.02)


def test_spectral_rigidity_uncertainty_is_calibrated_against_cue_realization_scatter() -> None:
    """The reported error must estimate the scatter over spectra, within a factor of two.

    Same acceptance criterion and the same measurement as for ``number_variance``,
    redone for this statistic rather than inherited: 80 Haar-CUE spectra at
    ``N=128`` with ``samples=1024`` give ratios of reported error to true
    realization standard deviation of 1.00 (``L=2``), 0.85 (``L=5``) and 0.79
    (``L=10``). Over four independent ensembles of 80 the ratios span 0.74 to 1.00,
    so the 0.5-2 band is not sitting on a boundary.
    """
    rng = np.random.default_rng(4242)
    dimension = 128
    lengths = np.asarray([2.0, 5.0, 10.0])
    estimates = np.empty((80, lengths.size))
    reported = np.empty_like(estimates)
    for index in range(80):
        prepared = prepare_eigenphases(
            np.angle(np.linalg.eigvals(_haar_unitary(rng, dimension))),
            symmetry_sector="cue",
            degeneracy_tolerance=0.0,
        )
        curve = spectral_rigidity(prepared, lengths, samples=1024)
        assert curve.uncertainty is not None
        estimates[index] = np.asarray(curve.values)
        reported[index] = np.asarray(curve.uncertainty)

    ratio = reported.mean(axis=0) / estimates.std(axis=0, ddof=1)

    assert bool(np.all((ratio >= 0.5) & (ratio <= 2.0))), ratio


def test_spectral_rigidity_bootstrap_and_metadata_follow_the_number_variance_contract() -> None:
    prepared = _equally_spaced(64)
    lengths = np.asarray([0.0, 1.0, 4.0, 10.0])

    result = spectral_rigidity(prepared, lengths, samples=256)

    assert float(result.values[0]) == 0.0
    assert result.metadata.parameters["statistic"] == "spectral_rigidity"
    assert result.metadata.parameters["effective_samples"] == (256, 64, 16, 6)
    assert (
        result.metadata.parameters["effective_samples_rule"]
        == "min(samples, floor(N/L)); samples at L=0"
    )
    assert result.metadata.parameters["window_fit"] == (
        "least-squares straight line, evaluated in closed form"
    )
    semantics = result.metadata.parameters["error_semantics"]
    assert isinstance(semantics, str)
    assert "batch-means" in semantics
    assert "NOT sqrt(var/samples)" in semantics

    first = spectral_rigidity(prepared, lengths, samples=256, bootstrap=24, seed=9)
    second = spectral_rigidity(prepared, lengths, samples=256, bootstrap=24, seed=9)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.uncertainty, second.uncertainty)
    assert first.uncertainty is not None
    assert bool(np.all(first.uncertainty >= 0.0))
    boot_semantics = first.metadata.parameters["error_semantics"]
    assert isinstance(boot_semantics, str)
    assert "block-bootstrap" in boot_semantics
    with pytest.raises(ValidationError, match="seed is required"):
        spectral_rigidity(prepared, lengths, samples=256, bootstrap=4)


def test_spectral_rigidity_warns_when_a_single_window_fills_the_spectrum() -> None:
    rng = np.random.default_rng(23)
    prepared = prepare_eigenphases(
        np.sort(rng.uniform(0.0, 2.0 * np.pi, 16)),
        symmetry_sector="uniform",
        degeneracy_tolerance=0.0,
    )

    with pytest.warns(NumericalWarning, match="fewer than two independent windows"):
        result = spectral_rigidity(prepared, [4.0, 12.0], samples=256)

    assert result.metadata.parameters["effective_samples"] == (4, 1)
    assert [item.code for item in result.metadata.warnings] == [
        "spectral-rigidity-unresolved-error"
    ]
    assert result.uncertainty is not None
    assert float(result.uncertainty[1]) > float(result.uncertainty[0])


@pytest.mark.parametrize(
    "roundtrip",
    [lambda item: pickle.loads(pickle.dumps(item)), copy.deepcopy],
    ids=["pickle", "deepcopy"],
)
def test_spectral_rigidity_curve_survives_pickle_and_deepcopy(
    roundtrip: Callable[[SpectralCurve], SpectralCurve],
) -> None:
    prepared = _equally_spaced(32)
    original = spectral_rigidity(prepared, [1.0, 2.5], samples=128, bootstrap=8, seed=2)

    restored = roundtrip(original)

    assert restored == original
    assert restored.uncertainty is not None
    assert restored.variance is not None
    for array in (restored.x, restored.values, restored.uncertainty, restored.variance):
        assert not array.flags.writeable
    assert restored.metadata.parameters == original.metadata.parameters
    assert restored != rmt_reference("spectral_rigidity", "gue", [1.0, 2.5])


def test_rmt_spectral_rigidity_poisson_is_exactly_l_over_fifteen() -> None:
    """``Delta_3_Poisson(L) = L/15`` is exact, so it pins the Mehta kernel outright.

    ``Y_2 = 0`` for a Poisson spectrum, so the Mehta relation reduces to
    ``(2/L^4) * int_0^L (L^3 - 2 L^2 r + r^3) r dr = (2/L^4)(L^5/30) = L/15`` with no
    quadrature left in it. Any error in the polynomial weight, the ``2/L^4``
    prefactor, or the reduction that turns the nested integral into one integral over
    ``Y_2`` shows up here immediately, which is why the tolerance is ``1e-6``
    relative and the implementation in fact reproduces it bit for bit.
    """
    lengths = np.asarray([0.0, 0.25, 1.0, 3.0, 7.5, 20.0, 100.0])

    curve = rmt_reference("spectral_rigidity", "poisson", lengths)

    np.testing.assert_allclose(curve.values, lengths / 15.0, rtol=1e-6, atol=1e-12)
    np.testing.assert_array_equal(curve.values, lengths / 15.0)
    assert curve.metadata.parameters["reference_type"] == "mehta_integral_of_number_variance"


@pytest.mark.parametrize("ensemble", ["goe", "gue", "gse"])
def test_rmt_spectral_rigidity_agrees_with_the_direct_mehta_integral(ensemble: Ensemble) -> None:
    """The single reduced quadrature must equal the nested integral over ``Sigma^2``.

    The implementation does the ``r`` integral of
    ``Delta_3(L) = (2/L^4) int_0^L (L^3 - 2 L^2 r + r^3) Sigma^2(r) dr`` analytically
    so that only one quadrature over the two-level cluster function remains. This
    test performs the nested integral instead, calling the *public*
    ``"number_variance"`` reference for every ``Sigma^2(r)``, and the two agree to
    1.8e-15 relative at worst over these six lengths.

    Running it for beta=4 as well is what ties the GSE rigidity to the GSE number
    variance: both are the same ``Y_2`` under different weights, and a doubling
    convention applied in one place and not the other would show up here even
    though every self-consistency check inside a single statistic would still pass.
    """
    for length in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):

        def weighted(radius: float, length: float = length) -> float:
            sigma = float(rmt_reference("number_variance", ensemble, [radius]).values[0])
            return (length**3 - 2.0 * length**2 * radius + radius**3) * sigma

        edges = np.unique(np.concatenate((np.arange(0.0, length, 1.0), [length])))
        nested = 0.0
        for lower, upper in itertools.pairwise(edges):
            nested += quad(weighted, lower, upper, epsabs=1e-11, epsrel=1e-11)[0]
        expected = 2.0 * nested / length**4

        curve = rmt_reference("spectral_rigidity", ensemble, [length])

        assert float(curve.values[0]) == pytest.approx(expected, rel=1e-12)


def test_rmt_spectral_rigidity_large_l_slopes_match_the_analytic_logarithms() -> None:
    """``Delta_3`` must grow as ``(1/pi^2) ln L`` for beta=1 and half that for beta=2.

    The slopes are determined here by regression rather than compared against a
    remembered constant. Over ``L`` in ``[512, 8192]`` the fit gives 0.1012571 for
    GOE against ``1/pi^2 = 0.1013212`` (ratio 0.99937) and 0.05066059 for GUE
    against ``1/(2*pi^2)`` (ratio 1.0000000). The GOE shortfall is a ``1/L``
    correction, not a wrong slope: refitting with a ``c/L`` term brings it to
    ``0.9999988/pi^2``, which is the tight assertion below.

    Reading the constants off with the slope pinned to the analytic value gives
    ``Delta_3_GOE -> (1/pi^2) ln L - 0.006950`` and
    ``Delta_3_GUE -> (1/(2 pi^2)) ln L + 0.0590243``. The GUE constant is stable to
    eight digits from ``L=512`` up; the GOE one still drifts by ``1e-5`` per octave
    and is quoted as a ``1/L`` extrapolation.
    """
    lengths = np.asarray([512.0, 1024.0, 2048.0, 4096.0, 8192.0])
    logarithm = np.log(lengths)
    fits: tuple[tuple[Ensemble, float], ...] = (
        ("goe", 1.0 / np.pi**2),
        ("gue", 1.0 / (2.0 * np.pi**2)),
    )
    for ensemble, analytic in fits:
        values = np.asarray(rmt_reference("spectral_rigidity", ensemble, lengths).values)
        slope, _ = np.polyfit(logarithm, values, 1)
        design = np.column_stack((logarithm, np.ones_like(lengths), 1.0 / lengths))
        corrected = np.linalg.lstsq(design, values, rcond=None)[0]

        assert float(slope) == pytest.approx(analytic, rel=1e-3)
        assert float(corrected[0]) == pytest.approx(analytic, rel=1e-5)

    gue_constant = np.asarray(
        rmt_reference("spectral_rigidity", "gue", lengths).values
    ) - logarithm / (2.0 * np.pi**2)
    np.testing.assert_allclose(gue_constant, 0.0590243, atol=1e-7)


def test_rmt_spectral_rigidity_goe_climbs_toward_twice_the_gue_curve() -> None:
    """The GOE/GUE ratio tends to 2, but only logarithmically slowly.

    Measured: 1.342 at ``L=10``, 1.576 at ``L=100``, 1.695 at ``L=1000``, 1.758 at
    ``L=8192``. Asserting "close to 2" at any reachable ``L`` would therefore be
    asserting the wrong thing; what is exactly 2 is the ratio of the two logarithmic
    slopes, which the previous test pins. Here the requirement is that the ratio is
    monotone, still below 2, and already past 1.75 by ``L=8192``.
    """
    lengths = np.asarray([10.0, 100.0, 1000.0, 8192.0])
    goe = np.asarray(rmt_reference("spectral_rigidity", "goe", lengths).values)
    gue = np.asarray(rmt_reference("spectral_rigidity", "gue", lengths).values)

    ratio = goe / gue

    np.testing.assert_allclose(ratio, [1.341554, 1.575816, 1.694605, 1.757552], atol=1e-5)
    assert bool(np.all(np.diff(ratio) > 0.0))
    assert bool(np.all(ratio < 2.0))
    assert float(ratio[-1]) > 1.75


@pytest.mark.parametrize(("label", "other"), [("cue", "goe"), ("coe", "gue")])
def test_measured_rigidity_tracks_its_own_ensemble_and_not_the_other_beta(
    label: Ensemble, other: Ensemble
) -> None:
    """Haar CUE and COE spectra must land on their own ``Delta_3`` reference.

    40 members of dimension 128 give a mean whose distance from the correct
    reference is at most 3.0e-3 over these lengths (worst case over three seeds),
    while the distance from the *other* Dyson index reaches 5.8e-2 at ``L=10`` --
    twenty times larger. The tolerance is therefore a statement about the sampling
    noise of this ensemble, not about how good the reference is, and the second
    assertion is what makes the test discriminating.
    """
    rng = np.random.default_rng(909)
    lengths = np.asarray([1.0, 2.0, 5.0, 10.0])
    samples = []
    for _ in range(40):
        unitary = _haar_unitary(rng, 128)
        if label == "coe":
            unitary = unitary.T @ unitary
        prepared = prepare_eigenphases(
            np.angle(np.linalg.eigvals(unitary)),
            symmetry_sector=label,
            degeneracy_tolerance=0.0,
        )
        samples.append(np.asarray(spectral_rigidity(prepared, lengths, samples=1024).values))
    measured = np.mean(samples, axis=0)

    own = np.asarray(rmt_reference("spectral_rigidity", label, lengths).values)
    wrong = np.asarray(rmt_reference("spectral_rigidity", other, lengths).values)

    np.testing.assert_allclose(measured, own, atol=5e-3)
    assert float(np.abs(measured[-1] - wrong[-1])) > 3e-2


def test_spectral_rigidity_of_a_uniform_random_spectrum_follows_poisson() -> None:
    """Uncorrelated levels must reproduce the exact Poisson ``L/15``.

    20 uniform spectra of 512 levels agree with ``L/15`` to 2.3% at worst over three
    seeds. The GUE reference over the same lengths is 9% lower at ``L=1``, where
    every ensemble still has ``Delta_3 -> L/15``, and 69% lower by ``L=8``, so the
    5% tolerance separates uncorrelated from rigid levels everywhere except at the
    smallest length, where nothing can.
    """
    rng = np.random.default_rng(77)
    lengths = np.asarray([1.0, 2.0, 4.0, 8.0])
    samples = []
    for _ in range(20):
        prepared = prepare_eigenphases(
            np.sort(rng.uniform(0.0, 2.0 * np.pi, 512)),
            symmetry_sector="uniform",
            degeneracy_tolerance=0.0,
        )
        samples.append(
            np.asarray(spectral_rigidity(unfold(prepared), lengths, samples=2048).values)
        )
    measured = np.mean(samples, axis=0)

    np.testing.assert_allclose(measured, lengths / 15.0, rtol=0.05)
    rigid = np.asarray(rmt_reference("spectral_rigidity", "gue", lengths).values)
    poisson = lengths / 15.0
    np.testing.assert_allclose(rigid / poisson, [0.910, 0.709, 0.485, 0.308], atol=1e-3)
    assert bool(np.all(rigid < poisson))


def test_number_variance_warns_when_a_single_window_fills_the_spectrum() -> None:
    rng = np.random.default_rng(23)
    prepared = prepare_eigenphases(
        np.sort(rng.uniform(0.0, 2.0 * np.pi, 16)),
        symmetry_sector="uniform",
        degeneracy_tolerance=0.0,
    )

    with pytest.warns(NumericalWarning) as caught:
        result = number_variance(prepared, [4.0, 12.0], samples=256)
    messages = [str(item.message) for item in caught]
    # L=12 leaves one window, L=4 leaves four: one of each diagnostic.
    assert any("fewer than two independent windows" in text for text in messages)
    assert any("leave fewer than 8 independent windows" in text for text in messages)

    assert result.metadata.parameters["effective_samples"] == (4, 1)
    assert [item.code for item in result.metadata.warnings] == [
        "number-variance-unresolved-error",
        "number-variance-coarse-error",
    ]
    assert result.uncertainty is not None
    # The single-window spread bounds the error from above rather than dividing it
    # down by an origin count that measures nothing.
    assert float(result.uncertainty[1]) > float(result.uncertainty[0])


def test_form_factor_bootstrap_variance_is_the_unbiased_two_sample_estimate() -> None:
    """``bootstrap=2`` pins ``ddof=1`` on the form-factor path as well.

    Dropping to ``ddof=0`` halves every bootstrap variance, and nothing about the
    shape, the sign, or the seeding of the result changes -- so only an identity
    on a known number of replicates catches it. With ``connected=False`` and a
    flat window the two replicates are a plain resampled complex sum, which is
    reproducible here from the same seeded generator.
    """
    dimension = 32
    times = np.asarray([0.3])
    seed = 6
    prepared = _equally_spaced(dimension)

    with pytest.warns(NumericalWarning, match="resampling diagnostic"):
        result = spectral_form_factor(
            prepared, times, window="none", connected=False, bootstrap=2, seed=seed
        )

    centered = np.arange(dimension, dtype=np.float64) - (dimension - 1) / 2.0
    phases = np.exp(2j * np.pi * np.outer(times, centered))
    replay = np.random.default_rng(seed)
    replicates = np.asarray(
        [
            float(np.abs(phases[:, replay.integers(0, dimension, size=dimension)].sum()) ** 2)
            / dimension
            for _ in range(2)
        ]
    )

    assert result.variance is not None
    assert result.uncertainty is not None
    unbiased = float((replicates[0] - replicates[1]) ** 2 / 2.0)
    assert unbiased > 0.0
    assert float(result.variance[0]) == pytest.approx(unbiased, rel=1e-10)
    assert float(result.uncertainty[0]) == pytest.approx(np.sqrt(unbiased), rel=1e-10)


def test_rmt_form_factor_and_number_variance_references() -> None:
    times = np.asarray([0.0, 0.25, 0.5, 1.0, 2.0])
    poisson = rmt_reference("spectral_form_factor", "poisson", times)
    gue = rmt_reference("spectral_form_factor", "gue", times)
    cue = rmt_reference("spectral_form_factor", "cue", times, dimension=64)
    goe = rmt_reference("spectral_form_factor", "goe", times)

    np.testing.assert_array_equal(poisson.values, [0.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(gue.values, np.minimum(times, 1.0))
    np.testing.assert_array_equal(cue.values, gue.values)
    assert bool(np.all(goe.values[1:3] >= gue.values[1:3]))
    assert bool(np.all(goe.values[3:] <= gue.values[3:]))

    lengths = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0])
    poisson_variance = rmt_reference("number_variance", "poisson", lengths)
    goe_variance = rmt_reference("number_variance", "goe", lengths)
    gue_variance = rmt_reference("number_variance", "gue", lengths)
    cue_variance = rmt_reference("number_variance", "cue", lengths, dimension=64)

    np.testing.assert_array_equal(poisson_variance.values, lengths)
    assert bool(np.all(goe_variance.values[1:] > gue_variance.values[1:]))
    # The finite-N=64 CUE kernel departs from the GUE bulk limit by 1.34e-3
    # relative at L=4, the largest length here; the bound is a genuine cap on
    # that difference rather than a blanket 2% allowance.
    np.testing.assert_allclose(cue_variance.values, gue_variance.values, rtol=3e-3, atol=1e-4)


def test_rmt_references_match_closed_form_and_asymptotic_values() -> None:
    """Pin the reference curves to values that exist independently of this code."""
    goe = rmt_reference("spectral_form_factor", "goe", [1.0])
    np.testing.assert_allclose(goe.values, [2.0 - np.log(3.0)], rtol=1e-12)
    assert 2.0 - np.log(3.0) == pytest.approx(0.9013877113318902, rel=1e-15)

    # Sigma^2_GUE(L) -> (1/pi^2)(ln(2 pi L) + gamma + 1) for large L.
    asymptotic = (np.log(2.0 * np.pi * 16.0) + _EULER_MASCHERONI + 1.0) / np.pi**2
    gue = rmt_reference("number_variance", "gue", [16.0])
    assert float(gue.values[0]) == pytest.approx(asymptotic, abs=5e-5)
    assert float(gue.values[0]) == pytest.approx(0.626933, abs=1e-6)


def test_goe_number_variance_matches_its_large_l_asymptote_and_not_only_gue() -> None:
    """Pin the GOE cluster function absolutely, not just relative to GUE.

    ``Sigma^2_GOE(L) -> (2/pi^2)(ln(2 pi L) + gamma + 1 - pi^2/8)``, which fixes
    the sign of the ``0.5 - Si(pi s)/pi`` tail of the two-level cluster function.
    Flipping that sign to ``0.5 + Si(pi s)/pi`` leaves ``Sigma^2_GOE > Sigma^2_GUE``
    intact -- the only thing the other tests check -- while sending
    ``Sigma^2_GOE(1)`` from 0.4463 to 1.0630 and ``Sigma^2_GOE(16)`` from 1.0039 to
    31.26.

    ``L=16`` is where the asymptote is worth testing against: the exact curve is
    2.0e-5 away there, but 4.3e-3 away at ``L=1``, so a tight bound at small ``L``
    would be testing the asymptotic expansion rather than the cluster integral.
    ``L=1`` is therefore pinned to its own numerical value instead.
    """
    asymptotic = (2.0 / np.pi**2) * (
        np.log(2.0 * np.pi * 16.0) + _EULER_MASCHERONI + 1.0 - np.pi**2 / 8.0
    )
    assert asymptotic == pytest.approx(1.0038856, abs=1e-6)

    curve = rmt_reference("number_variance", "goe", [1.0, 16.0])

    assert float(curve.values[1]) == pytest.approx(asymptotic, abs=5e-5)
    assert float(curve.values[1]) == pytest.approx(1.003906, abs=1e-6)
    assert float(curve.values[0]) == pytest.approx(0.446334, abs=1e-6)
    # The asymptote is only asymptotic: at L=1 it misses by 4.3e-3, two orders
    # above the L=16 agreement, which is why the bound above is applied at L=16.
    small = (2.0 / np.pi**2) * (np.log(2.0 * np.pi) + _EULER_MASCHERONI + 1.0 - np.pi**2 / 8.0)
    assert abs(float(curve.values[0]) - small) == pytest.approx(4.29e-3, rel=0.05)


def test_connected_form_factor_removes_the_background_away_from_the_integer_grid() -> None:
    """The subtracted background carries the level count, not just the ``sinc`` shape.

    Every existing form-factor test evaluates at ``tau`` where ``N*tau`` is an
    integer, and there ``sinc(N*tau)`` is zero, so the background term vanishes
    whatever prefactor it is given. Dropping the ``count`` factor -- subtracting
    ``sinc(N tau)`` instead of ``N sinc(N tau)`` -- is therefore invisible on that
    grid, while at ``tau=0`` it turns the exact ``K=0`` of an equally spaced
    spectrum into ``(N-1)^2/N = 62.0`` for ``N=64``.

    ``tau = 1/(2N)`` is the first half-integer point, where ``sinc = 2/pi`` and the
    background is at its largest away from the origin; the correct subtraction
    leaves ``2.6e-07`` there and the truncated one leaves about 25.
    """
    prepared = _equally_spaced(64)
    taus = np.asarray([0.0, 1.0 / 128.0])

    curve = spectral_form_factor(prepared, taus, window="none", connected=True)

    assert float(curve.values[0]) <= 1e-12
    assert float(curve.values[1]) <= 1e-5
    # Without the subtraction the same two points are of order N.
    unconnected = spectral_form_factor(prepared, taus, window="none", connected=False)
    assert float(unconnected.values[0]) == pytest.approx(64.0, rel=1e-12)


def test_form_factor_bootstrap_is_labelled_as_a_resampling_diagnostic() -> None:
    """``K(tau)`` does not self-average, so the bootstrap is not an ensemble error.

    Measured against 150 Haar-CUE spectra at ``N=128`` the bootstrap standard
    deviation is 4.4x the true realization scatter at ``tau=0.3``, about 1.5x at
    ``tau>=1``, and **24x at ``tau=0.05``** -- it grows without bound as ``tau``
    approaches zero, because resampling destroys the rigidity and the replicate
    spread stops depending on ``tau`` while the true scatter falls to zero with
    ``K``. It errs high, so it cannot fabricate a detection, but it must not be
    presented as an uncertainty on the physics.

    A metadata label is not enough on its own: a caller who reads
    ``curve.uncertainty`` in code never reads metadata either. So the call also
    warns, and that is asserted here alongside the label.
    """
    prepared = _equally_spaced(64)
    times = np.asarray([0.3, 1.0])

    plain = spectral_form_factor(prepared, times)
    with pytest.warns(NumericalWarning, match="not an error bar on K"):
        booted = spectral_form_factor(prepared, times, bootstrap=16, seed=4)

    assert plain.metadata.parameters["error_semantics"] == (
        "none; no uncertainty is reported without bootstrap"
    )
    assert plain.uncertainty is None
    semantics = booted.metadata.parameters["error_semantics"]
    assert isinstance(semantics, str)
    assert "NOT realization-to-" in semantics
    assert "does not self-average" in semantics
    assert booted.uncertainty is not None


@pytest.mark.parametrize(
    "roundtrip",
    [lambda item: pickle.loads(pickle.dumps(item)), copy.deepcopy],
    ids=["pickle", "deepcopy"],
)
def test_spectral_curve_survives_pickle_and_deepcopy(
    roundtrip: Callable[[SpectralCurve], SpectralCurve],
) -> None:
    """A round trip must preserve values, read-only arrays, and equality.

    NumPy drops ``writeable=False`` when it pickles an array, and ``deepcopy``
    goes through the same protocol, so a container without ``__reduce__`` hands
    back a mutable copy of something the public API documents as read-only.
    """
    prepared = _equally_spaced(32)
    original = number_variance(prepared, [1.0, 2.5], samples=128, bootstrap=8, seed=2)

    restored = roundtrip(original)

    assert restored == original
    np.testing.assert_array_equal(restored.x, original.x)
    np.testing.assert_array_equal(restored.values, original.values)
    assert restored.uncertainty is not None
    assert restored.variance is not None
    for array in (restored.x, restored.values, restored.uncertainty, restored.variance):
        assert not array.flags.writeable
    assert restored.metadata.parameters == original.metadata.parameters
    assert restored != rmt_reference("number_variance", "gue", [1.0, 2.5])
    assert original.__eq__(object()) is NotImplemented


def test_coe_is_accepted_as_a_goe_alias_and_errors_list_the_candidates() -> None:
    """Bulk statistics only depend on beta, so COE == GOE, CUE == GUE, CSE == GSE."""
    lengths = np.asarray([0.5, 2.0, 4.0])
    coe = rmt_reference("number_variance", "coe", lengths)
    goe = rmt_reference("number_variance", "goe", lengths)

    np.testing.assert_array_equal(coe.values, goe.values)
    np.testing.assert_array_equal(
        rmt_reference("spectral_form_factor", "coe", lengths).values,
        rmt_reference("spectral_form_factor", "goe", lengths).values,
    )
    assert coe.metadata.parameters["ensemble"] == "GOE"
    assert coe.metadata.parameters["requested_ensemble"] == "COE"
    assert coe.metadata.parameters["ensemble_alias_applied"] is True
    assert goe.metadata.parameters["requested_ensemble"] == "GOE"
    assert goe.metadata.parameters["ensemble_alias_applied"] is False

    # CUE == GUE in the bulk; only an explicit dimension selects the finite kernel.
    np.testing.assert_array_equal(
        rmt_reference("number_variance", "cue", lengths).values,
        rmt_reference("number_variance", "gue", lengths).values,
    )

    # CSE == GSE outright: unlike "cue"/"gue" there is no finite-N symplectic
    # kernel for a dimension to select, so the two labels can never differ.
    cse = rmt_reference("number_variance", "cse", lengths)
    np.testing.assert_array_equal(
        cse.values, rmt_reference("number_variance", "gse", lengths).values
    )
    assert cse.metadata.parameters["ensemble"] == "GSE"
    assert cse.metadata.parameters["requested_ensemble"] == "CSE"
    assert cse.metadata.parameters["ensemble_alias_applied"] is True

    with pytest.raises(
        ValidationError, match=r"'poisson', 'goe', 'gue', 'cue', 'coe', 'gse', 'cse'"
    ):
        rmt_reference("number_variance", "cpe", [1.0])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match=r"'spectral_form_factor', 'number_variance'"):
        rmt_reference("rigidity", "goe", [1.0])  # type: ignore[arg-type]


@pytest.mark.parametrize("ensemble", ["poisson", "goe", "coe", "gue", "cue", "gse", "cse"])
def test_spacing_distribution_reference_has_unit_mass_and_unit_mean(ensemble: Ensemble) -> None:
    """Every Wigner surmise must be normalized in both senses.

    ``integral P(s) ds = 1`` and ``integral s P(s) ds = 1`` are what make the curve
    comparable to a ``spacing_distribution`` histogram of spacings rescaled to unit
    empirical mean. The exponential rates ``pi/4``, ``4/pi`` and ``64/(9*pi)`` are
    fixed by the second condition, so a prefactor changed without its rate, or two
    Wigner-Dyson rows swapped, breaks one or both integrals. On this grid the
    trapezoid error is below 1e-10, so 1e-6 tests the formulas, not the quadrature.
    """
    grid = np.linspace(0.0, 40.0, 400_001)
    density = np.asarray(rmt_reference("spacing_distribution", ensemble, grid).values)

    assert float(trapezoid(density, grid)) == pytest.approx(1.0, rel=1e-6)
    assert float(trapezoid(grid * density, grid)) == pytest.approx(1.0, rel=1e-6)


def test_spacing_distribution_reference_matches_closed_form_peaks_and_repulsion() -> None:
    """Pin the surmise shapes to values derivable by hand from the formulas."""
    poisson = rmt_reference("spacing_distribution", "poisson", [0.0, 1.0, 2.0])
    np.testing.assert_allclose(poisson.values, [1.0, np.exp(-1.0), np.exp(-2.0)], rtol=1e-14)

    # dP/ds = 0 at s = sqrt(beta/(2b)) with b the Gaussian rate: sqrt(2/pi) for GOE
    # giving P = sqrt(pi/2)*exp(-1/2), sqrt(pi)/2 for GUE giving P = (8/pi)*exp(-1),
    # and (3/4)*sqrt(pi/2) for GSE giving P = (256/(9*pi))*exp(-2).
    goe_peak = rmt_reference("spacing_distribution", "goe", [np.sqrt(2.0 / np.pi)])
    gue_peak = rmt_reference("spacing_distribution", "gue", [0.5 * np.sqrt(np.pi)])
    gse_peak = rmt_reference("spacing_distribution", "gse", [0.75 * np.sqrt(0.5 * np.pi)])
    np.testing.assert_allclose(goe_peak.values, [np.sqrt(0.5 * np.pi) * np.exp(-0.5)], rtol=1e-14)
    np.testing.assert_allclose(gue_peak.values, [8.0 / np.pi * np.exp(-1.0)], rtol=1e-14)
    np.testing.assert_allclose(gse_peak.values, [256.0 / (9.0 * np.pi) * np.exp(-2.0)], rtol=1e-14)
    assert float(goe_peak.values[0]) == pytest.approx(0.7601734505331402, rel=1e-14)
    assert float(gue_peak.values[0]) == pytest.approx(0.9367973043891067, rel=1e-14)
    assert float(gse_peak.values[0]) == pytest.approx(1.2253456669496110, rel=1e-14)
    # beta=4 peaks later and higher: the s^4 repulsion pushes the mode from 0.5642
    # (beta=1) and 0.8862 (beta=2) out to 0.9400 while unit mean is preserved.
    assert 0.75 * np.sqrt(0.5 * np.pi) == pytest.approx(0.9399856029866251, rel=1e-14)

    # Level repulsion: s^beta at small s, and Poisson does not vanish at all.
    # A tiny s isolates the leading power.
    tiny = np.asarray([1e-4, 2e-4])
    for ensemble, index in (("goe", 1.0), ("gue", 2.0), ("gse", 4.0)):
        density = np.asarray(
            rmt_reference("spacing_distribution", ensemble, tiny).values  # type: ignore[arg-type]
        )
        assert float(density[1] / density[0]) == pytest.approx(2.0**index, rel=1e-7)


def test_spacing_distribution_reference_rejects_dimension_and_negative_spacings() -> None:
    """``dimension`` is refused rather than ignored, and ``s < 0`` is not a spacing.

    The surmise is the exact spacing density of a 2x2 matrix carried over to the
    bulk; it has no ``N`` in it, so there is no finite-size correction to apply.
    Accepting and ignoring the argument would let a caller believe one had been.
    """
    spacings = np.asarray([0.5, 1.0, 2.0])
    curve = rmt_reference("spacing_distribution", "gue", spacings)

    assert curve.metadata.parameters["reference_type"] == "wigner_surmise_2x2"
    assert curve.metadata.parameters["finite_size"] == "not applicable"
    assert curve.metadata.parameters["dimension"] is None

    np.testing.assert_array_equal(
        rmt_reference("spacing_distribution", "coe", spacings).values,
        rmt_reference("spacing_distribution", "goe", spacings).values,
    )
    np.testing.assert_array_equal(
        rmt_reference("spacing_distribution", "cue", spacings).values,
        curve.values,
    )

    with pytest.raises(ValidationError, match="dimension does not apply"):
        rmt_reference("spacing_distribution", "gue", spacings, dimension=64)
    with pytest.raises(ValidationError, match="dimension does not apply"):
        rmt_reference("spacing_distribution", "gue", spacings, dimension=0)
    with pytest.raises(ValidationError, match="x must be non-negative"):
        rmt_reference("spacing_distribution", "gue", [-0.5, 0.5])


@pytest.mark.parametrize(("label", "other"), [("coe", "cue"), ("cue", "coe")])
def test_measured_histograms_track_their_own_surmise_and_not_the_other_one(
    label: Ensemble, other: Ensemble
) -> None:
    """The histogram and the analytic curve must be the same object numerically.

    120 circular ensemble members of dimension 128 give 15360 spacings, so the
    counting noise on a width-0.2 density bin is about 0.013 near the peak and the
    maximum over 20 bins runs to 0.046 (worst over 9 seeds: COE 0.0456, CUE
    0.0414). The 0.075 bound is therefore a noise bound and says nothing about how
    good the surmise is. Raising the sample to 102400 spacings (400 members at
    dimension 256) only lowers the maximum deviation to 0.018-0.022 for COE and
    0.014-0.018 for CUE, which is still what pure counting noise predicts for that
    sample size; at this bin width the surmise-versus-exact discrepancy is
    therefore below roughly 0.02 and this comparison cannot resolve it. Anyone
    wanting to *measure* the discrepancy needs the exact Gaudin-Mehta density,
    which this library does not provide.

    What makes the test discriminating despite that is the second assertion. The
    same histogram compared against the *other* Dyson index misses by 0.17 to 0.22,
    and against Poisson by 0.74 to 0.87, so a swapped prefactor, a missing
    unit-mean rescaling, or a beta mix-up cannot pass.
    """
    rng = np.random.default_rng(127)
    densities: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    edges = np.empty(0)
    for _ in range(120):
        unitary = _haar_unitary(rng, 128)
        if label == "coe":
            unitary = unitary.T @ unitary
        prepared = prepare_eigenphases(np.angle(np.linalg.eigvals(unitary)), symmetry_sector=label)
        histogram = spacing_distribution(prepared, bins=20, value_range=(0.0, 4.0))
        densities.append(np.asarray(histogram.values))
        edges = np.asarray(histogram.bin_edges)
    measured = np.mean(densities, axis=0)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def deviation(ensemble: Ensemble) -> float:
        reference = np.asarray(rmt_reference("spacing_distribution", ensemble, centers).values)
        return float(np.max(np.abs(measured - reference)))

    assert deviation(label) <= 0.075
    assert deviation(other) > 0.1
    assert deviation("poisson") > 0.5


def test_seeded_poisson_number_variance_has_expected_trend() -> None:
    rng = np.random.default_rng(1250)
    spacings = rng.exponential(size=200_000)
    phases = 2.0 * np.pi * np.cumsum(np.concatenate(([0.0], spacings[:-1]))) / np.sum(spacings)
    # A genuine Poisson spectrum this long contains gaps below the default 1e-10
    # degeneracy tolerance; they are real and harmless for number variance, so the
    # diagnostic is switched off rather than suppressed at the warning level.
    spectrum = unfold(
        prepare_eigenphases(phases, symmetry_sector="poisson", degeneracy_tolerance=0.0)
    )
    lengths = np.asarray([1.0, 2.0, 4.0, 8.0])
    # The estimator error is set by the origin count (std(n-L)^2 / sqrt(samples)),
    # not by the spectrum length, so 32768 origins are what buy the 3% bound.
    # Worst budget use over seeds 1240..1269 is 0.50; this seed uses 0.17.
    result = number_variance(spectrum, lengths, samples=32768)

    np.testing.assert_allclose(result.values, lengths, rtol=0.03, atol=0.03)


def test_seeded_cue_sff_follows_ramp_and_poisson_is_flat() -> None:
    rng = np.random.default_rng(1251)
    dimension = 24
    times = np.asarray([0.25, 0.5, 1.0])
    cue_samples = []
    poisson_samples = []
    for _ in range(80):
        gaussian = rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal(
            (dimension, dimension)
        )
        unitary, triangular = np.linalg.qr(gaussian)
        diagonal = np.diag(triangular)
        unitary *= (diagonal / np.abs(diagonal)).conj()
        cue = prepare_eigenphases(np.angle(np.linalg.eigvals(unitary)), symmetry_sector="cue")
        cue_samples.append(spectral_form_factor(cue, times).values)

        poisson = prepare_eigenphases(
            rng.uniform(0.0, 2.0 * np.pi, dimension), symmetry_sector="poisson"
        )
        poisson_samples.append(spectral_form_factor(poisson, times).values)

    cue_mean = np.mean(cue_samples, axis=0)
    poisson_mean = np.mean(poisson_samples, axis=0)

    np.testing.assert_allclose(cue_mean, np.minimum(times, 1.0), rtol=0.25, atol=0.12)
    np.testing.assert_allclose(poisson_mean, np.ones_like(times), rtol=0.25, atol=0.12)


def test_new_statistics_appear_in_the_candidate_list_and_reject_inapplicable_arguments() -> None:
    """Every rejected argument must name the statistic and say why."""
    with pytest.raises(ValidationError, match=r"'spectral_rigidity', 'gap_ratio_distribution'"):
        rmt_reference("rigidity", "goe", [1.0])  # type: ignore[arg-type]

    # Delta_3 here is the Mehta transform of the *bulk* number variance, so there is
    # no finite-N circular kernel to substitute; the gap-ratio surmise has no N in it.
    with pytest.raises(ValidationError, match="dimension does not apply to the spectral_rigidity"):
        rmt_reference("spectral_rigidity", "cue", [1.0], dimension=64)
    with pytest.raises(
        ValidationError, match="dimension does not apply to the gap_ratio_distribution"
    ):
        rmt_reference("gap_ratio_distribution", "gue", [0.5], dimension=64)
    with pytest.raises(ValidationError, match="folded does not apply to the number_variance"):
        rmt_reference("number_variance", "gue", [1.0], folded=True)
    with pytest.raises(ValidationError, match="folded must be a bool"):
        rmt_reference("gap_ratio_distribution", "gue", [0.5], folded=1)  # type: ignore[arg-type]

    rigidity = rmt_reference("spectral_rigidity", "coe", [2.0])
    assert rigidity.metadata.parameters["ensemble"] == "GOE"
    assert rigidity.metadata.parameters["folded"] is None
    assert rigidity.metadata.parameters["finite_size"] == "not applicable"
    np.testing.assert_array_equal(
        rigidity.values, rmt_reference("spectral_rigidity", "goe", [2.0]).values
    )
    np.testing.assert_array_equal(
        rmt_reference("spectral_rigidity", "cue", [2.0]).values,
        rmt_reference("spectral_rigidity", "gue", [2.0]).values,
    )


def test_gse_reference_rejects_dimension_on_every_statistic() -> None:
    """beta=4 has no finite-``N`` kernel here, so ``dimension`` is refused.

    ``"cue"`` is the one label that does something with ``dimension``: it swaps the
    bulk sine kernel for ``[sin(pi s)/(N sin(pi s/N))]^2``. There is no symplectic
    counterpart implemented, so accepting a dimension for ``"gse"`` would be
    accepting and ignoring it, which is exactly what the surrounding API refuses to
    do elsewhere. The statistic-level rejection still wins where both apply, so the
    message a caller sees names the more specific problem.
    """
    statistics: tuple[Statistic, ...] = ("spectral_form_factor", "number_variance")
    labels: tuple[Ensemble, ...] = ("gse", "cse")
    for statistic in statistics:
        for ensemble in labels:
            with pytest.raises(ValidationError, match="dimension does not apply to the gse"):
                rmt_reference(statistic, ensemble, [0.5], dimension=64)
    with pytest.raises(ValidationError, match="dimension does not apply to the spectral_rigidity"):
        rmt_reference("spectral_rigidity", "gse", [1.0], dimension=64)

    # Without a dimension the metadata records the bulk evaluation, as for GOE.
    curve = rmt_reference("number_variance", "gse", [1.0])
    assert curve.metadata.parameters["finite_size"] == "bulk"
    assert curve.metadata.parameters["dimension"] is None


def test_gse_form_factor_is_the_fourier_transform_of_its_own_cluster_function() -> None:
    """``K_GSE(tau)`` and ``Y_2^GSE`` must be one Fourier pair, not two guesses.

    The form factor is coded as a closed form and the cluster function as a
    different closed form, so nothing inside the library forces them to describe
    the same ensemble. This test closes that loop numerically through
    ``K(tau) = 1 - 2 int_0^inf Y_2(r) cos(2 pi tau r) dr``, evaluating the
    oscillatory integral against the *number-variance* path into ``Y_2`` -- the
    same private kernel the rigidity uses -- and finds agreement to 1e-9 at every
    ``tau`` below and above the plateau edge.

    The beta=4 curve behaves unlike the other two in three ways that the assertions
    pin: the plateau starts at ``tau=2`` rather than ``tau=1``, the curve
    *overshoots* the plateau before settling (``K(1.1) = 1.1832``), and the
    approach to the plateau from below is ``tau/2`` rather than ``tau`` or
    ``2*tau``, so at small ``tau`` GSE sits at half the GUE ramp.
    """
    from chaos_numerics.spectral.long_range import _cluster_function

    def transform(tau: float) -> float:
        integral = quad(
            lambda r: float(_cluster_function(np.asarray([r]), "gse", None)[0]),
            0.0,
            np.inf,
            weight="cos",
            wvar=2.0 * np.pi * tau,
            limit=2000,
            epsabs=1e-10,
        )[0]
        return 1.0 - 2.0 * float(integral)

    taus = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 1.9, 2.0, 3.0])
    values = np.asarray(rmt_reference("spectral_form_factor", "gse", taus).values)

    np.testing.assert_allclose(values, [transform(float(t)) for t in taus], atol=1e-9)
    # tau/2 - (tau/4) ln|1-tau| at the four points worth reading off by hand.
    assert float(values[1]) == pytest.approx(0.14298013, abs=1e-8)
    assert float(values[5]) == pytest.approx(1.18321090, abs=1e-8)
    np.testing.assert_array_equal(values[-2:], [1.0, 1.0])
    assert float(values[6]) > 1.0

    small = np.asarray([1e-3, 2e-3])
    gse = np.asarray(rmt_reference("spectral_form_factor", "gse", small).values)
    gue = np.asarray(rmt_reference("spectral_form_factor", "gue", small).values)
    np.testing.assert_allclose(gse / gue, 0.5, rtol=2e-3)


def test_gse_form_factor_refuses_the_heisenberg_time_divergence() -> None:
    """``K_GSE`` has a real logarithmic pole at ``tau=1``, not a finite value.

    Returning ``inf`` would fail ``SpectralCurve``'s finiteness check with a
    message about curve values rather than about the physics, and clipping it would
    invent a number. The curve is finite arbitrarily close by, which is what makes
    the rejection an inconvenience rather than a gap: it is worth 4.0 at
    ``tau = 1 +/- 1e-6`` and keeps climbing like ``-ln|1-tau|/4``.
    """
    with pytest.raises(ValidationError, match="diverges logarithmically at tau=1"):
        rmt_reference("spectral_form_factor", "gse", [0.5, 1.0, 1.5])
    with pytest.raises(ValidationError, match="diverges logarithmically at tau=1"):
        rmt_reference("spectral_form_factor", "cse", np.linspace(0.0, 2.0, 5))

    near = np.asarray(rmt_reference("spectral_form_factor", "gse", [1.0 - 1e-6, 1.0 + 1e-6]).values)
    np.testing.assert_allclose(near, 0.5 + np.log(1e6) / 4.0, rtol=1e-5)
    assert bool(np.all(np.isfinite(near)))
    # Only beta=4 has the pole; the other ensembles are ordinary at tau=1.
    assert float(rmt_reference("spectral_form_factor", "goe", [1.0]).values[0]) == pytest.approx(
        2.0 - np.log(3.0), rel=1e-12
    )


def test_gse_rigidity_slope_is_one_over_four_pi_squared() -> None:
    """The ``Delta_3`` logarithmic slope goes as ``1/(beta*pi^2)``, not ``1/(2*beta*pi^2)``.

    Regression rather than a remembered constant, and run for all three indices at
    once so that the *series* is what is being tested. Over ``L`` in
    ``[512, 8192]`` the plain fit gives 0.10126081, 0.05066059 and 0.02533787
    against ``1/(beta*pi^2)`` of 0.10132118, 0.05066059 and 0.02533030; refitting
    with a ``c/L`` term brings all three onto the analytic value to 1e-5 relative.
    ``1/(2*beta*pi^2)`` would be 0.01266515 for beta=4, a factor of two away and
    far outside the bound below.

    The slope *ratios* are the sharpest statement available, because they are
    exactly 2 and 4 with no ``1/L`` ambiguity, and they are what a wrong doubling
    convention inside ``Y_2^GSE`` -- ``sin(pi r)`` where ``sin(2 pi r)`` belongs --
    would break: that variant produces the beta=2 slope instead.
    """
    lengths = np.asarray([512.0, 1024.0, 2048.0, 4096.0, 8192.0])
    logarithm = np.log(lengths)
    slopes: dict[str, float] = {}
    for ensemble, index in (("goe", 1.0), ("gue", 2.0), ("gse", 4.0)):
        values = np.asarray(
            rmt_reference("spectral_rigidity", ensemble, lengths).values  # type: ignore[arg-type]
        )
        design = np.column_stack((logarithm, np.ones_like(lengths), 1.0 / lengths))
        corrected = np.linalg.lstsq(design, values, rcond=None)[0]
        slopes[ensemble] = float(np.polyfit(logarithm, values, 1)[0])

        assert slopes[ensemble] == pytest.approx(1.0 / (index * np.pi**2), rel=1e-3)
        assert float(corrected[0]) == pytest.approx(1.0 / (index * np.pi**2), rel=1e-5)

    assert slopes["goe"] / slopes["gue"] == pytest.approx(2.0, rel=2e-3)
    assert slopes["goe"] / slopes["gse"] == pytest.approx(4.0, rel=2e-3)

    # Reading the constant off with the slope pinned: like beta=1, and unlike
    # beta=2, it is still creeping up by about 1.5e-6 per octave at L=8192, so what
    # is asserted is the drift and its 1/L extrapolation, not a converged value.
    constant = np.asarray(rmt_reference("spectral_rigidity", "gse", lengths).values) - logarithm / (
        4.0 * np.pi**2
    )
    np.testing.assert_allclose(constant, 0.078318, atol=3e-5)
    assert bool(np.all(np.diff(constant) > 0.0))
    assert float(2.0 * constant[-1] - constant[-2]) == pytest.approx(0.0783197, abs=1e-6)

    # Below L=2 the three curves are not yet ordered by beta: all start from the
    # common Delta_3 -> L/15, and at L=0.5 beta=4 is the *largest* of the three.
    short = np.asarray([0.5])
    triple = [
        float(rmt_reference("spectral_rigidity", name, short).values[0])
        for name in ("goe", "gue", "gse")
    ]
    np.testing.assert_allclose(triple, [0.03242198, 0.03282211, 0.03319691], atol=1e-8)


def test_gse_cluster_function_reproduces_haar_symplectic_long_range_statistics() -> None:
    """End-to-end: Haar CSE spectra must land on the beta=4 curves and no others.

    This is the test that the ``Y_2^GSE`` implementation exists to pass, and the
    only one in the suite that generates its own ensemble, because nothing this
    library simulates is symplectic. 40 members at ``2N=128`` -- 64 distinct levels
    each after the Kramers doublets are removed -- give a mean number variance
    within 0.014 of the GSE reference over ``L`` in ``[1, 10]`` and a mean
    ``Delta_3`` within 0.003 (worst case over seeds 909, 55 and 7). The same
    measurements sit 0.185 and 0.037 away from the beta=2 curves and 0.51 and 0.097
    away from beta=1, so the bounds below are sampling noise on one side and a
    thirteen-fold margin on the other.
    """
    rng = np.random.default_rng(909)
    lengths = np.asarray([1.0, 2.0, 5.0, 10.0])
    variances = []
    rigidities = []
    for _ in range(40):
        _, distinct = _haar_symplectic_eigenphases(rng, 64)
        prepared = prepare_eigenphases(distinct, symmetry_sector="cse")
        with coarse_arcs_expected():
            variances.append(number_variance(prepared, lengths, samples=1024).values)
        with coarse_arcs_expected():
            rigidities.append(spectral_rigidity(prepared, lengths, samples=1024).values)
    measured_variance = np.mean(variances, axis=0)
    measured_rigidity = np.mean(rigidities, axis=0)

    def distance(statistic: Statistic, ensemble: Ensemble, measured: object) -> float:
        reference = np.asarray(rmt_reference(statistic, ensemble, lengths).values)
        return float(np.max(np.abs(np.asarray(measured) - reference)))

    assert distance("number_variance", "gse", measured_variance) <= 0.03
    assert distance("number_variance", "gue", measured_variance) > 0.15
    assert distance("number_variance", "goe", measured_variance) > 0.4

    assert distance("spectral_rigidity", "gse", measured_rigidity) <= 0.008
    assert distance("spectral_rigidity", "gue", measured_rigidity) > 0.03
    assert distance("spectral_rigidity", "goe", measured_rigidity) > 0.08


def test_haar_symplectic_spacings_track_the_beta_four_surmise() -> None:
    """The measured ``P(s)`` of a CSE spectrum must find the beta=4 surmise.

    60 members at ``2N=128`` give 3840 spacings after the doublets are dropped, so
    the counting noise on a width-0.2 density bin runs to about 0.068 at the peak
    (worst 0.0679 over seeds 127, 3 and 91). The distance to the beta=2 surmise is
    0.28 to 0.32 and to beta=1 is 0.47 to 0.49, which is what makes the comparison
    discriminating: the ``s^4`` repulsion pushes the whole distribution to the
    right, and the peak of the beta=4 surmise sits at ``s = (3/4) sqrt(pi/2)``
    where the beta=2 one sits at ``sqrt(pi)/2``.
    """
    rng = np.random.default_rng(127)
    densities = []
    edges = np.empty(0)
    for _ in range(60):
        _, distinct = _haar_symplectic_eigenphases(rng, 64)
        histogram = spacing_distribution(
            prepare_eigenphases(distinct, symmetry_sector="cse"), bins=20, value_range=(0.0, 4.0)
        )
        densities.append(np.asarray(histogram.values))
        edges = np.asarray(histogram.bin_edges)
    measured = np.mean(densities, axis=0)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def deviation(ensemble: Ensemble) -> float:
        reference = np.asarray(rmt_reference("spacing_distribution", ensemble, centers).values)
        return float(np.max(np.abs(measured - reference)))

    assert deviation("gse") <= 0.10
    assert deviation("gue") > 0.2
    assert deviation("goe") > 0.35
    assert deviation("poisson") > 0.7


@pytest.mark.parametrize(
    "operation",
    [
        lambda: rmt_reference("bad", "gue", [1.0]),  # type: ignore[arg-type]
        lambda: rmt_reference("number_variance", "bad", [1.0]),  # type: ignore[arg-type]
        lambda: rmt_reference("spectral_rigidity", "gue", [-1.0]),
        lambda: number_variance(prepare_eigenphases([0.0, 1.0], symmetry_sector="all"), [3.0]),
        lambda: spectral_rigidity(prepare_eigenphases([0.0, 1.0], symmetry_sector="all"), [3.0]),
        lambda: spectral_rigidity(prepare_eigenphases([], symmetry_sector="all"), [1.0]),
        lambda: spectral_rigidity([0.0, 1.0], [1.0]),  # type: ignore[arg-type]
        lambda: spectral_rigidity(
            prepare_eigenphases([0.0, 1.0], symmetry_sector="all"), [1.0], samples=0
        ),
        lambda: spectral_form_factor(
            prepare_eigenphases([0.0, 1.0], symmetry_sector="all"), [1.0], window="hann"
        ),
    ],
)
def test_long_range_parameter_validation(operation: object) -> None:
    with pytest.raises(ValidationError):
        operation()  # type: ignore[operator]


def test_spectral_rigidity_accepts_an_empty_spectrum_with_no_lengths() -> None:
    with pytest.warns(NumericalWarning, match="symmetry sector is unknown"):
        empty = prepare_eigenphases([], symmetry_sector=None)

    curve = spectral_rigidity(empty, [])

    assert curve.values.size == 0
    assert curve.metadata.parameters["level_count"] == 0
    assert curve.metadata.parameters["effective_samples"] == ()


def _convergence_metadata() -> ExperimentMetadata:
    """Metadata carrying a convergence *history*, which is what exposed the asymmetry."""
    return ExperimentMetadata(
        parameters={"statistic": "guard"},
        precision="float64",
        convergence=ConvergenceInfo(
            converged=True,
            iterations=3,
            residual=1e-14,
            tolerance=1e-12,
            history=np.asarray([1.0, 0.1, 1e-14]),
            reason="converged",
        ),
    )


def _spectral_containers_with_history() -> list[tuple[str, object]]:
    metadata = _convergence_metadata()
    prepared = prepare_eigenphases([0.1, 1.0, 2.0, 4.0], symmetry_sector="all")
    unfolded = unfold(prepared)
    histogram = spacing_distribution(prepared)
    return [
        (
            "PreparedEigenphases",
            PreparedEigenphases(
                prepared.raw_phases, prepared.phases, prepared.spacings, "all", metadata
            ),
        ),
        (
            "UnfoldedSpectrum",
            UnfoldedSpectrum(
                unfolded.raw_phases,
                unfolded.phases,
                unfolded.values,
                unfolded.spacings,
                "all",
                "mean",
                metadata,
            ),
        ),
        (
            "SpacingDistributionResult",
            SpacingDistributionResult(
                histogram.values, histogram.bin_edges, histogram.sample_count, metadata
            ),
        ),
        (
            "SpectralCurve",
            SpectralCurve(
                np.asarray([1.0, 2.0]),
                np.asarray([0.5, 0.6]),
                np.asarray([0.01, 0.02]),
                np.asarray([1e-4, 4e-4]),
                metadata,
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("name", "container"),
    _spectral_containers_with_history(),
    ids=[name for name, _ in _spectral_containers_with_history()],
)
def test_spectral_containers_describe_every_array_they_persist(
    name: str, container: object
) -> None:
    """``array_payload()`` keys must be a subset of the descriptor's ``arrays`` keys.

    ``add_convergence_history`` adds ``"convergence_history"`` to the array payload
    whenever ``metadata.convergence.history`` is set, so the JSON descriptor has to
    list it too. All four spectral containers used to add it on the payload side
    only, and the storage layer matches the two sets when it computes array digests:
    saving a curve whose metadata carried a convergence history therefore failed with
    a bare ``KeyError`` instead of a diagnosable error. Nothing in the suite caught
    it because the round-trip tests build their examples from metadata with no
    history at all -- which is why this test constructs one deliberately rather than
    using whatever a statistic happens to return.
    """
    payload = container.array_payload()  # type: ignore[attr-defined]
    descriptor = container.metadata_payload()  # type: ignore[attr-defined]
    described = descriptor["arrays"]

    assert isinstance(described, dict)
    assert "convergence_history" in payload, name
    assert set(payload) <= set(described), (name, set(payload) - set(described))
    for key, array in payload.items():
        assert described[key] == {"shape": list(array.shape), "dtype": array.dtype.name}


def test_batch_means_variance_is_the_unbiased_estimate_over_arcs() -> None:
    """The **default** error path also needs ``ddof=1`` pinned, not just bootstrap.

    Two white-box tests in this file pin ``ddof=1`` for the two bootstrap paths, but
    the estimator that ``number_variance`` and ``spectral_rigidity`` report out of
    the box is the batch-means one, and the only thing constraining it was the
    0.5-2.0 calibration band. That band cannot see a ``ddof`` flip: with ``b`` arcs
    the variance changes by ``(b-1)/b``, which is 0.96-0.99 at the arc counts these
    tests use. So the flip survived.

    Forcing exactly two arcs makes the identity trivial and the difference maximal:
    ``var([m1, m2], ddof=1) = (m1-m2)**2/2``, so the reported standard error is
    ``|m1-m2|/2``, whereas ``ddof=0`` would give ``|m1-m2|/(2*sqrt(2))``.
    """
    dimension = 32
    samples = 64
    length = 12.0
    rng = np.random.default_rng(5)
    prepared = prepare_eigenphases(
        np.sort(rng.uniform(0.0, 2.0 * np.pi, dimension)),
        symmetry_sector="uniform",
        degeneracy_tolerance=0.0,
    )

    with coarse_arcs_expected():
        result = number_variance(prepared, [length], samples=samples)

    # Same reconstruction as the bootstrap white-box test above.
    levels = (np.asarray(prepared.phases) - prepared.phases[0]) / (2.0 * np.pi / dimension)
    canonical = np.sort(np.mod(levels, dimension))
    doubled = np.concatenate((canonical, canonical + dimension))
    origins = (np.arange(samples, dtype=np.float64) + 0.5) * dimension / samples
    starts = np.searchsorted(canonical, origins, side="left")
    ends = np.searchsorted(doubled, origins + length, side="left")
    squared = (ends - starts - length) ** 2.0
    arcs = min(samples, int(dimension // length))
    assert arcs == 2, "the identity below only holds for two arcs"
    first, second = (float(part.mean()) for part in np.array_split(squared, arcs))

    assert result.uncertainty is not None
    expected = abs(first - second) / 2.0
    assert float(result.uncertainty[0]) == pytest.approx(expected, rel=1e-12)
    # State what the rejected alternative would have produced, so the margin is visible.
    assert float(result.uncertainty[0]) != pytest.approx(expected / np.sqrt(2.0), rel=1e-6)


def test_hann_connected_background_removes_the_window_sidelobes() -> None:
    """A perfectly rigid spectrum has no connected fluctuations, in any window.

    The Hann background is ``0.5*N*sinc(N*tau) + 0.25*N*(sinc(N*tau+1) +
    sinc(N*tau-1))``. Every existing Hann assertion sits where the two sidelobe
    terms vanish exactly -- ``tau = 0`` and integer ``N*tau`` -- so dropping them
    left the suite green. An equally spaced comb is the sharpest probe available:
    its connected form factor must collapse to zero at generic ``tau``, and it only
    does so if the whole background, sidelobes included, is subtracted.

    Measured: with the sidelobes the residual is at most 7.4e-3; without them it
    ranges 0.32 to 10.4, three to four orders of magnitude larger.
    """
    dimension = 64
    rigid = unfold(
        prepare_eigenphases(
            np.arange(dimension) * 2.0 * np.pi / dimension,
            symmetry_sector="rigid",
            degeneracy_tolerance=0.0,
        ),
        method="mean",
    )
    # Deliberately away from integer ``N*tau``, where the sidelobes are largest.
    taus = np.asarray([0.3, 0.7, 1.3, 2.4]) / dimension

    connected = np.asarray(spectral_form_factor(rigid, taus, connected=True, window="hann").values)

    assert float(np.max(np.abs(connected))) <= 0.05
    # The flat window is already covered elsewhere; check it here too so that a
    # regression in the shared background helper cannot hide in one branch.
    flat = np.asarray(spectral_form_factor(rigid, taus, connected=True, window="none").values)
    assert float(np.max(np.abs(flat))) <= 0.05

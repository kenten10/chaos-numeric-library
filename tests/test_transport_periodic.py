from __future__ import annotations

import contextlib
import copy
import math
import pickle
import time

import numpy as np
import pytest
from numpy.typing import ArrayLike

from chaos_numerics.classical import (
    CatMap,
    LogisticMap,
    PeriodicOrbitResult,
    StandardMap,
    autocorrelation,
    find_periodic_orbits,
    iterate,
    local_diffusion_exponent,
    mean_square_displacement,
)
from chaos_numerics.core import (
    AnalysisResult,
    ConvergenceError,
    ConvergenceWarning,
    NumericalError,
    NumericalWarning,
    Trajectory,
    ValidationError,
)


def floats(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    return np.asarray(value, dtype=np.float64)


ENSEMBLE_SEMANTICS = "standard error of the mean over trajectories"
SINGLE_SEMANTICS = (
    "none: a single trajectory cannot calibrate the error on a time-origin-averaged curve"
)


def _plain(call: object, /, *args: object, **kwargs: object) -> AnalysisResult:
    """Call an estimator that is expected not to warn, matching ``single``'s shape."""
    assert callable(call)
    result = call(*args, **kwargs)
    assert isinstance(result, AnalysisResult)
    return result


def single(call: object, /, *args: object, **kwargs: object) -> AnalysisResult:
    """Run a transport estimator on one trajectory and capture its warning.

    ``autocorrelation`` and ``mean_square_displacement`` share one uncertainty
    convention: the error bar is the spread across independent trajectories, so a
    single trajectory yields ``uncertainty=None`` and says so with a
    ``NumericalWarning``. ``filterwarnings = ["error"]`` turns an uncaptured one
    into a failure, which is why every single-trajectory call in this file goes
    through this helper -- and why forgetting the warning cannot pass silently.
    """
    assert callable(call)
    with pytest.warns(NumericalWarning, match="received a single trajectory"):
        result = call(*args, **kwargs)
    assert isinstance(result, AnalysisResult)
    assert result.uncertainty is None
    return result


def test_autocorrelation_matches_direct_alternating_series() -> None:
    result = single(
        autocorrelation,
        floats([1.0, -1.0, 1.0, -1.0]),
        max_lag=3,
        demean=False,
    )

    np.testing.assert_allclose(result.values, floats([1.0, -1.0, 1.0, -1.0]))
    assert result.metadata.parameters["averaging"] == "all_time_origins"


def test_autocorrelation_observable_and_normalization_are_explicit() -> None:
    states = floats([[1.0, 9.0], [2.0, 8.0], [3.0, 7.0]])
    result = single(
        autocorrelation,
        states,
        observable=lambda values: values[:, 0],
        max_lag=1,
        demean=False,
        normalize=True,
    )

    assert result.values[0] == 1.0
    assert result.values[1] == pytest.approx(6.0 / 7.0)
    # A rejected normalization raises before the single-trajectory warning fires,
    # so the caller sees the actionable error and not a diagnostic about the error
    # bar of a result that is never returned.
    with pytest.raises(ValidationError, match="zero lag-zero"):
        autocorrelation(np.ones(4), normalize=True)


def test_msd_constant_velocity_matches_closed_form() -> None:
    times = np.arange(6, dtype=np.float64)
    positions = np.column_stack((2.0 * times, -times))
    result = single(mean_square_displacement, positions, max_lag=5)

    expected = 5.0 * np.arange(6, dtype=np.float64) ** 2
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-14)


def test_msd_periodic_unwrap_recovers_continuous_motion() -> None:
    wrapped = floats([0.8, 0.9, 0.0, 0.1, 0.2])
    result = single(
        mean_square_displacement,
        wrapped,
        max_lag=4,
        unwrap=True,
        periods=floats([1.0]),
    )

    expected = 0.01 * np.arange(5, dtype=np.float64) ** 2
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValidationError, match="periods are required"):
        mean_square_displacement(wrapped, unwrap=True)


def test_msd_unwrap_rejects_steps_reaching_half_the_period() -> None:
    # One standard-map kick displaces p by up to K / (2 pi), so K = 8 aliases the
    # minimum-image branch and used to report a silently far too small MSD.
    trajectory = iterate(StandardMap(kick_strength=8.0), floats([0.1, 0.2]), steps=400)

    with pytest.raises(ValidationError) as failure:
        mean_square_displacement(trajectory, unwrap=True, periods=floats([1.0, 1.0]))

    message = str(failure.value)
    assert "coordinate 0" in message
    assert "0.5" in message
    assert "unwrap=False" in message


def test_msd_unwrap_matches_directly_integrated_standard_map() -> None:
    # For K = 0.5 the momentum never leaves the interior of [0, 1), so the true
    # unwrapped orbit follows from the map itself: dq = p' and dp = p' - p.
    trajectory = iterate(StandardMap(kick_strength=0.5), floats([0.1, 0.2]), steps=200)
    momenta = trajectory.states[:, 1]
    assert float(np.min(momenta)) > 0.05
    assert float(np.max(momenta)) < 0.45
    reference = np.column_stack(
        (
            trajectory.states[0, 0] + np.concatenate(([0.0], np.cumsum(momenta[1:]))),
            momenta,
        )
    )

    result = single(mean_square_displacement, trajectory, unwrap=True, periods=floats([1.0, 1.0]))
    expected = single(mean_square_displacement, reference)

    assert float(result.values[-1]) > 1000.0
    np.testing.assert_allclose(result.values, expected.values, rtol=1e-9, atol=1e-9)


def test_msd_scalar_periods_broadcast_to_every_coordinate() -> None:
    trajectory = iterate(StandardMap(kick_strength=0.5), floats([0.1, 0.2]), steps=200)

    scalar = single(mean_square_displacement, trajectory, unwrap=True, periods=1.0)
    listed = single(mean_square_displacement, trajectory, unwrap=True, periods=floats([1.0, 1.0]))

    np.testing.assert_array_equal(scalar.values, listed.values)
    assert scalar.metadata.parameters["periods"] == listed.metadata.parameters["periods"]
    with pytest.raises(ValidationError, match="periods must be positive"):
        mean_square_displacement(trajectory, unwrap=True, periods=0.0)


def parameter(result: AnalysisResult, key: str) -> object:
    return result.metadata.parameters[key]


def test_local_diffusion_exponent_recovers_power_law_and_fit_metadata() -> None:
    times = np.arange(20, dtype=np.float64)
    msd = np.zeros_like(times)
    msd[1:] = 3.0 * times[1:] ** 1.5
    with pytest.warns(NumericalWarning, match="single MSD curve"):
        result = local_diffusion_exponent(msd, fit_start=1, fit_stop=20, times=times)

    assert result.values[0] == pytest.approx(1.5, abs=1e-12)
    assert result.residuals is not None
    assert result.residuals[0] <= 1e-12
    assert parameter(result, "fit_start") == 1
    assert parameter(result, "fit_stop") == 20
    assert parameter(result, "curve_count") == 1
    # One smooth curve carries no information about the sampling error of the
    # exponent, so none is reported instead of an OLS error 30x too small.
    assert result.uncertainty is None
    semantics = parameter(result, "error_semantics")
    assert isinstance(semantics, str)
    assert semantics.startswith("none")
    with pytest.raises(ValidationError, match="at least three positive"):
        local_diffusion_exponent(floats([0.0, 1.0, 0.0]), fit_start=0, fit_stop=3)


def test_diffusion_exponent_uncertainty_is_the_spread_of_the_per_curve_fits() -> None:
    # Exact power laws with slopes 1, 1.5 and 2 pin the estimator to a closed form:
    # std(slopes, ddof=1) / sqrt(3) = 0.5 / sqrt(3). Nothing here is approximate, so
    # dropping the ddof, dropping the sqrt(3), or substituting a residual-based
    # standard error all change the number.
    times = np.arange(1.0, 33.0, dtype=np.float64)
    curves = np.stack([times**exponent for exponent in (1.0, 1.5, 2.0)])
    result = local_diffusion_exponent(curves, fit_start=0, fit_stop=32, times=times)

    assert result.uncertainty is not None
    assert float(result.uncertainty[0]) == pytest.approx(0.5 / math.sqrt(3.0), abs=1e-14)
    assert parameter(result, "curve_count") == 3
    assert parameter(result, "error_semantics") == (
        "standard error of the mean over per-curve fits"
    )
    # The reported exponent is the fit of the ensemble-mean curve, not the mean of
    # the three slopes, so it does not sit at 1.5.
    assert 1.0 < float(result.values[0]) < 2.0


def test_identical_diffusion_curves_give_exactly_zero_uncertainty() -> None:
    times = np.arange(1.0, 33.0, dtype=np.float64)
    curves = np.repeat((3.0 * times**1.5)[None, :], 5, axis=0)
    result = local_diffusion_exponent(curves, fit_start=0, fit_stop=32, times=times)

    assert result.values[0] == pytest.approx(1.5, abs=1e-12)
    assert result.uncertainty is not None
    assert float(result.uncertainty[0]) == 0.0
    assert result.residuals is not None
    assert float(result.residuals[0]) <= 1e-12


def calibration(
    positions: object,
    *,
    realizations: int,
    max_lag: int,
    fit_start: int,
    truth: float,
) -> tuple[float, float]:
    """Return ``(true_std / reported_error, fraction beyond two sigma)``."""
    assert callable(positions)
    exponents: list[float] = []
    errors: list[float] = []
    for _ in range(realizations):
        curves = mean_square_displacement(positions(), max_lag=max_lag, per_trajectory=True)
        fit = local_diffusion_exponent(curves, fit_start=fit_start, fit_stop=max_lag)
        assert fit.uncertainty is not None
        exponents.append(float(fit.values[0]))
        errors.append(float(fit.uncertainty[0]))
    sampled = np.asarray(exponents)
    reported = np.asarray(errors)
    ratio = float(np.std(sampled, ddof=1) / np.mean(reported))
    beyond = float(np.mean(np.abs((sampled - truth) / reported) > 2.0))
    return ratio, beyond


def test_diffusion_exponent_error_is_calibrated_on_ordinary_random_walks() -> None:
    # The reported error must be the actual spread of the estimator, not a nominal
    # one: the OLS standard error this replaced was 30x too small, which put 93% of
    # realizations more than two of its sigma away from the true alpha = 1.
    rng = np.random.default_rng(4242)

    def walks() -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.cumsum(rng.standard_normal((8, 600, 1)), axis=1)

    ratio, beyond = calibration(walks, realizations=60, max_lag=60, fit_start=4, truth=1.0)

    assert 0.5 <= ratio <= 2.0, ratio
    assert beyond <= 0.2, beyond


def fractional_brownian_factor(hurst: float, steps: int) -> np.ndarray:
    """Cholesky factor of the fractional-Brownian covariance, exact for any lag."""
    times = np.arange(1.0, steps + 1.0, dtype=np.float64)
    rows = times[:, None]
    columns = times[None, :]
    covariance = 0.5 * (
        rows ** (2.0 * hurst) + columns ** (2.0 * hurst) - np.abs(rows - columns) ** (2.0 * hurst)
    )
    return np.linalg.cholesky(covariance + 1e-10 * np.eye(steps))


def test_diffusion_exponent_error_is_calibrated_on_a_superdiffusive_exponent() -> None:
    # Fractional Brownian motion with H = 0.75 has MSD(t) = t**1.5 exactly, so the
    # calibration is checked against a second known exponent and not only alpha = 1.
    factor = fractional_brownian_factor(0.75, 400)
    rng = np.random.default_rng(9091)

    def paths() -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return (factor @ rng.standard_normal((400, 16))).T[:, :, None]

    ratio, beyond = calibration(paths, realizations=60, max_lag=80, fit_start=5, truth=1.5)

    assert 0.5 <= ratio <= 2.0, ratio
    assert beyond <= 0.2, beyond


def random_walks(
    rng: np.random.Generator, *, batch: int, steps: int
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Return ``(batch, steps + 1, 1)`` unit random walks starting at the origin."""
    increments = rng.standard_normal((batch, steps))
    walk = np.concatenate((np.zeros((batch, 1)), np.cumsum(increments, axis=-1)), axis=-1)
    return floats(walk[:, :, None])


CALIBRATION_LAGS = (1, 5, 20, 100)


def test_msd_batch_standard_error_is_calibrated_on_unit_random_walks() -> None:
    # A unit random walk has <|x(t + lag) - x(t)|**2> = lag exactly, so the error
    # bar is checked against a known truth and not against another estimate of it.
    # This is the section 4.3 criterion of docs/design/numerical-standards.md, the
    # one local_diffusion_exponent already has to meet, applied to the curve: the
    # true spread over the reported error must land in [0.5, 2] and at most 20% of
    # realizations may sit more than two reported sigma from the truth.
    rng = np.random.default_rng(20260808)
    realizations = 120
    sampled = np.empty((realizations, 101), dtype=np.float64)
    reported = np.empty((realizations, 101), dtype=np.float64)
    for index in range(realizations):
        result = mean_square_displacement(random_walks(rng, batch=16, steps=400), max_lag=100)
        assert result.uncertainty is not None
        sampled[index] = result.values
        reported[index] = result.uncertainty

    for lag in CALIBRATION_LAGS:
        ratio = float(np.std(sampled[:, lag], ddof=1) / np.mean(reported[:, lag]))
        beyond = float(np.mean(np.abs((sampled[:, lag] - lag) / reported[:, lag]) > 2.0))
        assert 0.5 <= ratio <= 2.0, (lag, ratio)
        assert beyond <= 0.2, (lag, beyond)


def test_the_retired_time_origin_standard_error_fails_that_criterion() -> None:
    # mean_square_displacement used to report this estimator for a single
    # trajectory. Recomputing it here, on the same known truth, is what keeps it
    # retired: reinstating it would have to survive these two assertions, and it
    # cannot, because the time origins of one trajectory are strongly correlated
    # and the scatter over them therefore measures far less than the real spread.
    rng = np.random.default_rng(4242)
    realizations = 120
    steps, lag = 400, 100
    curve = np.empty(realizations, dtype=np.float64)
    reported = np.empty(realizations, dtype=np.float64)
    for index in range(realizations):
        walk = random_walks(rng, batch=1, steps=steps)[0, :, 0]
        squared = (walk[lag:] - walk[: walk.size - lag]) ** 2
        curve[index] = float(np.mean(squared))
        reported[index] = float(np.std(squared, ddof=1) / np.sqrt(squared.size))

    ratio = float(np.std(curve, ddof=1) / np.mean(reported))
    beyond = float(np.mean(np.abs((curve - lag) / reported) > 2.0))
    assert ratio > 4.0, ratio
    assert beyond > 0.5, beyond


def test_diffusion_exponent_rejects_shapes_that_are_not_curves() -> None:
    with pytest.raises(ValidationError, match="one-dimensional"):
        local_diffusion_exponent(np.zeros((2, 3, 4)), fit_start=0, fit_stop=3)
    with pytest.raises(ValidationError, match="at least one curve"):
        local_diffusion_exponent(np.zeros((0, 8)), fit_start=0, fit_stop=8)
    with pytest.raises(ValidationError, match=r"times must have shape \(8,\)"):
        local_diffusion_exponent(np.ones((3, 8)), fit_start=0, fit_stop=8, times=np.ones(7))


def test_cat_fixed_point_residual_monodromy_and_multipliers() -> None:
    model = CatMap()
    result = find_periodic_orbits(
        model,
        floats([[0.01, 0.02], [0.99, 0.98]]),
        period=1,
        tolerance=1e-12,
    )

    assert isinstance(result, PeriodicOrbitResult)
    assert result.count == 1
    assert result.residuals[0] <= 1e-12
    np.testing.assert_allclose(result.monodromy_matrices[0], floats(model.matrix))
    expected = np.sort_complex(np.linalg.eigvals(floats(model.matrix)))
    np.testing.assert_allclose(np.sort_complex(result.stability_multipliers[0]), expected)
    assert not result.points.flags.writeable
    assert "count=1" in repr(result)


def test_cat_period_two_phases_deduplicate_and_monodromy_is_ordered_product() -> None:
    model = CatMap()
    result = find_periodic_orbits(
        model,
        floats([[0.8, 0.6], [0.2, 0.4]]),
        period=2,
        tolerance=1e-12,
    )
    matrix = floats(model.matrix)

    assert result.count == 1
    np.testing.assert_allclose(result.monodromy_matrices[0], matrix @ matrix)
    assert result.residuals[0] <= 1e-12


def torus_distance(first: object, second: object) -> float:
    difference = np.mod(floats(first) - floats(second) + 0.5, 1.0) - 0.5
    return float(np.linalg.norm(difference))


def distinct_orbit_points(model: CatMap, result: PeriodicOrbitResult) -> list[np.ndarray]:
    """Expand every representative into its full orbit and drop torus duplicates."""
    collected: list[np.ndarray] = []
    for representative in result.points:
        current = floats(representative)
        for _ in range(result.period):
            if all(torus_distance(current, seen) > 1e-8 for seen in collected):
                collected.append(current)
            current = floats(model.step(current))
    return collected


def orbit_search_grid() -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    axis = np.linspace(0.02, 0.98, 9)
    return floats(np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2))


@pytest.mark.parametrize("period", [1, 2, 3])
def test_cat_map_periodic_point_count_matches_trace_formula(period: int) -> None:
    # A hyperbolic torus automorphism has |det(M**p - I)| = Tr(M**p) - 2 points of
    # period dividing p, an independent count the search must reproduce exactly.
    model = CatMap()
    matrix = floats(model.matrix)
    expected = round(float(np.trace(np.linalg.matrix_power(matrix, period)))) - 2

    with pytest.warns(ConvergenceWarning) if period == 1 else contextlib.nullcontext():
        result = find_periodic_orbits(model, orbit_search_grid(), period=period, tolerance=1e-12)

    assert len(distinct_orbit_points(model, result)) == expected


def test_cat_map_period_two_multipliers_match_squared_golden_eigenvalues() -> None:
    model = CatMap()
    result = find_periodic_orbits(model, orbit_search_grid(), period=2, tolerance=1e-12)
    expected = np.sort(
        floats([(7.0 - 3.0 * math.sqrt(5.0)) / 2.0, (7.0 + 3.0 * math.sqrt(5.0)) / 2.0])
    )

    assert result.count == 3
    for multipliers in result.stability_multipliers:
        assert float(np.max(np.abs(multipliers.imag))) <= 1e-14
        np.testing.assert_allclose(np.sort(multipliers.real), expected, rtol=1e-12, atol=0.0)


def test_periodic_orbit_points_respect_half_open_bounds() -> None:
    # A guess a rounding error below zero used to be reported at the excluded
    # upper endpoint 1.0 instead of inside the documented [0, 1) bounds.
    model = CatMap()
    result = find_periodic_orbits(
        model,
        floats([[1e-17, 1e-17], [0.8, 0.6], [0.2, 0.4]]),
        period=2,
        tolerance=1e-12,
    )

    assert bool(np.all((result.points >= 0.0) & (result.points < 1.0)))
    origins = [
        point for point in result.points if torus_distance(point, floats([0.0, 0.0])) <= 1e-12
    ]
    assert len(origins) == 1


class TranslationMap:
    @property
    def state_dim(self) -> int:
        return 1

    @property
    def is_periodic(self) -> tuple[bool]:
        return (True,)

    @property
    def bounds(self) -> tuple[tuple[float, float]]:
        return ((0.0, 1.0),)

    def step(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.mod(np.asarray(state, dtype=np.float64) + 0.25, 1.0)

    def jacobian(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        del state
        return np.ones((1, 1), dtype=np.float64)


def test_periodic_search_warns_or_raises_when_no_root_converges() -> None:
    with pytest.warns(ConvergenceWarning, match="no periodic orbit"):
        result = find_periodic_orbits(TranslationMap(), floats([0.1]), period=1, max_iterations=5)
    assert result.count == 0
    assert result.metadata.convergence is not None
    assert not result.metadata.convergence.converged

    with pytest.raises(ConvergenceError, match="no periodic orbit"):
        find_periodic_orbits(
            TranslationMap(), floats([0.1]), period=1, max_iterations=5, strict=True
        )


def momentum(states: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray:
    return states[..., 1]


def ensemble(count: int, *, steps: int, kick: float = 2.0) -> Trajectory:
    starts = np.column_stack(
        (
            np.linspace(0.01, 0.99, count, dtype=np.float64),
            np.linspace(0.02, 0.98, count, dtype=np.float64),
        )
    )
    return iterate(StandardMap(kick_strength=kick), starts, steps=steps)


def test_msd_accepts_a_batched_trajectory_and_records_the_batch_size() -> None:
    trajectory = ensemble(64, steps=2000)
    assert trajectory.shape == (64, 2001, 2)

    result = mean_square_displacement(trajectory)

    assert result.values.shape == (2001,)
    assert result.metadata.parameters["batch_size"] == 64
    assert parameter(result, "error_semantics") == ENSEMBLE_SEMANTICS
    assert result.uncertainty is not None
    assert float(result.values[0]) == 0.0

    # The ensemble is already collapsed into a one-dimensional curve, so the
    # diffusion fit consumes a batched result without any change of its own -- and
    # says so, because collapsing first destroys the between-trajectory spread the
    # error bar would need.
    with pytest.warns(NumericalWarning, match="single MSD curve"):
        exponent = local_diffusion_exponent(result, fit_start=1, fit_stop=200)
    assert exponent.values.shape == (1,)
    assert np.isfinite(float(exponent.values[0]))
    assert exponent.uncertainty is None


def test_msd_per_trajectory_keeps_the_individual_curves_for_the_diffusion_fit() -> None:
    trajectory = ensemble(6, steps=200)
    curves = mean_square_displacement(trajectory, max_lag=50, per_trajectory=True)
    collapsed = mean_square_displacement(trajectory, max_lag=50)
    singles = np.stack(
        [
            single(mean_square_displacement, trajectory.states[index], max_lag=50).values
            for index in range(6)
        ]
    )

    assert curves.values.shape == (6, 51)
    assert curves.uncertainty is None
    assert parameter(curves, "per_trajectory") is True
    # per_trajectory=True is the one None that is not a missing error bar, so it
    # gets its own semantics string and no warning.
    assert parameter(curves, "error_semantics") == (
        "none: per_trajectory=True returns the individual curves, whose spread is the error"
    )
    np.testing.assert_array_equal(curves.values, singles)
    np.testing.assert_array_equal(np.mean(curves.values, axis=0), collapsed.values)
    # The exponent is unchanged by keeping the curves apart; only the error appears.
    fit = local_diffusion_exponent(curves, fit_start=1, fit_stop=50)
    with pytest.warns(NumericalWarning, match="single MSD curve"):
        reference = local_diffusion_exponent(collapsed, fit_start=1, fit_stop=50)
    np.testing.assert_array_equal(fit.values, reference.values)
    assert fit.uncertainty is not None
    assert float(fit.uncertainty[0]) > 0.0
    with pytest.raises(ValidationError, match="per_trajectory must be a bool"):
        mean_square_displacement(trajectory, per_trajectory=1)  # type: ignore[arg-type]


def test_batch_of_one_reproduces_the_unbatched_result() -> None:
    states = iterate(StandardMap(1.5), floats([0.13, 0.27]), steps=300).states
    flat = single(mean_square_displacement, states, max_lag=80)
    batched = single(mean_square_displacement, states[None], max_lag=80)

    np.testing.assert_allclose(batched.values, flat.values, rtol=0.0, atol=1e-15)
    # A batch axis of length one is still one trajectory, so wrapping the input in
    # a leading axis must not conjure an error bar out of nothing.
    assert parameter(batched, "error_semantics") == SINGLE_SEMANTICS
    assert parameter(flat, "error_semantics") == SINGLE_SEMANTICS

    flat_acf = single(autocorrelation, states, observable=momentum, max_lag=40)
    batched_acf = single(autocorrelation, states[None], observable=momentum, max_lag=40)
    np.testing.assert_allclose(batched_acf.values, flat_acf.values, rtol=0.0, atol=1e-15)
    assert parameter(batched_acf, "error_semantics") == SINGLE_SEMANTICS


def test_batch_average_equals_the_mean_of_the_individual_curves() -> None:
    trajectory = ensemble(4, steps=200)
    states = trajectory.states
    result = mean_square_displacement(trajectory, max_lag=50)
    singles = np.stack(
        [single(mean_square_displacement, states[index], max_lag=50).values for index in range(4)]
    )

    np.testing.assert_allclose(result.values, np.mean(singles, axis=0), rtol=0.0, atol=1e-15)
    assert result.uncertainty is not None
    np.testing.assert_allclose(
        result.uncertainty,
        np.std(singles, axis=0, ddof=1) / 2.0,
        rtol=0.0,
        atol=1e-15,
    )

    correlation = autocorrelation(trajectory, observable=momentum, max_lag=30)
    single_acf = np.stack(
        [
            single(autocorrelation, states[index], observable=momentum, max_lag=30).values
            for index in range(4)
        ]
    )
    np.testing.assert_allclose(
        correlation.values, np.mean(single_acf, axis=0), rtol=0.0, atol=1e-15
    )


def test_both_transport_estimators_share_one_uncertainty_convention() -> None:
    # The two used to disagree: autocorrelation returned None for one trajectory
    # while mean_square_displacement returned a time-origin standard error that was
    # measured to be up to 21x too small. They now answer the same question the
    # same way, and this pins that so the split cannot come back after v0.1.
    states = ensemble(5, steps=120).states
    batched = (
        mean_square_displacement(states, max_lag=30),
        autocorrelation(states, observable=momentum, max_lag=30),
    )
    per_curve = (
        np.stack(
            [
                single(mean_square_displacement, states[index], max_lag=30).values
                for index in range(5)
            ]
        ),
        np.stack(
            [
                single(autocorrelation, states[index], observable=momentum, max_lag=30).values
                for index in range(5)
            ]
        ),
    )
    for result, curves in zip(batched, per_curve, strict=True):
        assert result.uncertainty is not None
        assert parameter(result, "error_semantics") == ENSEMBLE_SEMANTICS
        np.testing.assert_allclose(
            result.uncertainty,
            np.std(curves, axis=0, ddof=1) / math.sqrt(5.0),
            rtol=0.0,
            atol=1e-15,
        )

    for lone in (
        single(mean_square_displacement, states[0], max_lag=30),
        single(autocorrelation, states[0], observable=momentum, max_lag=30),
    ):
        assert lone.uncertainty is None
        assert parameter(lone, "error_semantics") == SINGLE_SEMANTICS


def test_three_dimensional_batch_axes_are_flattened_into_one_ensemble() -> None:
    flat = ensemble(6, steps=120).states
    nested = flat.reshape(2, 3, *flat.shape[-2:])

    grouped = mean_square_displacement(nested, max_lag=40)
    reference = mean_square_displacement(flat, max_lag=40)

    assert grouped.metadata.parameters["batch_size"] == 6
    np.testing.assert_array_equal(grouped.values, reference.values)


def test_unwrap_guard_fires_when_any_single_trajectory_aliases() -> None:
    safe = iterate(StandardMap(kick_strength=0.5), floats([0.1, 0.2]), steps=150).states
    aliased = iterate(StandardMap(kick_strength=8.0), floats([0.1, 0.2]), steps=150).states

    single(mean_square_displacement, safe[None], unwrap=True, periods=1.0, max_lag=50)
    with pytest.raises(ValidationError, match="unwrap cannot resolve coordinate"):
        mean_square_displacement(np.stack((safe, aliased)), unwrap=True, periods=1.0, max_lag=50)


def test_autocorrelation_of_multi_coordinate_trajectory_demands_an_observable() -> None:
    trajectory = ensemble(3, steps=50)

    with pytest.raises(ValidationError, match="needs a scalar time series"):
        autocorrelation(trajectory)
    scalar = iterate(LogisticMap(3.9), floats([0.3]), steps=50)
    assert single(autocorrelation, scalar).values.shape == (51,)


def test_batched_axes_must_not_be_empty() -> None:
    with pytest.raises(ValidationError, match="empty batch axis"):
        mean_square_displacement(np.zeros((0, 5, 2)))
    with pytest.raises(ValidationError, match="empty batch axis"):
        autocorrelation(np.zeros((0, 5)))


@pytest.mark.parametrize("count", [400, 401])
@pytest.mark.parametrize("max_lag", [None, 37])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("batch", [1, 5])
def test_fft_autocorrelation_matches_the_direct_reference(
    count: int, max_lag: int | None, demean: bool, normalize: bool, batch: int
) -> None:
    states = ensemble(batch, steps=count - 1).states
    data = states[0] if batch == 1 else states
    common = {"observable": momentum, "max_lag": max_lag, "demean": demean, "normalize": normalize}
    run = single if batch == 1 else _plain
    fast = run(autocorrelation, data, method="fft", **common)
    slow = run(autocorrelation, data, method="direct", **common)

    assert fast.metadata.parameters["method"] == "fft"
    assert slow.metadata.parameters["method"] == "direct"
    np.testing.assert_allclose(fast.values, slow.values, rtol=1e-9, atol=1e-12)
    if batch > 1:
        assert fast.uncertainty is not None
        assert slow.uncertainty is not None
        np.testing.assert_allclose(fast.uncertainty, slow.uncertainty, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("count", [400, 401])
@pytest.mark.parametrize("max_lag", [None, 37])
@pytest.mark.parametrize("time_origin_average", [True, False])
@pytest.mark.parametrize("batch", [1, 5])
def test_fft_msd_matches_the_direct_reference(
    count: int, max_lag: int | None, time_origin_average: bool, batch: int
) -> None:
    states = ensemble(batch, steps=count - 1).states
    data = states[0] if batch == 1 else states
    common = {"max_lag": max_lag, "time_origin_average": time_origin_average}
    run = single if batch == 1 else _plain
    fast = run(mean_square_displacement, data, method="fft", **common)
    slow = run(mean_square_displacement, data, method="direct", **common)

    expected = "single_origin" if not time_origin_average else "fft"
    assert fast.metadata.parameters["method"] == expected
    np.testing.assert_allclose(fast.values, slow.values, rtol=1e-9, atol=1e-12)
    if batch > 1:
        assert fast.uncertainty is not None
        assert slow.uncertainty is not None
        np.testing.assert_allclose(fast.uncertainty, slow.uncertainty, rtol=1e-9, atol=1e-12)


def test_fft_msd_matches_the_direct_reference_with_observable_and_unwrap() -> None:
    states = iterate(StandardMap(kick_strength=0.5), floats([0.11, 0.23]), steps=999).states
    projected = single(mean_square_displacement, states, observable=momentum, method="fft")
    reference = single(mean_square_displacement, states, observable=momentum, method="direct")
    np.testing.assert_allclose(projected.values, reference.values, rtol=1e-9, atol=1e-12)

    fast = single(mean_square_displacement, states, unwrap=True, periods=1.0, method="fft")
    slow = single(mean_square_displacement, states, unwrap=True, periods=1.0, method="direct")
    np.testing.assert_allclose(fast.values, slow.values, rtol=1e-9, atol=1e-12)


def test_auto_method_switches_across_the_measured_crossover() -> None:
    below = ensemble(1, steps=999).states[0]
    assert (
        parameter(single(autocorrelation, below, observable=momentum, max_lag=100), "method")
        == "direct"
    )
    assert parameter(single(autocorrelation, below, observable=momentum), "method") == "fft"
    assert parameter(single(mean_square_displacement, below, max_lag=100), "method") == "direct"
    assert parameter(single(mean_square_displacement, below), "method") == "fft"


def test_unknown_method_names_are_rejected_with_the_valid_choices() -> None:
    series = floats([1.0, 2.0, 3.0, 4.0])
    for call in (autocorrelation, mean_square_displacement):
        with pytest.raises(ValidationError, match="'auto', 'direct', 'fft'"):
            call(series, method="bogus")  # type: ignore[arg-type]


def test_fft_autocorrelation_is_faster_than_the_direct_loop() -> None:
    series = np.sin(np.arange(20_000, dtype=np.float64) * 0.37)

    start = time.perf_counter()
    single(autocorrelation, series, method="direct")
    direct = time.perf_counter() - start
    start = time.perf_counter()
    single(autocorrelation, series, method="fft")
    fast = time.perf_counter() - start

    # Measured ratio is about 200x; the assertion is deliberately loose so that a
    # loaded CI machine cannot make a real algorithmic win look like a regression.
    assert fast * 5.0 < direct


def test_msd_overflow_names_the_cause_and_the_remedy() -> None:
    # Coordinates near 1e160 used to leak a bare "RuntimeWarning: overflow" and then
    # stop on the generic "must contain only finite values" check, which names
    # neither the squaring nor a way out.
    huge = np.geomspace(1.0, 1e160, 64, dtype=np.float64)[:, None]

    with pytest.raises(NumericalError) as failure:
        mean_square_displacement(huge, max_lag=30)

    message = str(failure.value)
    assert "squared displacements" in message
    assert "1.000e+160" in message
    assert "Rescale the coordinates" in message
    # Squaring is now the only accumulation that can overflow. The retired
    # time-origin standard error needed |dx|**4 and gave up around 1e77, so a
    # magnitude that the curve survives must no longer raise on its behalf.
    survivable = np.geomspace(1.0, 1e90, 64, dtype=np.float64)[:, None]
    for algorithm in ("direct", "fft"):
        result = single(mean_square_displacement, survivable, max_lag=30, method=algorithm)
        assert bool(np.all(np.isfinite(result.values)))


def test_autocorrelation_overflow_names_the_cause_and_the_remedy() -> None:
    series = np.geomspace(1.0, 1e200, 64, dtype=np.float64)

    with pytest.raises(NumericalError) as failure:
        autocorrelation(series, max_lag=10, demean=False)

    message = str(failure.value)
    assert "lag products" in message
    assert "Rescale the series" in message


def test_periodic_search_names_a_foreign_model_as_a_protocol_violation() -> None:
    with pytest.raises(ValidationError, match="ClassicalMap protocol") as failure:
        find_periodic_orbits(object(), floats([0.1, 0.2]), period=1)  # type: ignore[arg-type]

    message = str(failure.value)
    for member in ("bounds", "is_periodic", "jacobian", "state_dim", "step"):
        assert member in message


@pytest.mark.parametrize(
    "roundtrip", [lambda item: pickle.loads(pickle.dumps(item)), copy.deepcopy]
)
def test_periodic_orbit_result_survives_pickle_and_deepcopy(roundtrip: object) -> None:
    # NumPy drops writeable=False on pickling, so without __reduce__ a deepcopy
    # round trip handed back a result whose arrays could be mutated in place.
    original = find_periodic_orbits(
        CatMap(), floats([[0.8, 0.6], [0.2, 0.4]]), period=2, tolerance=1e-12
    )
    assert callable(roundtrip)
    restored = roundtrip(original)

    assert isinstance(restored, PeriodicOrbitResult)
    assert restored == original
    assert restored.period == original.period
    assert not restored.points.flags.writeable
    assert not restored.residuals.flags.writeable
    assert not restored.monodromy_matrices.flags.writeable
    assert not restored.stability_multipliers.flags.writeable
    assert restored.metadata == original.metadata


def test_transport_and_periodic_parameters_are_validated() -> None:
    with pytest.raises(ValidationError, match="max_lag must be less"):
        autocorrelation(floats([1.0, 2.0]), max_lag=2)
    with pytest.raises(ValidationError, match="outside"):
        local_diffusion_exponent(floats([0.0, 1.0, 2.0]), fit_start=1, fit_stop=4)
    with pytest.raises(ValidationError, match="period must be positive"):
        find_periodic_orbits(CatMap(), floats([0.0, 0.0]), period=0)

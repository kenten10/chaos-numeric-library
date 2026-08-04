from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

from chaos_numerics.classical import (
    CatMap,
    PeriodicOrbitResult,
    autocorrelation,
    find_periodic_orbits,
    local_diffusion_exponent,
    mean_square_displacement,
)
from chaos_numerics.core import ConvergenceError, ConvergenceWarning, ValidationError


def floats(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    return np.asarray(value, dtype=np.float64)


def test_autocorrelation_matches_direct_alternating_series() -> None:
    result = autocorrelation(
        floats([1.0, -1.0, 1.0, -1.0]),
        max_lag=3,
        demean=False,
    )

    np.testing.assert_allclose(result.values, floats([1.0, -1.0, 1.0, -1.0]))
    assert result.metadata.parameters["averaging"] == "all_time_origins"


def test_autocorrelation_observable_and_normalization_are_explicit() -> None:
    states = floats([[1.0, 9.0], [2.0, 8.0], [3.0, 7.0]])
    result = autocorrelation(
        states,
        observable=lambda values: values[:, 0],
        max_lag=1,
        demean=False,
        normalize=True,
    )

    assert result.values[0] == 1.0
    assert result.values[1] == pytest.approx(6.0 / 7.0)
    with pytest.raises(ValidationError, match="zero lag-zero"):
        autocorrelation(np.ones(4), normalize=True)


def test_msd_constant_velocity_matches_closed_form() -> None:
    times = np.arange(6, dtype=np.float64)
    positions = np.column_stack((2.0 * times, -times))
    result = mean_square_displacement(positions, max_lag=5)

    expected = 5.0 * np.arange(6, dtype=np.float64) ** 2
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-14)
    assert result.uncertainty is not None
    assert float(np.max(np.abs(result.uncertainty))) <= 1e-14


def test_msd_periodic_unwrap_recovers_continuous_motion() -> None:
    wrapped = floats([0.8, 0.9, 0.0, 0.1, 0.2])
    result = mean_square_displacement(
        wrapped,
        max_lag=4,
        unwrap=True,
        periods=floats([1.0]),
    )

    expected = 0.01 * np.arange(5, dtype=np.float64) ** 2
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValidationError, match="periods are required"):
        mean_square_displacement(wrapped, unwrap=True)


def test_local_diffusion_exponent_recovers_power_law_and_fit_metadata() -> None:
    times = np.arange(20, dtype=np.float64)
    msd = np.zeros_like(times)
    msd[1:] = 3.0 * times[1:] ** 1.5
    result = local_diffusion_exponent(msd, fit_start=1, fit_stop=20, times=times)

    assert result.values[0] == pytest.approx(1.5, abs=5e-3)
    assert result.residuals is not None
    assert result.residuals[0] <= 1e-12
    assert result.metadata.parameters["fit_start"] == 1
    assert result.metadata.parameters["fit_stop"] == 20
    with pytest.raises(ValidationError, match="at least three positive"):
        local_diffusion_exponent(floats([0.0, 1.0, 0.0]), fit_start=0, fit_stop=3)


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


def test_transport_and_periodic_parameters_are_validated() -> None:
    with pytest.raises(ValidationError, match="max_lag must be less"):
        autocorrelation(floats([1.0, 2.0]), max_lag=2)
    with pytest.raises(ValidationError, match="outside"):
        local_diffusion_exponent(floats([0.0, 1.0, 2.0]), fit_start=1, fit_stop=4)
    with pytest.raises(ValidationError, match="period must be positive"):
        find_periodic_orbits(CatMap(), floats([0.0, 0.0]), period=0)

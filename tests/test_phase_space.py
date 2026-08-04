from __future__ import annotations

import numpy as np
import pytest

from chaos_numerics.core import ValidationError
from chaos_numerics.quantum import (
    BoundaryPhases,
    basis_state,
    coherent_state,
    husimi_distribution,
    inverse_participation_ratio,
    normalize_state,
    participation_ratio,
    shannon_entropy,
)


@pytest.mark.parametrize("dimension", [2, 8, 31, 32])
def test_periodized_coherent_state_is_normalized_and_periodic(dimension: int) -> None:
    phases = BoundaryPhases(0.5, 0.25)
    state = coherent_state(
        dimension=dimension,
        position=0.97,
        momentum=0.13,
        boundary_phases=phases,
    )
    translated = coherent_state(
        dimension=dimension,
        position=1.97,
        momentum=-0.87,
        boundary_phases=phases,
    )

    assert state.dtype == np.dtype(np.complex128)
    assert abs(np.linalg.norm(state) - 1.0) <= 1e-12
    np.testing.assert_allclose(translated, state, rtol=1e-13, atol=1e-14)


def test_coherent_state_matches_independent_periodized_sum() -> None:
    dimension = 6
    position = 0.2
    momentum = 0.7
    phases = BoundaryPhases(0.25, 0.5)
    q_basis = (np.arange(dimension) + phases.position) / dimension
    expected = np.zeros(dimension, dtype=np.complex128)
    for index, q_value in enumerate(q_basis):
        for image in range(-4, 5):
            displacement = q_value - position + image
            expected[index] += (
                np.exp(-np.pi * dimension * displacement**2)
                * np.exp(2j * np.pi * dimension * momentum * displacement)
                * np.exp(-2j * np.pi * phases.momentum * image)
            )
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(
        coherent_state(
            dimension=dimension,
            position=position,
            momentum=momentum,
            boundary_phases=phases,
        ),
        expected,
        rtol=1e-13,
        atol=1e-14,
    )


def test_husimi_is_normalized_plot_ready_and_supports_batches() -> None:
    states = np.stack(
        (
            coherent_state(dimension=16, position=0.25, momentum=0.75),
            basis_state(dimension=16, index=4),
        )
    )
    result = husimi_distribution(states, grid_shape=(24, 20), chunk_size=37)

    assert result.values.shape == (2, 24, 20)
    assert result.grid_shape == (24, 20)
    assert result.grid_points.shape == (24, 20, 2)
    np.testing.assert_allclose(result.integral, np.ones(2), atol=1e-12)
    assert result.raw_integral.shape == (2,)
    assert result.cell_area == pytest.approx(1.0 / (24 * 20))
    assert bool(np.all(result.grid_points[..., 0] == result.positions[:, None]))
    assert bool(np.all(result.grid_points[..., 1] == result.momenta[None, :]))
    assert not result.values.flags.writeable


def test_husimi_raw_quadrature_converges_with_grid_refinement() -> None:
    phases = BoundaryPhases(0.5, 0.5)
    state = coherent_state(
        dimension=32,
        position=0.97,
        momentum=0.13,
        boundary_phases=phases,
    )
    coarse = husimi_distribution(
        state,
        grid_shape=(8, 8),
        boundary_phases=phases,
        normalize=False,
    )
    fine = husimi_distribution(
        state,
        grid_shape=(32, 32),
        boundary_phases=phases,
        normalize=False,
    )

    coarse_error = abs(float(coarse.raw_integral) - 1.0)
    fine_error = abs(float(fine.raw_integral) - 1.0)
    assert fine_error < coarse_error
    assert fine_error <= 1e-10


def test_localization_measures_for_basis_uniform_and_batch_states() -> None:
    localized = basis_state(dimension=8, index=3)
    uniform = normalize_state(np.ones(8))
    batch = np.stack((localized, uniform))

    assert inverse_participation_ratio(localized) == pytest.approx(1.0, abs=1e-14)
    assert participation_ratio(localized) == pytest.approx(1.0, abs=1e-14)
    assert shannon_entropy(localized) == pytest.approx(0.0, abs=1e-14)
    assert inverse_participation_ratio(uniform) == pytest.approx(1.0 / 8.0, rel=1e-12)
    assert participation_ratio(uniform) == pytest.approx(8.0, rel=1e-12)
    assert shannon_entropy(uniform) == pytest.approx(np.log(8.0), abs=1e-12)
    np.testing.assert_allclose(inverse_participation_ratio(batch), [1.0, 1.0 / 8.0])
    np.testing.assert_allclose(participation_ratio(batch), [1.0, 8.0])
    np.testing.assert_allclose(shannon_entropy(batch), [0.0, np.log(8.0)], atol=1e-14)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: coherent_state(dimension=0, position=0.0, momentum=0.0),
        lambda: coherent_state(dimension=4, position=np.nan, momentum=0.0),
        lambda: coherent_state(dimension=4, position=0.0, momentum=0.0, images=0),
        lambda: husimi_distribution([1.0, 0.0], grid_shape=(0, 4)),
        lambda: husimi_distribution([1.0, 0.0], grid_shape=[4, 4]),  # type: ignore[arg-type]
        lambda: inverse_participation_ratio([1.0, 1.0]),
    ],
)
def test_phase_space_parameter_validation(operation: object) -> None:
    with pytest.raises(ValidationError):
        operation()  # type: ignore[operator]

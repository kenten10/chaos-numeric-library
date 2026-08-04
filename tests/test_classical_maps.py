from __future__ import annotations

import math

import numpy as np
import pytest

from chaos_numerics import BakerMap, CatMap, StandardMap
from chaos_numerics.classical import iterate
from chaos_numerics.core import ClassicalMap, ValidationError


def floats(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    return np.asarray(value, dtype=np.float64)


def test_standard_map_known_step_and_symplectic_jacobian() -> None:
    model = StandardMap(kick_strength=2.0)
    state = floats([0.25, 0.1])
    expected_p = 0.1 + 1.0 / math.pi
    expected = floats([0.25 + expected_p, expected_p])

    np.testing.assert_allclose(model.step(state), expected, rtol=1e-13, atol=1e-14)
    jacobian = model.jacobian(state)
    omega = floats([[0.0, 1.0], [-1.0, 0.0]])
    defect = np.linalg.norm(jacobian.T @ omega @ jacobian - omega) / np.linalg.norm(omega)
    assert defect <= 1e-12
    assert isinstance(model, ClassicalMap)


def test_standard_map_jacobian_matches_centered_difference() -> None:
    model = StandardMap(kick_strength=1.7)
    state = floats([0.17, 0.23])
    step = 1e-7
    finite_difference = np.column_stack(
        [
            (
                model.step(state + step * floats([1.0, 0.0]))
                - model.step(state - step * floats([1.0, 0.0]))
            )
            / (2.0 * step),
            (
                model.step(state + step * floats([0.0, 1.0]))
                - model.step(state - step * floats([0.0, 1.0]))
            )
            / (2.0 * step),
        ]
    )

    np.testing.assert_allclose(model.jacobian(state), finite_difference, rtol=5e-6, atol=5e-8)


def test_cat_map_known_step_area_and_integer_validation() -> None:
    model = CatMap(matrix=((2, 1), (1, 1)))

    np.testing.assert_allclose(
        model.step(floats([0.2, 0.3])),
        floats([0.7, 0.5]),
        rtol=1e-13,
        atol=1e-14,
    )
    assert np.linalg.det(model.jacobian(floats([0.2, 0.3]))) == pytest.approx(1.0, abs=1e-14)
    with pytest.raises(ValidationError, match="2 x 2 integer matrix"):
        CatMap(matrix=((1, 0), (0, 1.5)))  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="determinant 1"):
        CatMap(matrix=((2, 0), (0, 1)))


def test_baker_map_cut_is_right_owned_and_area_preserving() -> None:
    model = BakerMap(cut=0.5)
    points = floats([[0.25, 0.4], [0.5, 0.4], [1.0, 0.4]])
    expected = floats([[0.5, 0.2], [0.0, 0.7], [0.0, 0.2]])

    np.testing.assert_allclose(model.step(points), expected, rtol=1e-13, atol=1e-14)
    jacobians = model.jacobian(points)
    np.testing.assert_allclose(np.linalg.det(jacobians), np.ones(3), rtol=0.0, atol=1e-14)
    with pytest.raises(ValidationError, match="0 < cut < 1"):
        BakerMap(cut=1.0)


@pytest.mark.parametrize(
    "model",
    [StandardMap(3.0), CatMap(), BakerMap(0.3)],
)
def test_batch_step_and_jacobian_match_stacked_scalar_calls(model: ClassicalMap) -> None:
    batch = floats([[0.1, 0.2], [0.4, 0.7], [1.2, -0.1]])

    expected_steps = np.stack([model.step(state) for state in batch])
    expected_jacobians = np.stack([model.jacobian(state) for state in batch])

    np.testing.assert_allclose(model.step(batch), expected_steps, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(
        model.jacobian(batch),
        expected_jacobians,
        rtol=1e-12,
        atol=1e-13,
    )


def test_iterate_scalar_shape_values_metadata_and_wrapping() -> None:
    model = CatMap()
    trajectory = iterate(model, floats([1.2, -0.1]), steps=2)
    expected_initial = floats([0.2, 0.9])
    expected = np.stack(
        [
            expected_initial,
            model.step(expected_initial),
            model.step(model.step(expected_initial)),
        ]
    )

    assert trajectory.shape == (3, 2)
    np.testing.assert_allclose(trajectory.states, expected, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(trajectory.initial_state, expected_initial)
    assert trajectory.metadata.parameters["steps"] == 2
    assert trajectory.metadata.parameters["model"] == "CatMap"


def test_iterate_batch_matches_scalar_and_can_exclude_initial() -> None:
    model = StandardMap(0.5)
    batch = floats([[0.1, 0.2], [0.3, 0.4]])
    result = iterate(model, batch, steps=3, include_initial=False)
    scalar = np.stack(
        [iterate(model, state, steps=3, include_initial=False).states for state in batch]
    )

    assert result.shape == (2, 3, 2)
    np.testing.assert_allclose(result.states, scalar, rtol=1e-11, atol=1e-13)
    np.testing.assert_array_equal(result.initial_state, batch)
    with pytest.raises(ValidationError, match="steps must be positive"):
        iterate(model, batch, steps=0, include_initial=False)


def test_map_and_iteration_parameters_are_validated() -> None:
    with pytest.raises(ValidationError, match="finite real"):
        StandardMap(float("nan"))
    with pytest.raises(ValidationError, match="trailing dimension 2"):
        CatMap().step(floats([0.1, 0.2, 0.3]))
    with pytest.raises(ValidationError, match="non-negative integer"):
        iterate(CatMap(), floats([0.1, 0.2]), steps=True)

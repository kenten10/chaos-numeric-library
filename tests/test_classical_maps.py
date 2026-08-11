from __future__ import annotations

import math

import numpy as np
import pytest

from chaos_numerics import BakerMap, CatMap, StandardMap
from chaos_numerics.classical import (
    LogisticMap,
    find_periodic_orbits,
    iterate,
    largest_lyapunov_exponent,
)
from chaos_numerics.core import ClassicalMap, ValidationError
from chaos_numerics.operators import UniformPartition, build_ulam, stationary_density


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


@pytest.mark.parametrize(
    "model",
    [StandardMap(8.0), StandardMap(0.5), CatMap(), BakerMap(0.3), BakerMap(0.5)],
)
def test_step_image_stays_inside_half_open_unit_torus(model: ClassicalMap) -> None:
    below_one = np.nextafter(1.0, 0.0)
    probes = floats(
        [
            [-1e-17, -1e-17],
            [below_one, below_one],
            [below_one, -1e-17],
            [-1e-17, below_one],
            [0.0, 0.0],
            [1.0 - 1e-17, 0.5],
        ]
    )

    batched = model.step(probes)
    assert bool(np.all((batched >= 0.0) & (batched < 1.0)))
    for probe in probes:
        single = model.step(probe)
        assert bool(np.all((single >= 0.0) & (single < 1.0)))


def test_iterate_wraps_rounding_below_zero_into_half_open_bounds() -> None:
    trajectory = iterate(CatMap(), floats([-1e-17, 0.5]), steps=2)

    assert bool(np.all((trajectory.initial_state >= 0.0) & (trajectory.initial_state < 1.0)))
    assert bool(np.all((trajectory.states >= 0.0) & (trajectory.states < 1.0)))
    np.testing.assert_array_equal(trajectory.initial_state, floats([0.0, 0.5]))


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


def test_iterate_names_a_foreign_model_as_a_protocol_violation() -> None:
    # validate_protocol documents that every public entry point routes a missing
    # attribute through it; iterate used to leak a bare AttributeError from the
    # model.state_dim read instead.
    with pytest.raises(ValidationError, match="ClassicalMap protocol") as failure:
        iterate(object(), floats([0.1, 0.2]), steps=2)  # type: ignore[arg-type]

    message = str(failure.value)
    for member in ("bounds", "is_periodic", "jacobian", "state_dim", "step"):
        assert member in message


class ScriptedMap:
    """A user-defined map that returns a chosen non-finite value on one step."""

    def __init__(self, *, at: int, value: float, recover: bool = False) -> None:
        self.at = at
        self.value = value
        self.recover = recover
        self.calls = 0

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def is_periodic(self) -> tuple[bool, bool]:
        return (False, False)

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0.0, 1.0), (0.0, 1.0))

    def step(self, state: object, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        self.calls += 1
        base = np.asarray(state, dtype=np.float64)
        if self.calls == self.at:
            return np.full_like(base, self.value)
        if self.recover:
            return np.full_like(base, 0.25)
        return base * 0.5

    def jacobian(self, state: object, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        del state
        return 0.5 * np.eye(2, dtype=np.float64)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_iterate_detects_a_non_finite_state_produced_mid_trajectory(value: float) -> None:
    model = ScriptedMap(at=7, value=value)

    with pytest.raises(ValidationError, match="non-finite state within 20 steps"):
        iterate(model, floats([0.3, 0.4]), steps=20)


def test_iterate_detects_a_non_finite_state_that_a_later_step_swallows() -> None:
    # The trajectory is validated as a whole, so a value that does not propagate is
    # still caught: every state the map produced is stored and inspected.
    model = ScriptedMap(at=4, value=float("inf"), recover=True)

    with pytest.raises(ValidationError, match="non-finite state"):
        iterate(model, floats([0.3, 0.4]), steps=20)
    assert model.calls == 20


@pytest.mark.parametrize("model", [StandardMap(2.0), CatMap(), BakerMap(0.3)])
def test_iterate_equals_repeated_step_bit_for_bit(model: ClassicalMap) -> None:
    initial = floats([0.11, 0.23])
    state = initial
    expected = [state]
    for _ in range(2000):
        state = model.step(state)
        expected.append(state)

    result = iterate(model, initial, steps=2000)

    np.testing.assert_array_equal(result.states, np.stack(expected))


def test_iterate_still_names_a_badly_shaped_step_result() -> None:
    class WrongShapeMap(ScriptedMap):
        def step(self, state: object, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
            del state
            return np.zeros((3, 2), dtype=np.float64)

    with pytest.raises(ValidationError, match="model step must preserve state shape"):
        iterate(WrongShapeMap(at=0, value=0.0), floats([[0.3, 0.4], [0.5, 0.6]]), steps=2)
    with pytest.raises(ValidationError, match="model step result must have 1 dimensions"):
        iterate(WrongShapeMap(at=0, value=0.0), floats([0.3, 0.4]), steps=2)


def test_logistic_map_conforms_to_the_classical_map_protocol() -> None:
    model = LogisticMap()

    assert isinstance(model, ClassicalMap)
    assert model.state_dim == 1
    assert model.is_periodic == (False,)
    assert model.bounds == ((0.0, 1.0),)
    assert model.parameters == {"rate": 4.0}
    np.testing.assert_allclose(model.step(floats([0.25])), floats([0.75]), rtol=0.0, atol=1e-16)


def test_logistic_map_rate_four_clamps_the_measure_zero_image_below_one() -> None:
    # rate=4 sends x=1/2 to exactly 1.0, the excluded endpoint of bounds.
    model = LogisticMap(4.0)
    below_one = np.nextafter(1.0, 0.0)
    probes = floats(
        [
            [0.0],
            [0.5],
            [np.nextafter(0.5, 0.0)],
            [np.nextafter(0.5, 1.0)],
            [below_one],
            [1e-17],
        ]
    )

    assert float(model.step(floats([0.5]))[0]) == below_one
    batched = model.step(probes)
    assert bool(np.all((batched >= 0.0) & (batched < 1.0)))
    for probe in probes:
        assert bool(np.all((model.step(probe) >= 0.0) & (model.step(probe) < 1.0)))


def test_logistic_map_jacobian_matches_centered_difference() -> None:
    model = LogisticMap(3.7)
    state = floats([0.37])
    step = 1e-7
    finite_difference = ((model.step(state + step) - model.step(state - step)) / (2.0 * step))[
        :, None
    ]

    np.testing.assert_allclose(model.jacobian(state), finite_difference, rtol=5e-6, atol=5e-8)
    assert model.jacobian(floats([[0.1], [0.9]])).shape == (2, 1, 1)


def test_logistic_map_rate_and_state_shape_are_validated() -> None:
    with pytest.raises(ValidationError, match="0 < rate <= 4"):
        LogisticMap(4.5)
    with pytest.raises(ValidationError, match="0 < rate <= 4"):
        LogisticMap(0.0)
    with pytest.raises(ValidationError, match="finite real"):
        LogisticMap(float("inf"))
    with pytest.raises(ValidationError, match="trailing dimension 1"):
        LogisticMap().step(floats([0.1, 0.2]))
    # The domain is enforced where a state enters an experiment, not in step, so
    # that a derivative-free root solver may probe just outside a fixed point.
    np.testing.assert_allclose(LogisticMap(4.0).step(floats([-1e-9])), floats([-4e-9]))


def test_logistic_map_rate_four_lyapunov_exponent_matches_log_two() -> None:
    # x = sin(pi theta / 2)**2 conjugates rate=4 to the angle doubling map, so the
    # exponent is exactly log 2 independently of the (almost every) initial point.
    result = largest_lyapunov_exponent(
        LogisticMap(4.0),
        floats([0.2]),
        steps=20_000,
        transient=100,
        convergence_rtol=1e-2,
    )

    assert float(result.values[0]) == pytest.approx(math.log(2.0), abs=1e-3)
    assert result.metadata.convergence is not None
    assert result.metadata.convergence.converged


def test_logistic_map_rate_four_ulam_density_matches_the_arcsine_law() -> None:
    # The unique absolutely continuous invariant measure of rate=4 is the arcsine
    # law, whose exact mass on [a, b) is (arcsin(sqrt(b)) - arcsin(sqrt(a))) * 2/pi.
    cells = 400
    partition = UniformPartition([[0.0, 1.0]], [cells])
    matrix = build_ulam(LogisticMap(4.0), partition, samples_per_cell=4000, seed=20260808)
    density = stationary_density(matrix)

    edges = np.linspace(0.0, 1.0, cells + 1)
    exact = (np.arcsin(np.sqrt(edges[1:])) - np.arcsin(np.sqrt(edges[:-1]))) * 2.0 / math.pi
    assert float(np.sum(exact)) == pytest.approx(1.0, abs=1e-12)
    assert float(np.sum(np.abs(density.values - exact))) <= 0.07


def test_logistic_map_fixed_points_and_period_two_multipliers() -> None:
    # f'(0) = rate and f'(1 - 1/rate) = 2 - rate, and the period-two multiplier of
    # the logistic map is the exact polynomial 4 + 2 rate - rate**2.
    rate = 3.2
    model = LogisticMap(rate)
    fixed = find_periodic_orbits(model, floats([[1e-9], [1.0 - 1.0 / rate]]), period=1)

    multipliers = sorted(float(value.real) for value in fixed.stability_multipliers.ravel())
    assert multipliers == pytest.approx([2.0 - rate, rate], abs=1e-9)

    orbit = find_periodic_orbits(model, floats([[0.3], [0.5], [0.8], [0.9]]), period=2)
    assert orbit.count == 1
    assert float(orbit.stability_multipliers[0, 0].real) == pytest.approx(
        4.0 + 2.0 * rate - rate * rate, abs=1e-9
    )
    assert bool(np.all((orbit.points >= 0.0) & (orbit.points < 1.0)))


def test_iterate_supports_one_dimensional_logistic_states() -> None:
    trajectory = iterate(LogisticMap(4.0), floats([0.3]), steps=5)

    assert trajectory.shape == (6, 1)
    assert bool(np.all((trajectory.states >= 0.0) & (trajectory.states < 1.0)))
    assert trajectory.metadata.parameters["model"] == "LogisticMap"
    with pytest.raises(ValidationError, match=r"must lie in \[0.0, 1.0\)"):
        iterate(LogisticMap(), floats([1.5]), steps=1)

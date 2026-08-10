from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numpy.typing import ArrayLike

from chaos_numerics.classical import (
    CatMap,
    StandardMap,
    largest_lyapunov_exponent,
    lyapunov_spectrum,
)
from chaos_numerics.core import (
    ConvergenceError,
    ConvergenceWarning,
    NumericalError,
    NumericalWarning,
    ValidationError,
)


def floats(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    return np.asarray(value, dtype=np.float64)


def cat_exponents() -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    eigenvalues = np.linalg.eigvals(floats([[2.0, 1.0], [1.0, 1.0]]))
    return np.sort(np.log(np.abs(eigenvalues)))[::-1]


def test_cat_map_spectrum_matches_analytic_values_and_sum_rule() -> None:
    result = lyapunov_spectrum(
        CatMap(),
        floats([0.1, 0.2]),
        steps=128,
        transient=64,
        reorthogonalization_interval=4,
    )

    np.testing.assert_allclose(result.values, cat_exponents(), rtol=0.0, atol=1e-13)
    assert abs(float(np.sum(result.values))) <= 1e-10
    assert result.metadata.convergence is not None
    assert result.metadata.convergence.converged
    defect = result.metadata.parameters["maximum_orthogonality_defect"]
    assert isinstance(defect, float)
    assert defect <= 1e-12


def test_largest_exponent_tangent_method_matches_cat_map() -> None:
    result = largest_lyapunov_exponent(
        CatMap(),
        floats([0.3, 0.7]),
        steps=128,
        transient=64,
        reorthogonalization_interval=8,
    )

    assert result.values[0] == pytest.approx(cat_exponents()[0], abs=1e-10)
    assert result.metadata.convergence is not None
    assert result.metadata.convergence.converged


def standard_map_hyperbolic_exponent() -> float:
    """``ln(2 + sqrt(3))`` for the ``(0, 0)`` fixed point of the ``K = 2`` standard map."""
    return math.log(2.0 + math.sqrt(3.0))


def test_standard_map_hyperbolic_fixed_point_spectrum_is_analytic() -> None:
    # At (0, 0) with K = 2 the Jacobian is [[3, 1], [2, 1]], whose eigenvalues are
    # 2 +- sqrt(3); this also exercises a state-dependent Jacobian, so a wrong
    # step() would move the orbit off the fixed point and change the answer.
    result = lyapunov_spectrum(
        StandardMap(kick_strength=2.0),
        floats([0.0, 0.0]),
        steps=512,
        transient=128,
    )
    target = standard_map_hyperbolic_exponent()

    assert target == pytest.approx(1.3169578969248166, abs=0.0, rel=1e-15)
    np.testing.assert_allclose(result.values, floats([target, -target]), rtol=0.0, atol=1e-12)
    assert result.metadata.convergence is not None
    assert result.metadata.convergence.converged


def test_largest_exponent_matches_standard_map_hyperbolic_fixed_point() -> None:
    result = largest_lyapunov_exponent(
        StandardMap(kick_strength=2.0),
        floats([0.0, 0.0]),
        steps=512,
        transient=128,
    )

    assert result.values[0] == pytest.approx(standard_map_hyperbolic_exponent(), abs=1e-12)
    assert result.metadata.convergence is not None
    assert result.metadata.convergence.converged


def test_standard_map_elliptic_and_free_fixed_points_have_vanishing_exponents() -> None:
    # (0.5, 0) with K = 2 has Jacobian [[-1, 1], [-2, 1]], trace 0, so its
    # multipliers are +-i and every exponent vanishes. K = 0 is free streaming.
    elliptic = lyapunov_spectrum(
        StandardMap(kick_strength=2.0),
        floats([0.5, 0.0]),
        steps=512,
        transient=128,
    )
    free = lyapunov_spectrum(
        StandardMap(kick_strength=0.0),
        floats([0.0, 0.0]),
        steps=512,
        transient=128,
    )

    np.testing.assert_allclose(elliptic.values, np.zeros(2), rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(free.values, np.zeros(2))


def test_reorthogonalization_overflow_raises_numerical_error_with_remedy() -> None:
    with pytest.raises(NumericalError, match="reorthogonalization_interval") as spectrum_failure:
        lyapunov_spectrum(
            CatMap(),
            floats([0.1, 0.2]),
            steps=2000,
            reorthogonalization_interval=800,
        )
    with pytest.raises(NumericalError, match="reorthogonalization_interval") as vector_failure:
        largest_lyapunov_exponent(
            CatMap(),
            floats([0.1, 0.2]),
            steps=2000,
            reorthogonalization_interval=800,
        )

    assert "800" in str(spectrum_failure.value)
    assert "800" in str(vector_failure.value)


def test_scaled_norm_keeps_moderate_reorthogonalization_intervals_usable() -> None:
    # np.linalg.norm sums squares and used to overflow near 1e154, which made this
    # interval fail with an unexplained NumericalError instead of an answer.
    result = largest_lyapunov_exponent(
        CatMap(),
        floats([0.1, 0.2]),
        steps=2000,
        transient=64,
        reorthogonalization_interval=500,
    )

    assert result.values[0] == pytest.approx(cat_exponents()[0], abs=1e-12)


def test_finite_time_history_ends_at_result_and_respects_stride() -> None:
    result = lyapunov_spectrum(
        CatMap(),
        floats([0.2, 0.4]),
        steps=64,
        transient=32,
        reorthogonalization_interval=2,
        history_interval=4,
    )
    convergence = result.metadata.convergence

    assert convergence is not None
    assert convergence.history is not None
    assert convergence.history.ndim == 2
    assert convergence.history.shape[1] == 2
    assert convergence.history.shape[0] < 64 // 2
    np.testing.assert_array_equal(convergence.history[-1], result.values)
    assert not convergence.history.flags.writeable


def test_transient_removes_initial_basis_alignment_bias() -> None:
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        without_transient = lyapunov_spectrum(
            CatMap(),
            floats([0.1, 0.2]),
            steps=32,
            transient=0,
        )
    with_transient = lyapunov_spectrum(
        CatMap(),
        floats([0.1, 0.2]),
        steps=32,
        transient=64,
    )
    target = cat_exponents()[0]

    assert abs(with_transient.values[0] - target) < abs(without_transient.values[0] - target)


def test_insufficient_iterations_warn_and_strict_mode_raises() -> None:
    with pytest.warns(ConvergenceWarning, match="at least 32 measured steps"):
        result = lyapunov_spectrum(CatMap(), floats([0.1, 0.2]), steps=8, transient=16)

    assert result.metadata.convergence is not None
    assert not result.metadata.convergence.converged
    assert result.metadata.warnings[0].code == "insufficient-iterations"
    with pytest.raises(ConvergenceError, match="did not converge"):
        lyapunov_spectrum(
            CatMap(),
            floats([0.1, 0.2]),
            steps=8,
            transient=16,
            strict=True,
        )


def test_seeded_initial_basis_is_reproducible() -> None:
    first = lyapunov_spectrum(
        CatMap(),
        floats([0.1, 0.2]),
        steps=64,
        transient=32,
        reorthogonalization_interval=2,
        seed=7,
    )
    second = lyapunov_spectrum(
        CatMap(),
        floats([0.1, 0.2]),
        steps=64,
        transient=32,
        reorthogonalization_interval=2,
        seed=7,
    )

    np.testing.assert_array_equal(first.values, second.values)
    assert first.metadata == second.metadata


def test_seeded_tangent_vector_actually_changes_the_finite_time_estimate() -> None:
    # Reproducibility alone is satisfied by ignoring the seed, so pin the values:
    # without a seed the tangent starts at the all-ones direction, with seed=7 at a
    # standard-normal draw, and 32 steps from the same state keep the difference.
    with pytest.warns(ConvergenceWarning):
        unseeded = largest_lyapunov_exponent(CatMap(), floats([0.1, 0.2]), steps=32, transient=0)
    with pytest.warns(ConvergenceWarning):
        seeded = largest_lyapunov_exponent(
            CatMap(), floats([0.1, 0.2]), steps=32, transient=0, seed=7
        )

    assert float(unseeded.values[0]) == pytest.approx(0.9615762960317349, abs=1e-12)
    assert float(seeded.values[0]) == pytest.approx(0.9425382332727741, abs=1e-12)
    assert abs(float(seeded.values[0]) - float(unseeded.values[0])) > 1e-3
    assert unseeded.metadata.seed is None
    assert seeded.metadata.seed == 7


@pytest.mark.parametrize("call", [largest_lyapunov_exponent, lyapunov_spectrum])
def test_lyapunov_names_a_foreign_model_as_a_protocol_violation(call: object) -> None:
    assert callable(call)
    with pytest.raises(ValidationError, match="ClassicalMap protocol") as failure:
        call(object(), floats([0.1, 0.2]), steps=32)

    message = str(failure.value)
    for member in ("bounds", "is_periodic", "jacobian", "state_dim", "step"):
        assert member in message


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("steps", 1, "at least 2"),
        ("transient", -1, "non-negative"),
        ("reorthogonalization_interval", 0, "must be positive"),
        ("history_interval", 0, "must be positive"),
        ("convergence_rtol", -1.0, "non-negative"),
        ("seed", True, "non-negative integer"),
    ],
)
def test_lyapunov_parameters_are_validated(keyword: str, value: object, message: str) -> None:
    arguments: dict[str, object] = {"steps": 32, keyword: value}
    with pytest.raises(ValidationError, match=message):
        lyapunov_spectrum(CatMap(), floats([0.1, 0.2]), **arguments)  # type: ignore[arg-type]


def test_batched_initial_state_is_rejected_in_v01() -> None:
    with pytest.raises(ValidationError, match="initial_state must have 1 dimensions"):
        lyapunov_spectrum(CatMap(), floats([[0.1, 0.2]]), steps=32)


class SingularMap:
    @property
    def state_dim(self) -> int:
        return 2

    @property
    def is_periodic(self) -> tuple[bool, bool]:
        return (False, False)

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((-1.0, 1.0), (-1.0, 1.0))

    def step(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.asarray(state, dtype=np.float64)

    def jacobian(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        del state
        return np.zeros((2, 2), dtype=np.float64)


def test_singular_tangent_dynamics_raise_numerical_error() -> None:
    with pytest.raises(NumericalError, match="singular"):
        lyapunov_spectrum(SingularMap(), floats([0.0, 0.0]), steps=32)
    with pytest.raises(NumericalError, match="zero or non-finite norm"):
        largest_lyapunov_exponent(SingularMap(), floats([0.0, 0.0]), steps=32)


def test_ill_conditioned_qr_refuses_the_silently_wrong_trailing_exponent() -> None:
    # Before the conditioning guard, an interval of 40 returned +0.81 for the
    # trailing cat-map exponent whose exact value is -0.9624, with no warning and
    # a perfectly orthogonal basis (the orthogonality defect stayed at 1e-16).
    with pytest.raises(NumericalError, match="ill-conditioned") as failure:
        lyapunov_spectrum(
            CatMap(),
            floats([0.1234, 0.5678]),
            steps=3_000,
            transient=100,
            reorthogonalization_interval=40,
        )

    assert "reorthogonalization_interval" in str(failure.value)
    assert "40" in str(failure.value)


def test_marginal_qr_condition_warns_and_is_recorded() -> None:
    with pytest.warns(NumericalWarning, match="trailing exponents may have lost precision"):
        result = lyapunov_spectrum(
            CatMap(),
            floats([0.1234, 0.5678]),
            steps=3_000,
            transient=100,
            reorthogonalization_interval=24,
        )

    condition = result.metadata.parameters["maximum_qr_condition"]
    assert isinstance(condition, float)
    limit = result.metadata.parameters["qr_condition_warning_limit"]
    assert isinstance(limit, float)
    assert condition > limit
    # The exponents are still accurate here, which is why this warns rather than raises.
    np.testing.assert_allclose(result.values, cat_exponents(), rtol=0.0, atol=1e-12)


def test_short_intervals_stay_silent_and_record_the_condition() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", NumericalWarning)
        result = lyapunov_spectrum(
            CatMap(),
            floats([0.1234, 0.5678]),
            steps=3_000,
            transient=100,
            reorthogonalization_interval=8,
        )

    condition = result.metadata.parameters["maximum_qr_condition"]
    assert isinstance(condition, float)
    assert condition < result.metadata.parameters["qr_condition_warning_limit"]  # type: ignore[operator]
    np.testing.assert_allclose(result.values, cat_exponents(), rtol=0.0, atol=1e-13)

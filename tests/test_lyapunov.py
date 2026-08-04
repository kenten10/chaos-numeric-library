from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

from chaos_numerics.classical import (
    CatMap,
    largest_lyapunov_exponent,
    lyapunov_spectrum,
)
from chaos_numerics.core import (
    ConvergenceError,
    ConvergenceWarning,
    NumericalError,
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

    np.testing.assert_allclose(result.values, cat_exponents(), rtol=0.0, atol=1e-10)
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
    np.testing.assert_allclose(
        convergence.history[-1],
        result.values,
        rtol=1e-14,
        atol=1e-15,
    )
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

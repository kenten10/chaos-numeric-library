from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from chaos_numerics.classical import CatMap
from chaos_numerics.core import Spectrum, ValidationError
from chaos_numerics.operators import (
    UniformPartition,
    build_ulam,
    leading_eigenpairs,
    spectral_gap,
    stationary_density,
)


def markov_matrix() -> csr_matrix:
    return csr_matrix(np.asarray([[0.8, 0.1], [0.2, 0.9]], dtype=np.float64))


def test_dense_leading_eigenpairs_have_expected_values_and_residuals() -> None:
    spectrum = leading_eigenpairs(markov_matrix(), count=2)

    assert isinstance(spectrum, Spectrum)
    np.testing.assert_allclose(spectrum.eigenvalues, np.asarray([1.0, 0.7]), atol=1e-14)
    assert spectrum.eigenvectors is not None
    assert spectrum.eigenvectors.shape == (2, 2)
    assert spectrum.residuals is not None
    assert float(np.max(spectrum.residuals)) <= 1e-12
    assert spectrum.metadata.parameters["side"] == "right"
    assert spectrum.metadata.parameters["method"] == "dense"


def test_left_stationary_eigenvector_is_constant() -> None:
    spectrum = leading_eigenpairs(markov_matrix(), count=2, side="left")
    index = int(np.argmin(np.abs(spectrum.eigenvalues - 1.0)))
    assert spectrum.eigenvectors is not None
    vector = np.real(spectrum.eigenvectors[:, index])

    np.testing.assert_allclose(vector, np.ones(2) / np.sqrt(2.0), atol=1e-14)
    assert spectrum.residuals is not None
    assert spectrum.residuals[index] <= 1e-12


def test_stationary_density_is_normalized_invariant_and_nonnegative() -> None:
    matrix = markov_matrix()
    result = stationary_density(matrix)

    np.testing.assert_allclose(result.values, np.asarray([1.0 / 3.0, 2.0 / 3.0]), atol=1e-14)
    assert float(np.sum(result.values)) == pytest.approx(1.0, abs=1e-14)
    assert float(np.min(result.values)) >= -1e-12
    assert float(np.linalg.norm(matrix @ result.values - result.values, ord=1)) <= 1e-10
    assert result.metadata.convergence is not None
    assert result.metadata.convergence.converged


def test_spectral_gap_matches_known_second_eigenvalue() -> None:
    from_operator = spectral_gap(markov_matrix())
    spectrum = leading_eigenpairs(markov_matrix(), count=2)
    from_spectrum = spectral_gap(spectrum)

    assert from_operator.values[0] == pytest.approx(0.3, abs=1e-14)
    np.testing.assert_array_equal(from_operator.values, from_spectrum.values)


def test_modulus_tie_prioritizes_stationary_eigenvalue() -> None:
    periodic = csr_matrix(np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    spectrum = leading_eigenpairs(periodic, count=2)

    assert spectrum.eigenvalues[0] == pytest.approx(1.0)
    assert spectral_gap(spectrum).values[0] == pytest.approx(0.0, abs=1e-14)


def test_sparse_arpack_path_returns_small_independent_residuals() -> None:
    dimension = 12
    matrix = 0.75 * np.eye(dimension) + 0.25 * np.roll(np.eye(dimension), 1, axis=0)
    spectrum = leading_eigenpairs(csr_matrix(matrix), count=4, tolerance=1e-12)

    assert spectrum.metadata.parameters["method"] == "arpack"
    assert spectrum.count == 4
    assert spectrum.residuals is not None
    assert float(np.max(spectrum.residuals)) <= 1e-9
    assert abs(spectrum.eigenvalues[0] - 1.0) <= 1e-12


def test_cat_ulam_detects_eigenvalue_one_and_stationary_density() -> None:
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(4, 4),
        periodic=(True, True),
    )
    matrix = build_ulam(CatMap(), partition, samples_per_cell=256, seed=17)
    spectrum = leading_eigenpairs(matrix, count=4)
    density = stationary_density(matrix)

    assert abs(spectrum.eigenvalues[0] - 1.0) <= 1e-10
    assert spectrum.residuals is not None
    assert float(np.max(spectrum.residuals)) <= 1e-9
    assert float(np.sum(density.values)) == pytest.approx(1.0, abs=1e-12)
    assert float(np.min(density.values)) >= -1e-12
    assert float(np.linalg.norm(matrix @ density.values - density.values, ord=1)) <= 1e-10


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0}, "count must be positive"),
        ({"count": 3}, "must not exceed"),
        ({"count": 1, "side": "diagonal"}, "side must be"),
        ({"count": 1, "tolerance": 0.0}, "finite positive"),
    ],
)
def test_eigensolver_parameters_are_validated(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        leading_eigenpairs(markov_matrix(), **kwargs)  # type: ignore[arg-type]


def test_spectral_gap_requires_two_values() -> None:
    spectrum = leading_eigenpairs(np.asarray([[1.0]]), count=1)
    with pytest.raises(ValidationError, match="at least two"):
        spectral_gap(spectrum)

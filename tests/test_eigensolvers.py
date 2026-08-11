from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from chaos_numerics.classical import CatMap
from chaos_numerics.core import (
    ConvergenceError,
    ConvergenceWarning,
    NumericalWarning,
    Spectrum,
    ValidationError,
)
from chaos_numerics.operators import (
    UlamMatrix,
    UniformPartition,
    build_ulam,
    leading_eigenpairs,
    spectral_gap,
    stationary_density,
)


class LeakyDoublingMap:
    """``x -> 2x mod 1`` observed on ``[0, 0.75)``, so the hole leaks mass out."""

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def is_periodic(self) -> tuple[bool, ...]:
        return (False,)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return ((0.0, 1.0),)

    def step(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.mod(2.0 * np.asarray(state, dtype=np.float64), 1.0)

    def jacobian(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        del state
        return np.full((1, 1), 2.0, dtype=np.float64)


def leaky_ulam_matrix() -> UlamMatrix:
    partition = UniformPartition(bounds=((0.0, 0.75),), shape=(7,))
    return build_ulam(
        LeakyDoublingMap(),
        partition,
        samples_per_cell=64,
        seed=3,
        open_system=True,
    )


def markov_matrix() -> csr_matrix:
    return csr_matrix(np.asarray([[0.8, 0.1], [0.2, 0.9]], dtype=np.float64))


def known_spectrum_matrix() -> csr_matrix:
    """A non-symmetric matrix whose eigenvalues are exactly its diagonal."""
    diagonal = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float64)
    dense = np.diag(diagonal) + np.triu(np.full((diagonal.size, diagonal.size), 0.05), 1)
    return csr_matrix(dense)


def three_eigenvalue_matrix() -> csr_matrix:
    """Upper triangular, so the eigenvalues are exactly ``1.0``, ``0.9``, ``0.8``."""
    diagonal = np.asarray([1.0, 0.9, 0.8], dtype=np.float64)
    dense = np.diag(diagonal) + np.triu(np.full((3, 3), 0.05), 1)
    return csr_matrix(dense)


def column_stochastic_matrix(dimension: int) -> csr_matrix:
    """A dense-but-sparse-typed column-stochastic matrix with no special structure."""
    generator = np.random.default_rng(5)
    dense = generator.random((dimension, dimension)) ** 3
    dense /= dense.sum(axis=0)
    return csr_matrix(dense)


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


def test_spectral_gap_picks_the_largest_remaining_modulus_not_the_smallest() -> None:
    """``lambda_2`` is the *largest* remaining modulus, checked past ``count=2``.

    Every other caller of :func:`spectral_gap` passes a two-eigenvalue spectrum,
    which leaves exactly one remaining eigenvalue and makes ``argmax`` and
    ``argmin`` over it indistinguishable. With three known eigenvalues the two
    choices separate: the gap is ``1 - 0.9 = 0.1``, and picking the smallest
    remaining modulus would report ``1 - 0.8 = 0.2``.
    """
    spectrum = leading_eigenpairs(three_eigenvalue_matrix(), count=3)
    np.testing.assert_allclose(np.real(spectrum.eigenvalues), [1.0, 0.9, 0.8], atol=1e-12)

    gap = spectral_gap(spectrum)

    assert float(gap.values[0]) == pytest.approx(0.1, abs=1e-12)
    assert gap.metadata.parameters["lambda_2_real"] == pytest.approx(0.9, abs=1e-12)


def test_recorded_convergence_tolerance_is_the_residual_gate_actually_applied() -> None:
    """``metadata.convergence.tolerance`` is the gate, not a decorative echo.

    The ARPACK gate is ``max(requested_tolerance, 1e-9)``: the independently
    evaluated residuals cannot be pushed below roughly ``1e-9``, so a tighter
    request is floored, and a looser one is honoured verbatim. Pinning both ends
    is what stops the floor from being silently relaxed.
    """
    matrix = csr_matrix(0.75 * np.eye(12) + 0.25 * np.roll(np.eye(12), 1, axis=0))

    floored = leading_eigenpairs(matrix, count=4, tolerance=1e-10)
    assert floored.metadata.parameters["method"] == "arpack"
    assert floored.metadata.convergence is not None
    assert floored.metadata.convergence.tolerance == 1e-9
    assert floored.metadata.parameters["requested_tolerance"] == 1e-10

    honoured = leading_eigenpairs(matrix, count=4, tolerance=1e-7)
    assert honoured.metadata.convergence is not None
    assert honoured.metadata.convergence.tolerance == 1e-7

    dense = leading_eigenpairs(markov_matrix(), count=2, tolerance=1e-10)
    assert dense.metadata.parameters["method"] == "dense"
    assert dense.metadata.convergence is not None
    assert dense.metadata.convergence.tolerance == 1e-12


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


@pytest.mark.parametrize("count", [5, 6])
def test_eigenvalues_are_ordered_by_decreasing_modulus(count: int) -> None:
    """The documented ordering is verified past ``lambda_1``, on both solver paths.

    ``count=5`` uses ARPACK and ``count=6`` reaches the dense fallback, and the
    expected values pin the ordering rather than just the set, so replacing
    ``argsort(-moduli)`` with ``argsort(moduli)`` fails here.
    """
    expected = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float64)[:count]
    spectrum = leading_eigenpairs(known_spectrum_matrix(), count=count)

    np.testing.assert_allclose(np.real(spectrum.eigenvalues), expected, atol=1e-10)
    np.testing.assert_allclose(np.imag(spectrum.eigenvalues), np.zeros(count), atol=1e-12)
    moduli = np.abs(spectrum.eigenvalues)
    assert bool(np.all(np.diff(moduli) < 0.0))
    assert float(moduli[0]) == pytest.approx(1.0, abs=1e-10)


def test_arpack_non_convergence_warns_and_is_recorded_and_strict_raises() -> None:
    matrix = column_stochastic_matrix(40)
    starved = {"count": 6, "max_iterations": 1, "tolerance": 1e-14}

    with pytest.warns(ConvergenceWarning, match="did not fully converge"):
        spectrum = leading_eigenpairs(matrix, **starved)  # type: ignore[arg-type]

    assert spectrum.metadata.convergence is not None
    assert not spectrum.metadata.convergence.converged
    assert [warning.code for warning in spectrum.metadata.warnings] == ["eigenpairs-not-converged"]
    assert spectrum.count < 6

    with pytest.raises(ConvergenceError, match="did not fully converge"):
        leading_eigenpairs(matrix, strict=True, **starved)  # type: ignore[arg-type]


def test_dense_fallback_warns_with_a_memory_estimate() -> None:
    dimension = 300
    dense = 0.9 * np.eye(dimension) + 0.1 * np.roll(np.eye(dimension), 1, axis=0)

    with pytest.warns(NumericalWarning, match=r"dense 300x300 complex128 array of about 1\.4 MiB"):
        spectrum = leading_eigenpairs(csr_matrix(dense), count=dimension - 1)

    assert spectrum.metadata.parameters["method"] == "dense"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert leading_eigenpairs(markov_matrix(), count=2).count == 2


def test_cat_map_stationary_density_converges_to_the_uniform_measure() -> None:
    """Independent physics check: the cat map preserves Lebesgue measure.

    The Ulam stationary density must approach ``1 / n_cells`` uniformly, and the
    sampling error must fall like ``1 / sqrt(samples_per_cell)``.
    """
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(4, 4),
        periodic=(True, True),
    )
    uniform = 1.0 / partition.size
    errors = []
    for samples in (256, 1024, 8192):
        matrix = build_ulam(CatMap(), partition, samples_per_cell=samples, seed=17)
        density = stationary_density(matrix)
        assert float(np.sum(density.values)) == pytest.approx(1.0, abs=1e-12)
        errors.append(float(np.max(np.abs(density.values - uniform))))

    assert errors[0] < 0.02
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 0.005


def test_left_leading_eigenvector_of_a_column_stochastic_ulam_matrix_is_constant() -> None:
    """Column sums of one force the left leading eigenvector to be constant."""
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(4, 4),
        periodic=(True, True),
    )
    matrix = build_ulam(CatMap(), partition, samples_per_cell=1024, seed=17)
    spectrum = leading_eigenpairs(matrix, count=4, side="left")
    index = int(np.argmin(np.abs(spectrum.eigenvalues - 1.0)))
    assert spectrum.eigenvectors is not None
    vector = spectrum.eigenvectors[:, index]

    np.testing.assert_allclose(np.imag(vector), np.zeros(partition.size), atol=1e-14)
    np.testing.assert_allclose(
        np.real(vector),
        np.full(partition.size, 1.0 / np.sqrt(partition.size)),
        atol=1e-14,
    )


def test_open_system_ulam_matrix_flows_through_every_solver() -> None:
    """A composed ``UlamMatrix`` is unwrapped for ARPACK, ``.T``, and the norms.

    The escape vector must be genuinely nonzero, because a substochastic operator
    is what exercises the unwrapping: the leading eigenvalue is the escape rate
    rather than one, so the convergence checks of ``stationary_density`` fire and
    every code path that touches the operator is visited.
    """
    matrix = leaky_ulam_matrix()
    assert float(np.sum(matrix.escape_probabilities)) > 0.0

    right = leading_eigenpairs(matrix, count=2)
    left = leading_eigenpairs(matrix, count=2, side="left")
    assert right.residuals is not None
    assert left.residuals is not None
    assert float(np.max(right.residuals)) <= 1e-9
    assert float(np.max(left.residuals)) <= 1e-9
    np.testing.assert_allclose(right.eigenvalues, np.conjugate(left.eigenvalues), atol=1e-12)
    assert float(np.abs(right.eigenvalues[0])) < 1.0

    with pytest.warns(ConvergenceWarning, match="invariance or eigenvalue tolerance"):
        density = stationary_density(matrix, strict=False)
    assert float(np.sum(density.values)) == pytest.approx(1.0, abs=1e-12)
    assert float(np.min(density.values)) >= -1e-12

    gap = spectral_gap(matrix, strict=False)
    assert gap.values.shape == (1,)
    assert float(gap.values[0]) == pytest.approx(1.0 - abs(right.eigenvalues[1]), abs=1e-9)

    with pytest.raises(ConvergenceError, match="differs from one"):
        spectral_gap(matrix)


def test_every_operators_diagnostic_declares_its_category() -> None:
    """A diagnostic that lets ``category`` default is filed under the wrong label.

    ``Diagnostic.category`` defaults to ``"numerical"``, so a convergence failure
    that forgets to say so is indistinguishable from a rounding complaint when a
    caller filters ``metadata.warnings`` by category. Both diagnostics the
    ``operators`` package can emit are convergence failures.
    """
    with pytest.warns(ConvergenceWarning):
        spectrum = leading_eigenpairs(
            column_stochastic_matrix(40), count=6, max_iterations=1, tolerance=1e-14
        )
    with pytest.warns(ConvergenceWarning):
        density = stationary_density(leaky_ulam_matrix(), strict=False)

    emitted = (*spectrum.metadata.warnings, *density.metadata.warnings)
    assert {warning.code for warning in emitted} == {
        "eigenpairs-not-converged",
        "stationary-density-not-converged",
    }
    for warning in emitted:
        assert warning.category == "convergence", warning.code


def test_ulam_matrix_reaches_the_dense_fallback_warning() -> None:
    """The sparse-to-dense estimate must see through the composition wrapper.

    ``issparse`` is ``False`` for a ``UlamMatrix``, so the warning would go
    silent if the operator were not unwrapped before the size estimate.
    """
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(17, 17),
        periodic=(True, True),
    )
    matrix = build_ulam(CatMap(), partition, samples_per_cell=8, seed=5)
    assert matrix.shape == (289, 289)

    with pytest.warns(NumericalWarning, match=r"dense 289x289 complex128 array"):
        spectrum = leading_eigenpairs(matrix, count=288)

    assert spectrum.metadata.parameters["method"] == "dense"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0}, "count must be positive"),
        ({"count": 3}, "must not exceed"),
        ({"count": 1, "side": "diagonal"}, "side must be one of 'right', 'left'"),
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


def test_non_finite_operators_are_rejected_on_every_solver_branch() -> None:
    """The same bad operator must give the same error whichever solver runs.

    The finiteness check used to sit only in the dense fallback, so ``count``
    picked the failure mode: ``count >= dimension - 1`` took the dense branch and
    raised a `ValidationError` naming the problem, while a smaller ``count`` went
    to ARPACK and surfaced `ArpackError -9999` about workspace on SciPy 1.18, or a
    `ConvergenceError` on SciPy 1.11 that blamed convergence for invalid input.
    One input, three outcomes, none of them dependent on anything the caller meant.

    A sparse operator is now screened through ``.data``, so nothing is densified to
    perform the check.
    """
    dimension = 10
    for label, operator in (
        ("sparse nan", csr_matrix(np.diag([np.nan] + [0.1] * (dimension - 1)))),
        ("sparse inf", csr_matrix(np.diag([np.inf] + [0.1] * (dimension - 1)))),
        ("dense nan", np.diag([np.nan] + [0.1] * (dimension - 1))),
    ):
        for count in (2, dimension - 1, dimension):
            with pytest.raises(ValidationError, match="only finite values"):
                leading_eigenpairs(operator, count=count)
        with pytest.raises(ValidationError, match="only finite values"):
            stationary_density(operator)
        with pytest.raises(ValidationError, match="only finite values"):
            spectral_gap(operator)
        assert label  # keep the label meaningful in a failure report

    # The screen must not reject a legitimate sparse operator whose *implicit*
    # zeros outnumber its stored entries.
    sparse = csr_matrix(np.eye(dimension) * 0.5)
    assert leading_eigenpairs(sparse, count=2).count == 2

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]

from chaos_numerics.core import NumericalError, QuantumMap, ValidationError
from chaos_numerics.quantum import (
    BoundaryPhases,
    DenseUnitary,
    QuantumBasis,
    basis_state,
    eigenphases,
    eigenstates,
    evolve,
    normalize_state,
    quantum_state,
    unitarity_defect,
)


def complexes(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    return np.asarray(value, dtype=np.complex128)


def hadamard() -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    return np.asarray(
        complexes([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
        dtype=np.complex128,
    )


class MatrixFreeUnitary:
    def __init__(self, matrix: np.ndarray[tuple[int, ...], np.dtype[np.complex128]]) -> None:
        self._matrix = matrix.copy()

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[0])

    def apply(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        values = np.asarray(state, dtype=np.complex128)
        return np.asarray(values @ self._matrix.T, dtype=np.complex128)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            self._matrix.shape,
            matvec=lambda vector: self._matrix @ vector,
            rmatvec=lambda vector: self._matrix.conj().T @ vector,
            dtype=np.dtype(np.complex128),
        )


def test_basis_state_and_state_normalization() -> None:
    state = basis_state(dimension=4, index=2, basis=QuantumBasis.MOMENTUM)

    assert state.dtype == np.dtype(np.complex128)
    assert np.linalg.norm(state) == 1.0
    np.testing.assert_array_equal(state, complexes([0.0, 0.0, 1.0, 0.0]))
    batch = normalize_state(complexes([[3.0, 4.0], [1.0j, 1.0j]]))
    np.testing.assert_allclose(np.linalg.norm(batch, axis=-1), np.ones(2), atol=5e-15)

    with pytest.raises(ValidationError, match=r"index must lie in \[0, 4\)"):
        basis_state(dimension=4, index=4)
    with pytest.raises(ValidationError, match="zero norm"):
        normalize_state(complexes([0.0, 0.0]))


def test_boundary_phases_are_canonical_modulo_one() -> None:
    phases = BoundaryPhases(position=1.25, momentum=-0.25)

    assert phases.position == 0.25
    assert phases.momentum == 0.75
    assert phases.to_dict() == {"position": 0.25, "momentum": 0.75}
    with pytest.raises(ValidationError, match="finite real"):
        BoundaryPhases(position=float("nan"))


def test_quantum_state_validates_norm_shape_and_dtype() -> None:
    state = quantum_state([1, 0], dimension=2)
    assert state.dtype == np.dtype(np.complex128)

    with pytest.raises(ValidationError, match="norm must equal one"):
        quantum_state(complexes([1.0, 1.0]), dimension=2)
    with pytest.raises(ValidationError, match="trailing dimension 3"):
        quantum_state(complexes([1.0, 0.0]), dimension=3)
    with pytest.raises(ValidationError, match="finite"):
        quantum_state(complexes([np.inf, 0.0]), normalized=False)


def test_dense_unitary_apply_and_linear_operator_match() -> None:
    model = DenseUnitary(
        hadamard(),
        basis="position",
        boundary_phases=BoundaryPhases(0.5, 0.25),
        name="hadamard",
    )
    state = normalize_state(complexes([1.0 + 1.0j, 2.0]))
    batch = np.stack((state, basis_state(dimension=2, index=0)))

    np.testing.assert_allclose(model.apply(state), hadamard() @ state, atol=1e-14)
    np.testing.assert_allclose(model.apply(batch), batch @ hadamard().T, atol=1e-14)
    linear = model.as_linear_operator()
    np.testing.assert_allclose(linear.matvec(state), model.apply(state), atol=1e-14)
    np.testing.assert_allclose(linear.rmatvec(state), hadamard().conj().T @ state, atol=1e-14)
    assert isinstance(model, QuantumMap)
    assert not model.matrix.flags.writeable
    assert model.defect <= 1e-12


def test_dense_and_matrix_free_evolve_match_for_scalar_batch_and_history() -> None:
    dense = DenseUnitary(hadamard())
    matrix_free = MatrixFreeUnitary(hadamard())
    batch = np.stack(
        (
            basis_state(dimension=2, index=0),
            normalize_state(complexes([1.0, 1.0j])),
        )
    )

    dense_final = evolve(dense, batch, steps=7, method="dense")
    free_final = evolve(matrix_free, batch, steps=7, method="matrix_free")
    history = evolve(matrix_free, batch, steps=7, return_history=True)

    np.testing.assert_allclose(dense_final, free_final, atol=5e-12)
    assert history.shape == (2, 8, 2)
    np.testing.assert_allclose(history[..., -1, :], free_final, atol=5e-12)
    np.testing.assert_allclose(np.linalg.norm(history, axis=-1), np.ones((2, 8)), atol=2e-13)


def test_unitarity_defect_and_invalid_dense_operator() -> None:
    assert unitarity_defect(hadamard()) <= 1e-12
    assert unitarity_defect(MatrixFreeUnitary(hadamard())) <= 1e-12

    with pytest.raises(ValidationError, match="unitary defect"):
        DenseUnitary(complexes([[1.0, 0.0], [0.0, 2.0]]))
    with pytest.raises(ValidationError, match="exceeds dense_limit"):
        unitarity_defect(MatrixFreeUnitary(np.eye(4, dtype=np.complex128)), dense_limit=3)


def test_known_diagonal_eigensystem_phases_and_residuals() -> None:
    diagonal = complexes([1.0, 1.0j, -1.0, -1.0j])
    model = DenseUnitary(np.diag(diagonal))
    result = eigenstates(model)
    phase_result = eigenphases(model)

    np.testing.assert_allclose(
        result.eigenphases,
        np.asarray([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0]),
        atol=1e-14,
    )
    np.testing.assert_array_equal(phase_result.values, result.eigenphases)
    assert result.count == 4
    assert float(np.max(result.residuals)) <= 1e-12
    assert result.metadata.parameters["phase_interval"] == "[-pi, pi)"
    assert result.metadata.convergence is not None
    assert result.metadata.convergence.converged


def test_matrix_free_eigensystem_materializes_same_dense_reference() -> None:
    dense = eigenstates(DenseUnitary(hadamard()))
    matrix_free = eigenstates(MatrixFreeUnitary(hadamard()))

    np.testing.assert_allclose(matrix_free.eigenphases, dense.eigenphases, atol=1e-14)
    assert float(np.max(matrix_free.residuals)) <= 1e-12


class ExpandingMap(MatrixFreeUnitary):
    def __init__(self) -> None:
        super().__init__(2.0 * np.eye(2, dtype=np.complex128))


def test_evolve_rejects_norm_drift_and_invalid_parameters() -> None:
    state = basis_state(dimension=2, index=0)
    with pytest.raises(NumericalError, match="norm drift"):
        evolve(ExpandingMap(), state, steps=1)
    with pytest.raises(ValidationError, match="non-negative integer"):
        evolve(DenseUnitary(hadamard()), state, steps=True)
    with pytest.raises(ValidationError, match="invalid evolution method"):
        evolve(DenseUnitary(hadamard()), state, steps=1, method="fft")

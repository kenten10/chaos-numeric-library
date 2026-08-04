"""Dense unitary adapters, diagnostics, and common eigensystem routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]

from chaos_numerics.core import (
    AnalysisResult,
    ConvergenceInfo,
    Diagnostic,
    EigenstateResult,
    ExperimentMetadata,
    NumericalError,
    QuantumMap,
    ValidationError,
)
from chaos_numerics.core._validation import as_complex_array
from chaos_numerics.core.types import ArrayLike, ComplexArray
from chaos_numerics.quantum.states import BoundaryPhases, QuantumBasis, quantum_state


@dataclass(frozen=True, slots=True, eq=False, init=False)
class DenseUnitary:
    """Immutable dense unitary implementing the ``QuantumMap`` protocol."""

    matrix: ComplexArray
    basis: QuantumBasis
    boundary_phases: BoundaryPhases
    name: str
    defect: float

    def __init__(
        self,
        matrix: ArrayLike,
        *,
        basis: QuantumBasis | str = QuantumBasis.POSITION,
        boundary_phases: BoundaryPhases | None = None,
        name: str = "dense_unitary",
        tolerance: float = 1e-12,
    ) -> None:
        dense = as_complex_array(matrix, name="unitary matrix", ndim=2, copy=True)
        if dense.shape[0] != dense.shape[1] or dense.shape[0] == 0:
            raise ValidationError(f"unitary matrix must be non-empty and square; got {dense.shape}")
        if not name or name.strip() != name:
            raise ValidationError("name must be a non-empty trimmed string")
        try:
            basis_value = QuantumBasis(basis)
        except ValueError as error:
            raise ValidationError(f"invalid quantum basis {basis!r}") from error
        phases = BoundaryPhases() if boundary_phases is None else boundary_phases
        if not isinstance(phases, BoundaryPhases):
            raise ValidationError("boundary_phases must be a BoundaryPhases instance")
        limit = _positive_float(tolerance, name="tolerance")
        defect = _dense_unitarity_defect(dense)
        if defect > limit:
            raise ValidationError(f"unitary defect {defect} exceeds construction tolerance {limit}")
        dense.setflags(write=False)
        object.__setattr__(self, "matrix", dense)
        object.__setattr__(self, "basis", basis_value)
        object.__setattr__(self, "boundary_phases", phases)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "defect", defect)

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "basis": self.basis.value,
            "boundary_phases": self.boundary_phases.to_dict(),
        }

    def apply(self, state: ArrayLike, /) -> ComplexArray:
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        return np.asarray(values @ self.matrix.T, dtype=np.complex128)

    def as_linear_operator(self) -> LinearOperator:
        matrix = self.matrix
        return LinearOperator(
            shape=matrix.shape,
            dtype=np.dtype(np.complex128),
            matvec=lambda vector: matrix @ vector,
            rmatvec=lambda vector: matrix.conj().T @ vector,
            matmat=lambda values: matrix @ values,
        )

    def to_dense(self) -> ComplexArray:
        """Return an independent writable dense copy."""
        return self.matrix.copy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DenseUnitary):
            return NotImplemented
        return (
            np.array_equal(self.matrix, other.matrix)
            and self.basis == other.basis
            and self.boundary_phases == other.boundary_phases
            and self.name == other.name
        )


def unitarity_defect(model_or_matrix: QuantumMap | ArrayLike, *, dense_limit: int = 256) -> float:
    """Return ``||U.H U - I||_F / sqrt(N)`` from a bounded dense representation."""
    dense = _dense_representation(model_or_matrix, dense_limit=dense_limit)
    return _dense_unitarity_defect(dense)


def eigenstates(model: QuantumMap, *, dense_limit: int = 256) -> EigenstateResult:
    """Return circularly sorted dense-reference eigenphases and eigenstates."""
    dense = _dense_representation(model, dense_limit=dense_limit)
    defect = _dense_unitarity_defect(dense)
    if defect > 1e-12:
        raise NumericalError(f"unitary defect {defect} exceeds eigensystem tolerance 1e-12")
    eigenvalues, vectors = np.linalg.eig(dense)
    vectors = _normalize_columns(np.asarray(vectors, dtype=np.complex128))
    phases = np.mod(np.angle(eigenvalues) + np.pi, 2.0 * np.pi) - np.pi
    order = np.argsort(phases, kind="stable")
    phases = np.asarray(phases[order], dtype=np.float64)
    vectors = vectors[:, order]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.complex128)
    residuals = np.empty(model.dimension, dtype=np.float64)
    for index in range(model.dimension):
        vector = vectors[:, index]
        numerator = np.linalg.norm(model.apply(vector) - eigenvalues[index] * vector)
        residuals[index] = numerator / max(
            np.finfo(np.float64).tiny,
            (1.0 + abs(eigenvalues[index])) * np.linalg.norm(vector),
        )
    converged = bool(np.all(residuals <= 1e-12))
    if not converged:
        raise NumericalError(
            f"dense unitary eigenpair residual {float(np.max(residuals))} exceeds 1e-12"
        )
    convergence = ConvergenceInfo(
        converged=True,
        iterations=0,
        residual=float(np.max(residuals)),
        tolerance=1e-12,
        history=residuals,
        reason="all dense eigenpair residuals satisfy tolerance",
    )
    circular_gaps = np.diff(np.concatenate((phases, phases[:1] + 2.0 * np.pi)))
    degeneracy_tolerance = 1e-10
    degenerate_pairs = int(np.count_nonzero(circular_gaps <= degeneracy_tolerance))
    warnings: list[Diagnostic] = []
    if degenerate_pairs:
        warnings.append(
            Diagnostic(
                "eigenphase-degeneracy",
                f"detected {degenerate_pairs} circular eigenphase gaps at or below "
                f"{degeneracy_tolerance}",
            )
        )
    symmetry_tolerance = 1e-10
    symmetry_defects = _symmetry_defects(model, dense)
    for name, symmetry_defect in symmetry_defects.items():
        if symmetry_defect > symmetry_tolerance:
            warnings.append(
                Diagnostic(
                    "broken-symmetry",
                    f"{name} commutator defect {symmetry_defect} exceeds {symmetry_tolerance}",
                )
            )
    metadata = ExperimentMetadata(
        parameters={
            "phase_interval": "[-pi, pi)",
            "ordering": "ascending eigenphase",
            "eigenstate_normalization": "unit L2 norm; largest component positive real",
            "unitarity_defect": defect,
            "dense_limit": dense_limit,
            "minimum_circular_phase_gap": float(np.min(circular_gaps)),
            "degenerate_pairs": degenerate_pairs,
            "degeneracy_tolerance": degeneracy_tolerance,
            "symmetry_defects": symmetry_defects,
            "symmetry_tolerance": symmetry_tolerance,
        },
        precision="complex128",
        warnings=tuple(warnings),
        convergence=convergence,
    )
    return EigenstateResult(phases, vectors, residuals, metadata)


def eigenphases(model: QuantumMap, *, dense_limit: int = 256) -> AnalysisResult:
    """Return sorted phases while retaining eigensystem residuals and metadata."""
    result = eigenstates(model, dense_limit=dense_limit)
    return AnalysisResult(
        "eigenphases",
        result.eigenphases,
        residuals=result.residuals,
        metadata=result.metadata,
    )


def _dense_representation(
    model_or_matrix: QuantumMap | ArrayLike,
    *,
    dense_limit: int,
) -> ComplexArray:
    limit = _positive_int(dense_limit, name="dense_limit")
    if not isinstance(model_or_matrix, QuantumMap):
        dense = as_complex_array(model_or_matrix, name="operator", ndim=2, copy=True)
        if dense.shape[0] != dense.shape[1]:
            raise ValidationError(f"operator must be square; got {dense.shape}")
        return dense
    model = model_or_matrix
    dimension = _positive_int(model.dimension, name="model dimension")
    if dimension > limit:
        raise ValidationError(
            f"dense materialization dimension {dimension} exceeds dense_limit {limit}"
        )
    to_dense = getattr(model, "to_dense", None)
    if callable(to_dense):
        dense = as_complex_array(to_dense(), name="dense operator", ndim=2, copy=True)
    else:
        basis = np.eye(dimension, dtype=np.complex128)
        dense = np.column_stack([model.apply(basis[:, index]) for index in range(dimension)])
    if dense.shape != (dimension, dimension):
        raise ValidationError(
            f"dense representation must have shape ({dimension}, {dimension}); got {dense.shape}"
        )
    return dense


def _dense_unitarity_defect(matrix: ComplexArray) -> float:
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    return float(np.linalg.norm(matrix.conj().T @ matrix - identity) / np.sqrt(matrix.shape[0]))


def _symmetry_defects(model: QuantumMap, dense: ComplexArray) -> dict[str, float]:
    operators = getattr(model, "symmetry_operators", {})
    if not isinstance(operators, dict):
        raise ValidationError("symmetry_operators must be a dictionary")
    defects: dict[str, float] = {}
    for name, operator in operators.items():
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValidationError("symmetry operator names must be non-empty trimmed strings")
        symmetry = as_complex_array(operator, name=f"{name} symmetry", ndim=2, copy=False)
        if symmetry.shape != dense.shape:
            raise ValidationError(
                f"{name} symmetry must have shape {dense.shape}; got {symmetry.shape}"
            )
        defects[name] = float(
            np.linalg.norm(dense @ symmetry - symmetry @ dense) / np.sqrt(dense.shape[0])
        )
    return defects


def _normalize_columns(vectors: ComplexArray) -> ComplexArray:
    result = vectors.copy()
    for column in range(result.shape[1]):
        norm = np.linalg.norm(result[:, column])
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            raise NumericalError("eigensolver returned a zero or non-finite eigenstate")
        result[:, column] /= norm
        pivot = int(np.argmax(np.abs(result[:, column])))
        result[:, column] *= np.exp(-1j * np.angle(result[pivot, column]))
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a positive integer; got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValidationError(f"{name} must be positive; got {result}")
    return result


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite positive number; got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValidationError(f"{name} must be a finite positive number; got {value!r}")
    return result


__all__ = ["DenseUnitary", "eigenphases", "eigenstates", "unitarity_defect"]

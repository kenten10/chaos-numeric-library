"""Dense reference quantizations of the cat and symmetric baker maps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]

from chaos_numerics.core import ValidationError
from chaos_numerics.core.types import ArrayLike, ComplexArray
from chaos_numerics.quantum.states import BoundaryPhases
from chaos_numerics.quantum.unitary import DenseUnitary

CatMatrix = tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class QuantumCatMap:
    """Periodic metaplectic quantization for cat maps with ``b = +/-1``.

    The supported kernel uses an even-dimensional position grid ``q_j=j/N``.
    Restricting the upper-right matrix entry to ``+/-1`` makes the generating
    function single-valued in this finite periodic convention.
    """

    dimension: int
    matrix: CatMatrix
    boundary_phases: BoundaryPhases
    _operator: DenseUnitary = field(repr=False)
    _parity: ComplexArray = field(repr=False)

    def __init__(
        self,
        dimension: int,
        matrix: CatMatrix = ((2, 1), (1, 1)),
        boundary_phase: float | BoundaryPhases = 0.0,
    ) -> None:
        size = _even_dimension(dimension, model="QuantumCatMap")
        canonical_matrix = _cat_matrix(matrix)
        if abs(canonical_matrix[0][1]) != 1:
            raise ValidationError(
                "QuantumCatMap requires matrix[0][1] to be +1 or -1 in the "
                "supported generating-function quantization"
            )
        phases = _boundary_phases(boundary_phase)
        if phases != BoundaryPhases():
            raise ValidationError("QuantumCatMap supports only periodic boundary phases (0, 0)")

        dense = _cat_floquet(size, canonical_matrix)
        operator = DenseUnitary(
            dense,
            boundary_phases=phases,
            name="quantum_cat_map",
        )
        parity = _parity_operator(size, antiperiodic=False)
        object.__setattr__(self, "dimension", size)
        object.__setattr__(self, "matrix", canonical_matrix)
        object.__setattr__(self, "boundary_phases", phases)
        object.__setattr__(self, "_operator", operator)
        object.__setattr__(self, "_parity", parity)

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "matrix": self.matrix,
            "boundary_phases": self.boundary_phases.to_dict(),
            "basis": "position",
            "quantization": "periodic_metaplectic_b_unit",
        }

    @property
    def symmetry_operators(self) -> dict[str, ComplexArray]:
        """Return the position-space parity operator used by diagnostics."""
        return {"parity": self._parity}

    def apply(self, state: ArrayLike, /) -> ComplexArray:
        return self._operator.apply(state)

    def as_linear_operator(self) -> LinearOperator:
        return self._operator.as_linear_operator()

    def to_dense(self) -> ComplexArray:
        return self._operator.to_dense()


@dataclass(frozen=True, slots=True, eq=False, init=False)
class QuantumBakerMap:
    """Saraceno quantization of the symmetric classical baker map.

    This convention requires even ``N`` and anti-periodic boundary phases
    ``(alpha, beta) = (1/2, 1/2)`` to preserve parity symmetry.
    """

    dimension: int
    boundary_phases: BoundaryPhases
    _operator: DenseUnitary = field(repr=False)
    _parity: ComplexArray = field(repr=False)

    def __init__(
        self,
        dimension: int,
        boundary_phase: float | BoundaryPhases = 0.5,
    ) -> None:
        size = _even_dimension(dimension, model="QuantumBakerMap")
        phases = _boundary_phases(boundary_phase)
        if phases != BoundaryPhases(0.5, 0.5):
            raise ValidationError(
                "QuantumBakerMap requires anti-periodic boundary phases (0.5, 0.5)"
            )

        fourier = _twisted_fourier(size, phases)
        half_fourier = _twisted_fourier(size // 2, phases)
        blocks = np.zeros((size, size), dtype=np.complex128)
        blocks[: size // 2, : size // 2] = half_fourier
        blocks[size // 2 :, size // 2 :] = half_fourier
        dense = fourier.conj().T @ blocks
        operator = DenseUnitary(
            dense,
            boundary_phases=phases,
            name="quantum_baker_map",
        )
        parity = _parity_operator(size, antiperiodic=True)
        object.__setattr__(self, "dimension", size)
        object.__setattr__(self, "boundary_phases", phases)
        object.__setattr__(self, "_operator", operator)
        object.__setattr__(self, "_parity", parity)

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "cut": 0.5,
            "boundary_phases": self.boundary_phases.to_dict(),
            "basis": "position",
            "quantization": "saraceno_symmetric",
        }

    @property
    def symmetry_operators(self) -> dict[str, ComplexArray]:
        """Return the position-space parity operator used by diagnostics."""
        return {"parity": self._parity}

    def apply(self, state: ArrayLike, /) -> ComplexArray:
        return self._operator.apply(state)

    def as_linear_operator(self) -> LinearOperator:
        return self._operator.as_linear_operator()

    def to_dense(self) -> ComplexArray:
        return self._operator.to_dense()


def _cat_floquet(dimension: int, matrix: CatMatrix) -> ComplexArray:
    a, b = matrix[0]
    _, d = matrix[1]
    indices = np.arange(dimension, dtype=np.float64)
    input_index = indices[None, :]
    output_index = indices[:, None]
    action = (a * input_index**2 - 2.0 * input_index * output_index + d * output_index**2) / (
        2.0 * b
    )
    prefactor = np.exp(-0.25j * np.pi * np.sign(b)) / np.sqrt(dimension)
    return np.asarray(
        prefactor * np.exp(2j * np.pi * action / dimension),
        dtype=np.complex128,
    )


def _twisted_fourier(dimension: int, phases: BoundaryPhases) -> ComplexArray:
    indices = np.arange(dimension, dtype=np.float64)
    exponent = np.outer(indices + phases.momentum, indices + phases.position)
    return np.asarray(
        np.exp(-2j * np.pi * exponent / dimension) / np.sqrt(dimension),
        dtype=np.complex128,
    )


def _parity_operator(dimension: int, *, antiperiodic: bool) -> ComplexArray:
    indices = np.arange(dimension)
    targets = dimension - 1 - indices if antiperiodic else np.mod(-indices, dimension)
    parity = np.zeros((dimension, dimension), dtype=np.complex128)
    parity[targets, indices] = 1.0
    parity.setflags(write=False)
    return parity


def _boundary_phases(value: float | BoundaryPhases) -> BoundaryPhases:
    if isinstance(value, BoundaryPhases):
        return value
    return BoundaryPhases(position=value, momentum=value)


def _even_dimension(value: object, *, model: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{model} dimension must be an even positive integer; got {value!r}")
    result = int(value)
    if result <= 0 or result % 2:
        raise ValidationError(f"{model} dimension must be an even positive integer; got {result}")
    return result


def _cat_matrix(value: object) -> CatMatrix:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "QuantumCatMap matrix must be a regular 2 x 2 integer matrix"
        ) from error
    if raw.shape != (2, 2) or raw.dtype.kind not in "iu":
        raise ValidationError(
            "QuantumCatMap matrix must be a 2 x 2 integer matrix; "
            f"got shape {raw.shape}, dtype {raw.dtype}"
        )
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValidationError("QuantumCatMap matrix entries must fit in signed 64-bit integers")
    values = raw.astype(np.int64, copy=False)
    result: CatMatrix = (
        (int(values[0, 0]), int(values[0, 1])),
        (int(values[1, 0]), int(values[1, 1])),
    )
    determinant = result[0][0] * result[1][1] - result[0][1] * result[1][0]
    if determinant != 1:
        raise ValidationError(
            f"QuantumCatMap matrix must have exact integer determinant 1; got {determinant}"
        )
    return result


__all__ = ["QuantumBakerMap", "QuantumCatMap"]

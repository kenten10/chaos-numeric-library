"""Dense reference quantizations of the cat and symmetric baker maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]

from chaos_numerics.core import ValidationError
from chaos_numerics.core.types import ArrayLike, ComplexArray
from chaos_numerics.quantum.states import BoundaryPhases, _parity_operator
from chaos_numerics.quantum.unitary import DenseUnitary

CatMatrix: TypeAlias = tuple[tuple[int, int], tuple[int, int]]
"""Row-major ``2 x 2`` integer matrix accepted by :class:`QuantumCatMap`.

Written out, ``((a, b), (c, d))`` for the linear torus map
``(q, p) -> (a q + b p, c q + d p) mod 1``. It is exported because it appears in
the ``matrix`` parameter of :class:`QuantumCatMap` and in its ``matrix``
attribute, and a library that ships ``py.typed`` has to let callers annotate
their own wrappers with the same type.

Valid values are constrained twice over: the determinant ``a d - b c`` must be
exactly ``1`` so that the map is area-preserving, and ``b`` must be ``+1`` or
``-1`` for the periodic metaplectic quantization used here to stay
single-valued. Anything else is a :class:`ValidationError`.
"""


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
        boundary_phases: float | BoundaryPhases = 0.0,
    ) -> None:
        size = _even_dimension(dimension, model="QuantumCatMap")
        canonical_matrix = _cat_matrix(matrix)
        if abs(canonical_matrix[0][1]) != 1:
            raise ValidationError(
                "QuantumCatMap requires matrix[0][1] to be +1 or -1 in the "
                "supported generating-function quantization"
            )
        phases = _boundary_phases(boundary_phases)
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

    **Avoid ``N = 2**k`` for spectral statistics.** The classical baker map is
    the binary shift, and a power-of-two dimension resonates with it: the
    eigenphase spectrum keeps an arithmetic structure that survives
    desymmetrization. Measured mean adjacent gap ratio of each parity sector
    against the COE reference 0.5307, standard error of the sector mean in
    brackets:

    ============= ================= =================
    ``N``         even sector       odd sector
    ============= ================= =================
    256           0.3806 (0.0244)   0.4038 (0.0238)
    512           0.4078 (0.0177)   0.4309 (0.0175)
    1024          0.4389 (0.0122)   0.4559 (0.0122)
    700           0.5211 (0.0134)   0.5324 (0.0134)
    802           0.5386 (0.0129)   0.5568 (0.0132)
    900           0.5093 (0.0120)   0.5310 (0.0120)
    ============= ================= =================

    The power-of-two rows sit 6 to 8 standard errors below COE and hardly
    improve on the raw two-sector value of 0.42, while the generic even rows
    agree with COE to within 2. This is a known property of the map at these
    dimensions and not an implementation defect -- measured unitarity defect
    ``||U.H U - I||_F / sqrt(N)`` stays at or below ``2.0e-13`` and the parity
    expectation values are ``+/-1`` to ``2.3e-15`` at every ``N`` in the table.
    Dimensions with a large power of two in them (``768 = 3 * 2**8``, even
    sector 0.5170) are intermediate. Pick a dimension such as 700, 802, or 900
    when the point of the calculation is a comparison with random-matrix theory;
    ``N = 2**k`` remains the right choice for studying the symbolic dynamics
    itself.
    """

    dimension: int
    boundary_phases: BoundaryPhases
    _operator: DenseUnitary = field(repr=False)
    _parity: ComplexArray = field(repr=False)

    def __init__(
        self,
        dimension: int,
        boundary_phases: float | BoundaryPhases = 0.5,
    ) -> None:
        size = _even_dimension(dimension, model="QuantumBakerMap")
        phases = _boundary_phases(boundary_phases)
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


__all__ = ["CatMatrix", "QuantumBakerMap", "QuantumCatMap"]

"""Dense unitary adapters, diagnostics, and common eigensystem routines.

``SymmetrySector`` is exported because it appears in the signature of
:func:`desymmetrize`, and a library that ships ``py.typed`` has to let callers
annotate their own wrappers with the same type. It is the eigenvalue of the
symmetry operator that selects a block: ``"even"`` keeps the states with
``<psi|S|psi> = +1`` and ``"odd"`` the states with ``-1``. Those are the only two
values, because :func:`desymmetrize` only accepts involutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, get_args

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
from chaos_numerics.core._validation import (
    as_complex_array,
    validate_protocol,
    wrap_into_half_open,
)
from chaos_numerics.core.types import ArrayLike, ComplexArray, FloatArray, IndexArray
from chaos_numerics.quantum.states import BoundaryPhases, QuantumBasis, quantum_state

DEFAULT_DENSE_LIMIT: Final[int] = 512
"""Largest Hilbert-space dimension densified without an explicit opt-in.

Every routine that materializes an ``O(N**2)`` reference matrix defaults to this
limit, so raising or lowering the guard is a one-line change here rather than a
hunt for scattered literals.

The limit is a deliberate guard, not a capability boundary: pass a larger
``dense_limit`` when the ``O(N**2)`` memory and ``O(N**3)`` time are acceptable.
Measured cost of one :func:`eigenstates` call on this machine, complex128:

===== ========== ==========
``N`` wall time  memory
===== ========== ==========
512   ~0.5 s     ~4 MiB
1024  ~2.2 s     ~17 MiB
2048  ~12 s      ~67 MiB
===== ========== ==========

The default sits at 512 because that is the smallest size at which circular
level statistics are worth computing at all; the previous value of 256 made
every spectral-statistics user hit the guard on their first call.
"""

SymmetrySector: TypeAlias = Literal["even", "odd"]
_SYMMETRY_SECTORS: tuple[str, ...] = get_args(SymmetrySector)
_INVOLUTION_TOLERANCE: Final[float] = 1e-12
"""Bound on ``||S**2 - I||_F / sqrt(N)`` accepted by :func:`desymmetrize`.

The shipped symmetries are signed permutation matrices -- entries in
``{-1, 0, 1}`` -- whose defect is identically zero either way, since ``(+-1)**2``
is exactly ``1``. So this only leaves room for a caller-assembled involution that
rounds at the last bits.
"""


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


def unitarity_defect(
    model_or_matrix: QuantumMap | ArrayLike,
    *,
    dense_limit: int = DEFAULT_DENSE_LIMIT,
) -> float:
    """Return ``||U.H U - I||_F / sqrt(N)`` from a bounded dense representation.

    Accepts either a square dense operator array or a :class:`QuantumMap`; an
    argument that is neither is rejected with a :class:`ValidationError` naming
    both accepted forms. An array argument is used as given and never measured
    against ``dense_limit``, which only guards materializing a model.

    ``dense_limit`` defaults to :data:`DEFAULT_DENSE_LIMIT`. The limit is a
    deliberate guard, not a capability boundary: raise it only knowing that a
    dense reference costs ``O(N**2)`` memory and ``O(N**3)`` time. Measured
    ``complex128`` cost is about 0.5 s / 4 MiB at ``N = 512``, 2.2 s / 17 MiB at
    ``N = 1024``, and 12 s / 67 MiB at ``N = 2048``.
    """
    dense = _dense_representation(model_or_matrix, dense_limit=dense_limit)
    return _dense_unitarity_defect(dense)


def eigenstates(model: QuantumMap, *, dense_limit: int = DEFAULT_DENSE_LIMIT) -> EigenstateResult:
    """Return circularly sorted dense-reference eigenphases and eigenstates.

    ``dense_limit`` defaults to :data:`DEFAULT_DENSE_LIMIT`. The limit is a
    deliberate guard, not a capability boundary: raise it only knowing that a
    dense reference costs ``O(N**2)`` memory and ``O(N**3)`` time. Measured
    ``complex128`` cost is about 0.5 s / 4 MiB at ``N = 512``, 2.2 s / 17 MiB at
    ``N = 1024``, and 12 s / 67 MiB at ``N = 2048``.
    """
    validate_protocol(model, QuantumMap, name="model")
    dense = _dense_representation(model, dense_limit=dense_limit)
    defect = _dense_unitarity_defect(dense)
    if defect > 1e-12:
        raise NumericalError(f"unitary defect {defect} exceeds eigensystem tolerance 1e-12")
    eigenvalues, vectors = np.linalg.eig(dense)
    vectors = _normalize_columns(np.asarray(vectors, dtype=np.complex128))
    phases = wrap_into_half_open(
        np.asarray(np.angle(eigenvalues), dtype=np.float64),
        lower=-np.pi,
        upper=np.pi,
    )
    order = np.argsort(phases, kind="stable")
    phases = np.asarray(phases[order], dtype=np.float64)
    vectors = vectors[:, order]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.complex128)
    residuals = _eigenpair_residuals(model, vectors, eigenvalues)
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


def eigenphases(model: QuantumMap, *, dense_limit: int = DEFAULT_DENSE_LIMIT) -> AnalysisResult:
    """Return sorted phases while retaining eigensystem residuals and metadata.

    Eigenvectors are computed as well, because the returned metadata carries
    per-eigenpair residuals as a convergence diagnostic. This routine therefore
    costs the same as :func:`eigenstates`; there is deliberately no
    eigenvalue-only fast path.

    ``dense_limit`` defaults to :data:`DEFAULT_DENSE_LIMIT`. The limit is a
    deliberate guard, not a capability boundary: raise it only knowing that a
    dense reference costs ``O(N**2)`` memory and ``O(N**3)`` time. Measured
    ``complex128`` cost is about 0.5 s / 4 MiB at ``N = 512``, 2.2 s / 17 MiB at
    ``N = 1024``, and 12 s / 67 MiB at ``N = 2048``.
    """
    validate_protocol(model, QuantumMap, name="model")
    result = eigenstates(model, dense_limit=dense_limit)
    return AnalysisResult(
        "eigenphases",
        result.eigenphases,
        residuals=result.residuals,
        metadata=result.metadata,
    )


def desymmetrize(
    eigensystem: EigenstateResult,
    symmetry: ArrayLike,
    *,
    sector: SymmetrySector,
    tolerance: float = 1e-10,
) -> EigenstateResult:
    """Split a symmetry-resolved eigensystem into one eigenvalue-of-``S`` sector.

    **Desymmetrize before comparing anything to random-matrix theory.** RMT
    ensembles describe a single irreducible block; superposing two independent
    blocks dilutes level repulsion and pulls every statistic toward Poisson. The
    raw number is not a weak result, it is the wrong quantity.

    **Desymmetrizing is necessary but not sufficient.** Measured mean adjacent
    gap ratio of the Saraceno baker map against a COE reference of 0.5307, with
    the standard error of the sector mean:

    ======== ======== ================= =================
    ``N``    raw      even sector       odd sector
    ======== ======== ================= =================
    700      0.4208   0.5211 +- 0.0134  0.5324 +- 0.0134
    802      0.4205   0.5386 +- 0.0129  0.5568 +- 0.0132
    900      0.4112   0.5093 +- 0.0120  0.5310 +- 0.0120
    256      0.4313   0.3806 +- 0.0244  0.4038 +- 0.0238
    512      0.4207   0.4078 +- 0.0177  0.4309 +- 0.0175
    1024     0.4214   0.4389 +- 0.0122  0.4559 +- 0.0122
    ======== ======== ================= =================

    At a generic even ``N`` the sectors land on the COE value within one or two
    standard errors, which is the whole point of the routine. At ``N = 2**k``
    they do not: the sectors stay 6 to 8 standard errors low and barely move away
    from the raw 0.42. That is the arithmetic anomaly of power-of-two dimensions
    documented on :class:`~chaos_numerics.quantum.QuantumBakerMap`, not a defect
    of this routine, and no amount of desymmetrizing repairs it. Choose a
    non-power-of-two dimension when the point of the calculation is a comparison
    with RMT. This is also why the worked examples here use ``N = 700`` and
    friends rather than the more tempting ``N = 1024``.

    ``symmetry`` must be an involution (``S @ S == I``) that is Hermitian or
    unitary; both properties together follow from either one plus the
    involution, and anything else is rejected with :class:`ValidationError`.

    For an eigenbasis that is already resolved by ``S`` every expectation value
    ``<psi|S|psi>`` is exactly ``+1`` or ``-1`` -- measured worst deviation
    ``2.2e-15`` for the baker map at ``N = 700``. A deviation above
    ``tolerance`` therefore means the eigensolver returned mixed vectors inside
    a degenerate subspace, and the sign split would be meaningless, so it is a
    :class:`ValidationError` rather than a warning. Recovering a sector from a
    degenerate spectrum needs projection followed by re-diagonalization inside
    each block, which is out of scope for v0.1.

    ``metadata.parameters`` records ``symmetry_sector``, the original
    ``dimension``, the ``sector_dimension``, the expectation tolerance, and the
    worst expectation-value deviation. Keys inherited from ``eigensystem`` that
    describe the shape of the spectrum -- the minimum circular gap and the
    degenerate-pair count -- are recomputed for the sector rather than carried
    over, because a sector has different gaps from the spectrum it came out of.
    """
    if not isinstance(eigensystem, EigenstateResult):
        raise ValidationError(
            f"eigensystem must be an EigenstateResult; got {type(eigensystem).__name__}"
        )
    if sector not in _SYMMETRY_SECTORS:
        candidates = ", ".join(repr(name) for name in _SYMMETRY_SECTORS)
        raise ValidationError(f"invalid sector {sector!r}; expected one of {candidates}")
    limit = _positive_float(tolerance, name="tolerance")
    vectors = eigensystem.eigenstates
    dimension = int(vectors.shape[0])
    operator = as_complex_array(symmetry, name="symmetry", ndim=2, copy=False)
    if operator.shape != (dimension, dimension):
        raise ValidationError(
            f"symmetry must have shape ({dimension}, {dimension}); got {operator.shape}"
        )
    _validate_involution(operator, dimension=dimension)

    applied = operator @ vectors
    expectations = np.einsum("ij,ij->j", vectors.conj(), applied)
    deviation = float(np.max(np.abs(np.abs(expectations) - 1.0))) if expectations.size else 0.0
    if deviation > limit:
        raise ValidationError(
            f"symmetry expectation values deviate from +/-1 by {deviation}, above tolerance "
            f"{limit}. A basis resolved by this symmetry has expectations exactly +/-1; a "
            "larger deviation means the eigensolver mixed vectors inside a degenerate "
            "subspace. Projecting and re-diagonalizing each degenerate block is out of "
            "scope for v0.1, so desymmetrize a non-degenerate spectrum instead, for "
            "example by choosing a dimension without arithmetic degeneracies"
        )
    signs = np.real(expectations)
    selected = signs > 0.0 if sector == "even" else signs < 0.0
    indices = np.flatnonzero(selected)
    phases = np.asarray(eigensystem.eigenphases[indices], dtype=np.float64)
    residuals = np.asarray(eigensystem.residuals[indices], dtype=np.float64)
    convergence = ConvergenceInfo(
        converged=True,
        iterations=0,
        residual=float(np.max(residuals)) if residuals.size else 0.0,
        tolerance=limit,
        history=residuals,
        reason=f"residuals of the {sector} sector carried over from the full eigensystem",
    )
    inherited = dict(eigensystem.metadata.parameters)
    degeneracy_tolerance = _finite_nonnegative(
        inherited.get("degeneracy_tolerance", 1e-10),
        name="inherited degeneracy_tolerance",
    )
    circular_gaps = (
        np.diff(np.concatenate((phases, phases[:1] + 2.0 * np.pi)))
        if phases.size
        else np.empty(0, dtype=np.float64)
    )
    metadata = ExperimentMetadata(
        parameters={
            **inherited,
            "symmetry_sector": sector,
            "dimension": dimension,
            "sector_dimension": int(indices.size),
            "sector_expectation_tolerance": limit,
            "sector_expectation_deviation": deviation,
            "degeneracy_tolerance": degeneracy_tolerance,
            "degenerate_pairs": int(np.count_nonzero(circular_gaps <= degeneracy_tolerance)),
            "minimum_circular_phase_gap": (
                float(np.min(circular_gaps)) if circular_gaps.size else None
            ),
        },
        precision=eigensystem.metadata.precision,
        seed=eigensystem.metadata.seed,
        environment=dict(eigensystem.metadata.environment),
        git_commit=eigensystem.metadata.git_commit,
        warnings=eigensystem.metadata.warnings,
        convergence=convergence,
    )
    return EigenstateResult(
        phases,
        np.asarray(vectors[:, indices], dtype=np.complex128),
        residuals,
        metadata,
    )


def _validate_involution(operator: ComplexArray, *, dimension: int) -> None:
    """Reject symmetries that are not Hermitian-or-unitary involutions."""
    identity = np.eye(dimension, dtype=np.complex128)
    scale = np.sqrt(dimension)
    involution_defect = float(np.linalg.norm(operator @ operator - identity) / scale)
    if involution_defect > _INVOLUTION_TOLERANCE:
        raise ValidationError(
            f"symmetry must be an involution with S @ S == I; ||S**2 - I||_F / sqrt(N) is "
            f"{involution_defect}, above {_INVOLUTION_TOLERANCE}"
        )
    hermitian_defect = float(np.linalg.norm(operator - operator.conj().T) / scale)
    unitary_defect = _dense_unitarity_defect(operator)
    if min(hermitian_defect, unitary_defect) > _INVOLUTION_TOLERANCE:
        raise ValidationError(
            "symmetry must be Hermitian or unitary; measured Hermitian defect "
            f"{hermitian_defect} and unitary defect {unitary_defect}, both above "
            f"{_INVOLUTION_TOLERANCE}"
        )


def _dense_representation(
    model_or_matrix: QuantumMap | ArrayLike,
    *,
    dense_limit: int,
) -> ComplexArray:
    limit = _positive_int(dense_limit, name="dense_limit")
    if not isinstance(model_or_matrix, QuantumMap):
        try:
            dense = as_complex_array(model_or_matrix, name="operator", ndim=2, copy=True)
        except ValidationError as error:
            raise ValidationError(
                "operator must be either a square dense numeric array or an object "
                f"implementing the QuantumMap protocol; got {type(model_or_matrix).__name__} "
                f"({error})"
            ) from error
        if dense.shape[0] != dense.shape[1]:
            raise ValidationError(f"operator must be square; got {dense.shape}")
        return dense
    model = model_or_matrix
    dimension = _positive_int(model.dimension, name="model dimension")
    if dimension > limit:
        megabytes = 16.0 * dimension**2 / 2**20
        raise ValidationError(
            f"dense materialization dimension {dimension} exceeds dense_limit {limit}. "
            f"The guard is deliberate: a dense reference costs O(N**2) memory "
            f"({megabytes:.1f} MiB of complex128 here) and O(N**3) time. Pass "
            f"dense_limit={dimension} to accept that cost, or use a matrix-free path "
            f"such as evolve(..., method='fft') if the full spectrum is not required"
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


def _eigenpair_residuals(
    model: QuantumMap,
    vectors: ComplexArray,
    eigenvalues: ComplexArray,
) -> FloatArray:
    """Return normalized ``||U v - lambda v||`` for every column of ``vectors``.

    ``QuantumMap.apply`` is contractually batched over a leading axis, so all
    eigenvectors are applied in one call. Models that only handle single states
    fall back to a per-column loop.
    """
    try:
        applied = np.asarray(model.apply(vectors.T), dtype=np.complex128).T
        if applied.shape != vectors.shape:
            raise ValueError("batched apply did not preserve shape")
    except (TypeError, ValueError, IndexError):
        applied = np.empty_like(vectors)
        for index in range(vectors.shape[1]):
            applied[:, index] = model.apply(vectors[:, index])
    numerators = np.linalg.norm(applied - eigenvalues[None, :] * vectors, axis=0)
    denominators = np.maximum(
        np.finfo(np.float64).tiny,
        (1.0 + np.abs(eigenvalues)) * np.linalg.norm(vectors, axis=0),
    )
    return np.asarray(numerators / denominators, dtype=np.float64)


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
            np.linalg.norm(_commutator(dense, symmetry)) / np.sqrt(dense.shape[0])
        )
    return defects


def _commutator(dense: ComplexArray, symmetry: ComplexArray) -> ComplexArray:
    """Return ``[U, S]`` using index permutation when ``S`` is a permutation.

    ``U @ S`` and ``S @ U`` are exact column and row permutations when ``S`` is a
        0/1 permutation matrix. Taking that route turns two ``O(N**3)`` GEMMs into
        ``O(N**2)`` gathers and is bit-for-bit identical, because the skipped products
        are exact zeros. Anything else falls back to the general matrix products.

        Most shipped symmetries qualify, but not all: the ``KickedRotor`` parity at
        ``BoundaryPhases(0, 1/2)`` is a *signed* permutation, with entries in
        ``{-1, 0, 1}``, and takes the GEMM route. That is correct rather than
        unfortunate -- the reflection at that twist genuinely needs the sign, which is
        why it commutes exactly where a bare permutation does not (defect ``6.8e-14``
        against ``0.249``) -- and it is worth naming here because the fast path is
        sometimes described as covering everything the library ships.
    """
    row_to_column = _permutation_of(symmetry)
    if row_to_column is None:
        return np.asarray(dense @ symmetry - symmetry @ dense, dtype=np.complex128)
    inverse = np.empty_like(row_to_column)
    inverse[row_to_column] = np.arange(row_to_column.size)
    return np.asarray(dense[:, inverse] - dense[row_to_column, :], dtype=np.complex128)


def _permutation_of(symmetry: ComplexArray) -> IndexArray | None:
    """Return ``i -> j`` with ``S[i, j] == 1`` when ``S`` is a permutation matrix."""
    size = symmetry.shape[0]
    rows, columns = np.nonzero(symmetry)
    if rows.size != size or not np.array_equal(rows, np.arange(size)):
        return None
    if not np.array_equal(np.sort(columns), np.arange(size)):
        return None
    if not bool(np.all(symmetry[rows, columns] == 1.0)):
        return None
    return np.asarray(columns, dtype=np.intp)


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


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite non-negative number; got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValidationError(f"{name} must be a finite non-negative number; got {value!r}")
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


__all__ = [
    "DEFAULT_DENSE_LIMIT",
    "DenseUnitary",
    "SymmetrySector",
    "desymmetrize",
    "eigenphases",
    "eigenstates",
    "unitarity_defect",
]

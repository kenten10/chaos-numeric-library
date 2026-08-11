from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping
from typing import get_args, get_type_hints

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]

from chaos_numerics.classical import StandardMap, largest_lyapunov_exponent
from chaos_numerics.core import (
    ChaosNumericsError,
    ChaosNumericsWarning,
    ConvergenceError,
    ConvergenceWarning,
    NumericalError,
    NumericalWarning,
    QuantumMap,
    ReproducibilityWarning,
    ValidationError,
)
from chaos_numerics.quantum import (
    DEFAULT_DENSE_LIMIT,
    BoundaryPhases,
    DenseUnitary,
    KickedRotor,
    QuantumBakerMap,
    QuantumBasis,
    QuantumCatMap,
    basis_state,
    coherent_state,
    desymmetrize,
    eigenphases,
    eigenstates,
    evolve,
    loschmidt_echo,
    normalize_state,
    otoc,
    quantum_state,
    unitarity_defect,
    weyl_translations,
)
from chaos_numerics.spectral import adjacent_gap_ratios, prepare_eigenphases, unfold


def complexes(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    return np.asarray(value, dtype=np.complex128)


def hadamard() -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    return np.asarray(
        complexes([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
        dtype=np.complex128,
    )


def twisted_shift() -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    """Return a unitary that is neither real, symmetric, nor Hermitian.

    ``hadamard()`` is real symmetric, so a test written with it passes even if
    the implementation confuses ``M`` with ``M.T`` or with ``M.conj().T``. This
    phased cyclic shift distinguishes all three.
    """
    size = 3
    matrix = np.zeros((size, size), dtype=np.complex128)
    indices = np.arange(size)
    matrix[indices, (indices + 1) % size] = np.exp(1j * np.asarray([0.3, 1.7, -2.1]))
    return matrix


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
    matrix = twisted_shift()
    model = DenseUnitary(
        matrix,
        basis="position",
        boundary_phases=BoundaryPhases(0.5, 0.25),
        name="twisted_shift",
    )
    state = normalize_state(complexes([1.0 + 1.0j, 2.0, -0.5j]))
    batch = np.stack((state, basis_state(dimension=3, index=0)))

    # The reference matrix is asymmetric and complex, so each of these would
    # fail if M, M.T, or M.conj().T were interchanged.
    assert not np.allclose(matrix, matrix.T)
    assert not np.allclose(matrix, matrix.conj().T)
    np.testing.assert_allclose(model.apply(state), matrix @ state, atol=1e-15)
    np.testing.assert_allclose(model.apply(batch), batch @ matrix.T, atol=1e-15)
    linear = model.as_linear_operator()
    np.testing.assert_allclose(linear.matvec(state), model.apply(state), atol=1e-15)
    np.testing.assert_allclose(linear.rmatvec(state), matrix.conj().T @ state, atol=1e-15)
    assert isinstance(model, QuantumMap)
    assert not model.matrix.flags.writeable
    # ``defect <= 1e-12`` would be tautological: the constructor already refuses
    # anything above its tolerance. Pin the published normalization instead.
    independent = float(
        np.linalg.norm(matrix.conj().T @ matrix - np.eye(3, dtype=np.complex128)) / np.sqrt(3.0)
    )
    assert model.defect == pytest.approx(independent, abs=1e-16)


def test_dense_and_matrix_free_evolve_match_for_scalar_batch_and_history() -> None:
    dense = DenseUnitary(hadamard())
    matrix_free = MatrixFreeUnitary(hadamard())
    batch = np.stack(
        (
            basis_state(dimension=2, index=0),
            normalize_state(complexes([1.0, 1.0j])),
        )
    )

    dense_run = evolve(dense, batch, steps=7, method="dense")
    free_run = evolve(matrix_free, batch, steps=7, method="matrix_free")
    stored = evolve(matrix_free, batch, steps=7, return_history=True)

    # Measured differences are at the 1e-16 level for all three comparisons.
    np.testing.assert_allclose(dense_run.final_state, free_run.final_state, atol=1e-14)
    # The return type does not depend on ``return_history``; only ``history`` does.
    assert dense_run.history is None
    assert dense_run.batch_shape == (2,) and dense_run.dimension == 2
    assert stored.history is not None
    assert stored.history.shape == (2, 8, 2)
    assert stored.time_count == 8
    assert dense_run.time_count == 0
    np.testing.assert_allclose(stored.history[..., -1, :], stored.final_state, atol=0.0)
    np.testing.assert_allclose(stored.final_state, free_run.final_state, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(stored.history, axis=-1), np.ones((2, 8)), atol=1e-14)


def test_unitarity_defect_matches_its_documented_normalization() -> None:
    """Pin ``||U.H U - I||_F / sqrt(N)`` on a deliberately non-unitary operator.

    Asserting ``defect <= 1e-12`` for a unitary input is tautological because
    ``DenseUnitary`` already enforces exactly that at construction. Feeding
    ``2 I`` of size four gives ``||4I - I||_F / 2 = 3`` in closed form, which
    fixes both the ``sqrt(N)`` normalization and the overall scale.
    """
    assert unitarity_defect(2.0 * np.eye(4, dtype=np.complex128)) == pytest.approx(3.0, rel=1e-14)
    assert unitarity_defect(twisted_shift()) <= 1e-15
    assert unitarity_defect(MatrixFreeUnitary(twisted_shift())) <= 1e-15

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
    # An asymmetric complex unitary is required here: the column-by-column
    # materialization would also pass with a transposed reference if the test
    # operator were real symmetric.
    matrix = twisted_shift()
    dense = eigenstates(DenseUnitary(matrix))
    matrix_free = eigenstates(MatrixFreeUnitary(matrix))

    np.testing.assert_allclose(matrix_free.eigenphases, dense.eigenphases, atol=1e-14)
    assert float(np.max(matrix_free.residuals)) <= 1e-12
    # The shift's spectrum is the cube roots of the product of its phases, so a
    # transposed materialization would be caught by the phases themselves.
    total = float(np.sum([0.3, 1.7, -2.1]))
    expected = np.sort(np.angle(np.exp(1j * (total + 2.0 * np.pi * np.arange(3)) / 3.0)))
    np.testing.assert_allclose(dense.eigenphases, expected, atol=1e-14)


class ExpandingMap(MatrixFreeUnitary):
    def __init__(self) -> None:
        super().__init__(2.0 * np.eye(2, dtype=np.complex128))


def test_evolve_rejects_norm_drift_and_invalid_parameters() -> None:
    state = basis_state(dimension=2, index=0)
    with pytest.raises(NumericalError, match="norm drift"):
        evolve(ExpandingMap(), state, steps=1)
    with pytest.raises(ValidationError, match="non-negative integer"):
        evolve(DenseUnitary(hadamard()), state, steps=True)
    # ``method="fft"`` is a valid literal, so it used to hit the "does not
    # provide apply_fft" branch and only match this pattern by accident. An
    # unknown method must name the accepted values.
    with pytest.raises(
        ValidationError,
        match=r"invalid evolution method 'bogus'; expected one of .*'dense'.*",
    ):
        evolve(DenseUnitary(hadamard()), state, steps=1, method="bogus")  # type: ignore[arg-type]


def test_public_entry_points_reject_non_protocol_objects() -> None:
    """Protocol violations must surface as ValidationError, not AttributeError.

    ``docs/design/public-api.md`` section 7.4 requires ValidationError for
    invalid input, and a bare ``AttributeError: 'object' object has no attribute
    'dimension'`` from inside an algorithm does not tell the caller what to fix.
    """
    stranger = object()
    with pytest.raises(ValidationError, match="model must implement the QuantumMap protocol"):
        evolve(stranger, basis_state(dimension=2, index=0), steps=1)  # type: ignore[arg-type]
    for entry in (eigenstates, eigenphases):
        with pytest.raises(ValidationError, match="missing apply, as_linear_operator, dimension"):
            entry(stranger)  # type: ignore[arg-type]
    # ``unitarity_defect`` accepts a bare matrix too, so its message must name
    # both accepted forms rather than complaining about an object dtype.
    with pytest.raises(
        ValidationError,
        match="either a square dense numeric array or an object implementing the QuantumMap",
    ):
        unitarity_defect(stranger)  # type: ignore[arg-type]


def test_public_exception_hierarchy_is_stable() -> None:
    """Downstream code catches these base classes, so pin the whole hierarchy."""
    assert issubclass(ValidationError, ChaosNumericsError)
    assert issubclass(NumericalError, ChaosNumericsError)
    assert issubclass(ConvergenceError, NumericalError)
    assert issubclass(ConvergenceError, ChaosNumericsError)
    assert issubclass(ChaosNumericsError, Exception)
    # Scripts written before this library existed catch the builtin ABCs.
    assert issubclass(ValidationError, ValueError)
    assert issubclass(NumericalError, ArithmeticError)
    assert not issubclass(ValidationError, NumericalError)

    assert issubclass(NumericalWarning, ChaosNumericsWarning)
    assert issubclass(ConvergenceWarning, NumericalWarning)
    assert issubclass(ReproducibilityWarning, ChaosNumericsWarning)
    assert issubclass(ChaosNumericsWarning, UserWarning)


def test_default_dense_limit_is_one_named_constant_used_by_every_entry_point() -> None:
    """The guard is one constant, not four literals that can drift apart.

    Raising the limit is a deliberate act, so it must be raised in exactly one
    place. 512 is the smallest dimension at which circular level statistics are
    worth computing, and the previous value of 256 made every spectral user hit
    the guard on their first call.
    """
    assert DEFAULT_DENSE_LIMIT == 512
    for entry in (eigenstates, eigenphases, unitarity_defect, evolve):
        default = inspect.signature(entry).parameters["dense_limit"].default
        assert default is DEFAULT_DENSE_LIMIT, entry.__name__

    # 512 still guards: one dimension above the default is refused, and the
    # message names both escape hatches rather than only the failure.
    oversized = MatrixFreeUnitary(np.eye(513, dtype=np.complex128))
    with pytest.raises(ValidationError, match=r"513 exceeds dense_limit 512"):
        eigenstates(oversized)
    with pytest.raises(ValidationError, match=r"Pass dense_limit=513 to accept that cost"):
        unitarity_defect(oversized)
    assert unitarity_defect(oversized, dense_limit=513) <= 1e-15


def test_desymmetrize_splits_the_baker_map_into_two_equal_parity_sectors() -> None:
    """Sector dimensions must partition the spectrum exactly, with no leftovers."""
    model = QuantumBakerMap(128)
    eigensystem = eigenstates(model)
    parity = model.symmetry_operators["parity"]

    even = desymmetrize(eigensystem, parity, sector="even")
    odd = desymmetrize(eigensystem, parity, sector="odd")

    # The Saraceno baker map at even N has exactly N/2 states of each parity.
    assert (even.count, odd.count) == (64, 64)
    assert even.count + odd.count == eigensystem.count == 128
    assert even.eigenstates.shape == (128, 64)
    assert float(np.max(even.residuals)) <= 1e-12

    # Every sector eigenphase is one of the original eigenphases, unmodified.
    for sector in (even, odd):
        matched = np.abs(sector.eigenphases[:, None] - eigensystem.eigenphases[None, :])
        assert np.all(np.min(matched, axis=1) == 0.0)
    combined = np.sort(np.concatenate((even.eigenphases, odd.eigenphases)))
    np.testing.assert_array_equal(combined, eigensystem.eigenphases)

    parameters = even.metadata.parameters
    assert parameters["symmetry_sector"] == "even"
    assert parameters["dimension"] == 128
    assert parameters["sector_dimension"] == 64
    assert parameters["sector_expectation_tolerance"] == 1e-10
    # A parity-resolved eigenbasis has expectations exactly +/-1; measured worst
    # deviation is 8.9e-16 at N=128 and 2.3e-15 at N=1024.
    deviation = parameters["sector_expectation_deviation"]
    assert isinstance(deviation, float)
    assert deviation <= 1e-14
    # Gap statistics describe the sector, not the spectrum it was cut out of, so
    # the inherited values must have been recomputed.
    sector_gap = parameters["minimum_circular_phase_gap"]
    parent_gap = eigensystem.metadata.parameters["minimum_circular_phase_gap"]
    assert isinstance(sector_gap, float) and isinstance(parent_gap, float)
    # Removing half the levels can only widen the smallest gap, and here it does.
    assert sector_gap > parent_gap
    assert parameters["degenerate_pairs"] == 0
    assert parameters["degeneracy_tolerance"] == 1e-10


def test_desymmetrize_moves_the_gap_ratio_toward_the_coe_reference() -> None:
    """The whole point of desymmetrizing: raw statistics answer the wrong question.

    Superposing two independent parity blocks dilutes level repulsion, so the raw
    spectrum sits well below COE. Measured at N=512, K=10 with parity-preserving
    twists: raw 0.4153, even 0.5041, odd 0.5397, against a COE reference 0.5307.
    """
    coe_reference = 0.5307
    model = KickedRotor(512, 10.0)
    eigensystem = eigenstates(model, dense_limit=512)
    parity = model.symmetry_operators["parity"]

    def mean_gap_ratio(phases: ArrayLike, label: str) -> float:
        prepared = prepare_eigenphases(phases, symmetry_sector=label)
        ratios = adjacent_gap_ratios(unfold(prepared, method="mean")).values
        return float(np.mean(np.asarray(ratios)))

    raw = mean_gap_ratio(eigensystem.eigenphases, "mixed parity sectors")
    even = desymmetrize(eigensystem, parity, sector="even")
    odd = desymmetrize(eigensystem, parity, sector="odd")
    even_ratio = mean_gap_ratio(even.eigenphases, "parity-even")
    odd_ratio = mean_gap_ratio(odd.eigenphases, "parity-odd")

    assert even.count + odd.count == 512
    assert raw == pytest.approx(0.4153, abs=0.01)
    assert even_ratio == pytest.approx(0.5041, abs=0.01)
    assert odd_ratio == pytest.approx(0.5397, abs=0.01)
    # Both sectors are closer to COE than the raw spectrum is: that is the claim.
    assert abs(even_ratio - coe_reference) < abs(raw - coe_reference)
    assert abs(odd_ratio - coe_reference) < abs(raw - coe_reference)


def test_desymmetrize_rejects_operators_that_are_not_involutions() -> None:
    model = QuantumBakerMap(8)
    eigensystem = eigenstates(model)
    parity = model.symmetry_operators["parity"]

    # A cyclic shift is unitary but has S**3 = I, not S**2 = I.
    shift = np.roll(np.eye(8, dtype=np.complex128), 1, axis=0)
    with pytest.raises(ValidationError, match="must be an involution"):
        desymmetrize(eigensystem, shift, sector="even")
    # 2 I squares to 4 I, so the involution check must fire before anything else.
    with pytest.raises(ValidationError, match="must be an involution"):
        desymmetrize(eigensystem, 2.0 * np.eye(8, dtype=np.complex128), sector="even")
    # Involutive but neither Hermitian nor unitary.
    skewed = np.eye(8, dtype=np.complex128)
    skewed[0, 1] = 3.0
    skewed[1, 1] = -1.0
    with pytest.raises(ValidationError, match="must be Hermitian or unitary"):
        desymmetrize(eigensystem, skewed, sector="even")

    with pytest.raises(ValidationError, match="symmetry must have shape"):
        desymmetrize(eigensystem, np.eye(4, dtype=np.complex128), sector="even")
    with pytest.raises(ValidationError, match="invalid sector"):
        desymmetrize(eigensystem, parity, sector="both")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="must be an EigenstateResult"):
        desymmetrize(parity, parity, sector="even")  # type: ignore[arg-type]


def test_desymmetrize_is_meaningless_when_the_model_publishes_no_symmetry() -> None:
    """A generic Bloch twist breaks parity, so there is nothing to project onto.

    ``symmetry_operators`` is empty rather than reporting a large commutator
    defect, which is the signal that this spectrum is already a single block and
    must be compared against CUE without desymmetrization. Handing the parity
    permutation over anyway is rejected: the eigenbasis is not resolved by it, so
    the sign of every expectation value is an artifact of the eigensolver.
    """
    model = KickedRotor(64, 10.0, boundary_phases=BoundaryPhases(0.25, 0.13))

    assert model.symmetry_operators == {}

    eigensystem = eigenstates(model, dense_limit=64)
    indices = np.arange(64)
    parity = np.zeros((64, 64), dtype=np.complex128)
    parity[np.mod(-indices, 64), indices] = 1.0
    with pytest.raises(ValidationError, match=r"deviate from \+/-1"):
        desymmetrize(eigensystem, parity, sector="even")


def test_public_type_aliases_are_reachable_from_the_quantum_namespace() -> None:
    """Aliases that appear in a public signature are part of the public surface.

    ``docs/design/public-api.md`` states the rule and ``py.typed`` is what makes
    it bite: a caller who wraps ``evolve`` or ``desymmetrize`` has to be able to
    name the type of the argument they are forwarding. Before these entries
    existed the only import path was the defining submodule, which is not part of
    the documented surface.
    """
    import chaos_numerics.quantum as quantum

    for name in ("CatMatrix", "EvolutionMethod", "SymmetrySector"):
        assert name in quantum.__all__
        assert hasattr(quantum, name)

    # The aliases really are the annotations in those signatures, not lookalikes.
    hints = get_type_hints(evolve)
    assert hints["method"] is quantum.EvolutionMethod
    assert get_type_hints(desymmetrize)["sector"] is quantum.SymmetrySector
    assert get_args(quantum.SymmetrySector) == ("even", "odd")
    assert get_args(quantum.EvolutionMethod) == ("auto", "dense", "matrix_free", "fft")
    # ``get_type_hints`` re-evaluates the deferred annotation, so a generic alias
    # compares equal rather than identical; ``Literal`` instances are interned.
    assert get_type_hints(QuantumCatMap.__init__)["matrix"] == quantum.CatMatrix
    assert get_args(quantum.CatMatrix) == (tuple[int, int], tuple[int, int])


def _classical_lyapunov(
    kick_strength: float, *, orbits: int, steps: int = 5_000, seed: int = 20260811
) -> float:
    """Return the phase-space mean standard-map exponent over ``orbits`` orbits.

    **A single orbit cannot be pinned across platforms, and trying to do so is
    what broke this test in CI.** The standard map is chaotic, so any difference
    in rounding -- a different libm ``sin``, a different SIMD path -- is amplified
    by ``exp(lambda * n)`` and the orbit decorrelates within a few hundred steps.
    After 5000 steps two machines started from the same initial condition are
    following entirely different orbits, and their finite-time exponents differ by
    far more than any rounding tolerance: the earlier single-orbit version of this
    helper returned 1.00604 on the development machine and 0.98407 on the CI Linux
    and macOS runners, a 2.2% gap that broke an ``abs=0.02`` bound on six of seven
    test jobs while passing on Windows by luck.

    Only an ensemble mean is reproducible, and only to its own sampling error, so
    the caller gets a mean and the assertions below are sized from the measured
    spread across orbit sets rather than from rounding. Measured over eight seeds:

    ============ ========== =================== ==============================
    ``K``        ``orbits`` mean spread          note
    ============ ========== =================== ==============================
    ``5.0``      32         ``0.9416``-``0.9807`` mixed phase space; 1.2% of
                                                orbits sit in islands with
                                                ``lambda`` near zero, so the
                                                mean converges slowly
    ``10.0``     24         ``1.6182``-``1.6268`` globally chaotic, no islands
    ============ ========== =================== ==============================

    ``largest_lyapunov_exponent`` reports a ``ConvergenceWarning`` whenever its
    two equal-length windows still disagree, which at these run lengths depends on
    the orbit rather than on anything the OTOC tests are about. The warning is
    silenced here and nowhere else; the classical estimator has its own
    convergence tests in ``tests/test_lyapunov.py``, which pin an exponent at a
    *fixed point* where the Jacobian is constant and the answer is therefore
    exact and orbit-independent -- the pattern this helper should have followed.
    """
    generator = np.random.default_rng(seed)
    total = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(orbits):
            result = largest_lyapunov_exponent(
                StandardMap(kick_strength=kick_strength),
                initial_state=tuple(generator.random(2)),
                steps=steps,
                transient=500,
            )
            total += float(np.ravel(result.values)[-1])
    return total / orbits


def _log_slope(
    values: np.ndarray[tuple[int, ...], np.dtype[np.float64]], lo: int, hi: int
) -> float:
    times = np.arange(lo, hi + 1, dtype=np.float64)
    slope, _ = np.polyfit(times, np.log(values[lo : hi + 1]), 1)
    return float(slope)


def test_weyl_translations_are_unitary_and_satisfy_the_weyl_relation() -> None:
    """Pin the algebra the OTOC default operators rest on, before using them.

    ``T_q T_p = exp(2 pi i / N) T_p T_q`` is what makes ``C(0)`` have a closed
    form, so it is fixed here rather than inferred from the correlator.
    """
    worst_unitary = 0.0
    worst_relation = 0.0
    for dimension in (2, 3, 8, 16, 64, 128, 257, 512):
        for phases in (BoundaryPhases(), BoundaryPhases(0.5, 0.5), BoundaryPhases(0.25, 0.13)):
            clock, shift = weyl_translations(dimension, boundary_phases=phases)
            scale = np.sqrt(dimension)
            identity = np.eye(dimension, dtype=np.complex128)
            for operator in (clock, shift):
                assert operator.shape == (dimension, dimension)
                assert operator.dtype == np.complex128
                worst_unitary = max(
                    worst_unitary,
                    float(np.linalg.norm(operator.conj().T @ operator - identity) / scale),
                )
            residual = clock @ shift - np.exp(2j * np.pi / dimension) * (shift @ clock)
            worst_relation = max(worst_relation, float(np.linalg.norm(residual) / scale))

    # Both defects are pure rounding in ``exp(2 pi i (j + alpha) / N)``, the only
    # inexact factor in either matrix, so the bounds are stated in units of the
    # machine epsilon rather than as absolute numbers. Measured over this grid:
    # unitarity 0.38 eps on numpy 2.5 but 0.72 eps on numpy 1.26, which is why a
    # 1e-16 bound passed on one and failed on the other; the Weyl relation is
    # 2.43 eps on both. numpy 1.26's complex128 elementwise multiply is not even
    # reproducible call to call (see docs/design/numerical-standards.md), so any
    # bound tighter than a few eps here is pinning a NumPy build, not this code.
    epsilon = float(np.finfo(np.float64).eps)
    assert worst_unitary <= 2.0 * epsilon
    assert worst_relation <= 4.0 * epsilon

    # The shift really is the twisted cyclic one, not a bare permutation.
    twisted = weyl_translations(4, boundary_phases=BoundaryPhases(0.0, 0.5))[1]
    assert twisted[0, 3] == pytest.approx(-1.0)
    assert np.array_equal(np.abs(twisted), np.roll(np.eye(4), 1, axis=0))


def test_weyl_translations_return_read_only_arrays_and_validate_arguments() -> None:
    clock, shift = weyl_translations(6)
    assert not clock.flags.writeable
    assert not shift.flags.writeable
    with pytest.raises(ValidationError, match="dimension must be positive"):
        weyl_translations(0)
    with pytest.raises(ValidationError, match="non-negative integer"):
        weyl_translations(4.0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="BoundaryPhases"):
        weyl_translations(4, boundary_phases=0.5)  # type: ignore[arg-type]


def test_otoc_starts_at_zero_for_one_operator_with_itself() -> None:
    """``[A, A] == 0``, so ``C(0)`` must be bit-exactly zero, not merely small."""
    clock, _ = weyl_translations(32)
    result = otoc(KickedRotor(32, 7.0), steps=3, operator_a=clock, operator_b=clock)
    assert result.name == "otoc"
    assert result.values[0] == 0.0
    # It does not stay zero: ``A(t)`` stops commuting with ``A`` immediately.
    assert result.values[1] > 1e-3


def test_otoc_initial_value_matches_the_closed_form_from_the_weyl_relation() -> None:
    """``[T_q, T_p] = (e^(2 pi i / N) - 1) T_p T_q`` gives ``C(0) = 4 sin(pi/N)**2``.

    Because ``T_p T_q`` is unitary, ``[A, B].H [A, B]`` is a multiple of the
    identity, so the closed form holds for the infinite-temperature trace *and*
    for every pure state. Both are checked.
    """
    worst = 0.0
    for dimension in (4, 8, 16, 32, 64, 128, 256):
        expected = 4.0 * np.sin(np.pi / dimension) ** 2
        thermal = otoc(KickedRotor(dimension, 7.0), steps=0)
        worst = max(worst, abs(float(thermal.values[0]) - expected))
        state = normalize_state(np.exp(-((np.arange(dimension) - dimension / 3.0) ** 2) / 8.0) + 0j)
        pure = otoc(KickedRotor(dimension, 7.0), steps=0, state=state)
        worst = max(worst, abs(float(pure.values[0]) - expected))
    assert worst <= 1e-15  # measured 2.22e-16
    assert otoc(KickedRotor(64, 7.0), steps=0).metadata.parameters["expectation"] == (
        "infinite_temperature"
    )


def test_otoc_matches_a_brute_force_heisenberg_reference() -> None:
    """Recompute ``A(t) = U^(-t) A U^t`` from scratch at each ``t``, not iteratively.

    Every other test here checks a *property* of the correlator -- a closed form at
    ``t = 0``, a bound, a ratio between regimes. None of them would catch the
    recursion accumulating the wrong operator, because a wrong-but-consistent
    recursion still starts at the right place, stays bounded, and grows. This
    rebuilds each ``A(t)`` with an independent ``matrix_power`` and compares the
    whole curve. A non-trivial pair of boundary phases is used so the twisted
    transform is exercised rather than the untwisted special case.
    """
    dimension, steps = 12, 6
    model = KickedRotor(dimension, 7.0, boundary_phases=BoundaryPhases(0.25, 0.13))
    unitary = model.to_dense()
    operator_a, operator_b = weyl_translations(dimension, boundary_phases=model.boundary_phases)
    reference = []
    for time in range(steps + 1):
        power = np.linalg.matrix_power(unitary, time)
        heisenberg = power.conj().T @ operator_a @ power
        commutator = heisenberg @ operator_b - operator_b @ heisenberg
        reference.append(float(np.sum(np.abs(commutator) ** 2)) / dimension)

    measured = np.asarray(otoc(model, steps=steps).values)
    # Measured worst difference 1.3e-15; the two routes differ only in rounding.
    np.testing.assert_allclose(measured, reference, rtol=0.0, atol=1e-14)
    # The curve is not trivially flat, so the comparison has something to catch:
    # measured C(0) = 4 sin(pi/12)^2 = 0.2679 rising to 2.427, a factor of 9.1.
    assert float(np.max(measured)) / float(measured[0]) > 5.0


def test_loschmidt_echo_matches_a_brute_force_reference_and_is_symmetric() -> None:
    """Recompute both orbits with ``matrix_power`` and swap the two models.

    ``M`` is built from ``|<psi| U_b^(-t) U_a^t |psi>|**2``, whose modulus is
    unchanged by exchanging the two operators -- the overlap becomes its own
    conjugate. That symmetry is a property of the definition, not of the
    implementation, so an implementation that quietly evolved one of the two states
    for the wrong number of steps would break it.

    The swap is compared at rounding level rather than bit-exactly. On NumPy 2.5 it
    *is* bit-exact, because the two runs are the same code on swapped inputs, but
    the oldest supported NumPy does not make complex elementwise arithmetic
    reproducible even call to call: measured worst difference 1.8e-15 on NumPy
    1.26.4. See the reproducibility section of docs/design/numerical-standards.md.
    """
    dimension, steps = 10, 8
    reference_model = KickedRotor(dimension, 5.0, boundary_phases=0.2)
    perturbed_model = KickedRotor(dimension, 5.05, boundary_phases=0.2)
    state = coherent_state(dimension=dimension, position=0.4, momentum=0.1)

    first, second = reference_model.to_dense(), perturbed_model.to_dense()
    expected = []
    for time in range(steps + 1):
        expected.append(
            abs(
                np.vdot(
                    np.linalg.matrix_power(second, time) @ state,
                    np.linalg.matrix_power(first, time) @ state,
                )
            )
            ** 2
        )

    measured = np.asarray(
        loschmidt_echo(reference_model, perturbed_model, state, steps=steps).values
    )
    # Measured worst difference 5.1e-15.
    np.testing.assert_allclose(measured, expected, rtol=0.0, atol=1e-13)
    swapped = np.asarray(
        loschmidt_echo(perturbed_model, reference_model, state, steps=steps).values
    )
    np.testing.assert_allclose(measured, swapped, rtol=0.0, atol=1e-14)
    # The echo has actually decayed, so the comparison is not against a flat one.
    # ``delta K = 0.05`` over eight steps at N = 10 leaves M(8) = 0.9609.
    assert 1.0 - float(measured[-1]) > 0.03


def test_loschmidt_echo_accepts_a_pair_of_different_model_classes() -> None:
    """The contract is the ``QuantumMap`` protocol, not a shared class.

    ``H^2 = I``, so evolving under the identity and under the Hadamard gives
    ``|<0| H^t |0>|**2 = 1, 1/2, 1, 1/2, ...`` in closed form, which also fixes
    that ``t`` indexes the same power in both orbits.
    """
    result = loschmidt_echo(
        DenseUnitary(np.eye(2, dtype=np.complex128)),
        DenseUnitary(hadamard()),
        basis_state(dimension=2, index=0),
        steps=4,
    )
    np.testing.assert_allclose(result.values, [1.0, 0.5, 1.0, 0.5, 1.0], atol=1e-15)


def test_otoc_is_bounded_and_saturates_near_two() -> None:
    """Bounded by 4 from ``||[A, B]|| <= 2 ||A|| ||B||``; settles at 2 in practice."""
    values = otoc(KickedRotor(128, 10.0), steps=200).values
    assert np.all(values >= 0.0)
    assert np.all(values <= 4.0)
    # Measured max 2.0426 over t <= 200, min 1.9557 for t >= 12, mean 1.9946
    # over 100 <= t <= 200.
    assert float(np.max(values)) == pytest.approx(2.0426, abs=5e-4)
    assert float(np.min(values[12:])) == pytest.approx(1.9557, abs=5e-4)
    assert float(np.mean(values[100:])) == pytest.approx(1.9946, abs=5e-4)


def test_otoc_separates_the_integrable_and_chaotic_regimes() -> None:
    """The reason this function exists: growth appears only when the map is chaotic.

    ``K = 0`` is the free rotor, which is integrable and for which the
    correlator is *exactly* flat -- the free rotation only multiplies ``T_q`` by
    phases. Growth over four steps then spans four orders of magnitude between
    the near-integrable and the globally chaotic rows.
    """
    ratios: dict[float, float] = {}
    for kick in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        values = otoc(KickedRotor(512, kick), steps=8).values
        ratios[kick] = float(values[4] / values[0])
        if kick == 0.0:
            # Measured 8.7e-19: the integrable baseline is flat to rounding.
            assert float(np.max(np.abs(values - values[0]))) <= 1e-17

    # Measured C(4)/C(0) at N = 512.
    assert ratios[0.0] == pytest.approx(1.0, abs=1e-12)
    assert ratios[0.5] == pytest.approx(5.27, rel=2e-3)
    assert ratios[1.0] == pytest.approx(24.87, rel=2e-3)
    assert ratios[2.0] == pytest.approx(263.6, rel=2e-3)
    assert ratios[5.0] == pytest.approx(6727.0, rel=2e-3)
    assert ratios[10.0] == pytest.approx(11760.1, rel=2e-3)
    # Four orders of magnitude between the two regimes is the diagnostic.
    assert ratios[10.0] / ratios[0.5] > 2_000.0


def test_otoc_growth_rate_is_not_twice_the_classical_lyapunov_exponent() -> None:
    """Report the ``C ~ exp(2 lambda t)`` expectation as measured, not as assumed.

    It does not hold at ``hbar_eff = 2 pi / N``. The fitted rate overshoots
    ``2 lambda`` where the finite-time exponent fluctuates -- ``C`` is a second
    moment, so its rate is the order-2 generalized exponent -- and undershoots
    once the Ehrenfest time drops below the fit window. The ratio therefore
    crosses one instead of converging to it, and that is what is pinned.
    """
    rates = {
        kick: _log_slope(otoc(KickedRotor(512, kick), steps=8).values, 1, 4)
        for kick in (2.0, 5.0, 10.0)
    }
    # Measured fitted rates at N = 512 over 1 <= t <= 4.
    assert rates[2.0] == pytest.approx(1.4915, abs=1e-3)
    assert rates[5.0] == pytest.approx(2.1186, abs=1e-3)
    assert rates[10.0] == pytest.approx(1.7794, abs=1e-3)

    # Phase-space means, not single orbits -- see ``_classical_lyapunov`` for why a
    # single orbit is not reproducible across platforms. The bounds below are the
    # measured spread over eight independent orbit sets, widened a little, and they
    # are ranges rather than ``approx`` values because the K = 5 mean is genuinely
    # slow to converge: its phase space is mixed, so a few orbits land in islands.
    lyapunov_five = _classical_lyapunov(5.0, orbits=32)
    lyapunov_ten = _classical_lyapunov(10.0, orbits=24)
    assert 0.92 <= lyapunov_five <= 1.00  # measured 0.9416-0.9807
    assert lyapunov_ten == pytest.approx(1.622, abs=0.02)  # measured 1.6182-1.6268

    ratio_five = rates[5.0] / (2.0 * lyapunov_five)
    ratio_ten = rates[10.0] / (2.0 * lyapunov_ten)
    # The point of the test: the prediction is right to within about 10% at K = 5
    # and wrong by nearly a factor of two at K = 10, so it is not an identity. The
    # ratio *crossing* one is the reproducible statement; neither endpoint is a
    # constant of the model. Measured over eight orbit sets: ratio_five
    # 1.0802-1.1250, ratio_ten 0.5469-0.5498, and their quotient at worst 1.970.
    assert 1.02 <= ratio_five <= 1.20
    assert 0.52 <= ratio_ten <= 0.58
    assert ratio_five > 1.0 > ratio_ten
    assert ratio_five / ratio_ten > 1.5


def test_otoc_is_invariant_under_a_global_unitary_change_of_basis() -> None:
    """``C`` depends on the algebra, not on the basis it is written in.

    Rotating ``A`` and ``B`` alone would *not* leave ``C`` invariant -- the
    evolution would still be the old one. The invariance is of the triple
    ``(U, A, B) -> (V U V.H, V A V.H, V B V.H)``, and that is what is tested.
    """
    generator = np.random.default_rng(11)
    dimension = 24
    rotation, _ = np.linalg.qr(
        generator.normal(size=(dimension, dimension))
        + 1j * generator.normal(size=(dimension, dimension))
    )
    model = QuantumBakerMap(dimension)
    clock, shift = weyl_translations(dimension)
    plain = otoc(model, steps=6, operator_a=clock, operator_b=shift)
    rotated = otoc(
        DenseUnitary(rotation @ model.to_dense() @ rotation.conj().T, name="rotated"),
        steps=6,
        operator_a=rotation @ clock @ rotation.conj().T,
        operator_b=rotation @ shift @ rotation.conj().T,
    )
    # Measured 1.42e-14; a rounding-level comparison rather than array equality,
    # because complex products are not bit-reproducible on the oldest supported
    # NumPy.
    assert np.max(np.abs(plain.values - rotated.values)) <= 1e-13


def test_otoc_validates_its_arguments_and_respects_the_dense_limit() -> None:
    model = KickedRotor(8, 3.0)
    with pytest.raises(ValidationError, match="steps must be non-negative"):
        otoc(model, steps=-1)
    with pytest.raises(ValidationError, match=r"operator_a must have shape \(8, 8\)"):
        otoc(model, steps=1, operator_a=np.eye(4))
    with pytest.raises(ValidationError, match=r"operator_b must have shape \(8, 8\)"):
        otoc(model, steps=1, operator_b=np.eye(4))
    with pytest.raises(ValidationError, match="not a batch"):
        otoc(model, steps=1, state=np.eye(8)[:2])
    with pytest.raises(ValidationError, match="exceeds dense_limit"):
        otoc(model, steps=1, dense_limit=4)
    with pytest.raises(ValidationError, match="model"):
        otoc(object(), steps=1)  # type: ignore[arg-type]


def test_loschmidt_echo_is_identically_one_when_both_models_agree() -> None:
    """The defining sanity check: no perturbation, no decay."""
    model = KickedRotor(128, 9.0)
    result = loschmidt_echo(
        model, model, basis_state(dimension=128, index=0), steps=50, method="fft"
    )
    assert result.name == "loschmidt_echo"
    assert result.values.shape == (51,)
    # M(0) is bit-exactly 1 for a basis state; measured worst drift 2.7e-14.
    assert result.values[0] == 1.0
    assert float(np.max(np.abs(result.values - 1.0))) <= 1e-13

    # A general normalized state only reaches 1 to the rounding of one
    # normalization: measured worst |M(0) - 1| 2.2e-15 and worst excess above
    # one 1.8e-15 over 200 random states. Values are deliberately not clipped.
    worst_initial = 0.0
    worst_excess = 0.0
    for seed in range(50):
        generator = np.random.default_rng(seed)
        state = normalize_state(generator.normal(size=128) + 1j * generator.normal(size=128))
        values = loschmidt_echo(model, model, state, steps=8, method="fft").values
        worst_initial = max(worst_initial, abs(float(values[0]) - 1.0))
        worst_excess = max(worst_excess, float(np.max(values)) - 1.0)
        assert np.all(values >= 0.0)
    assert worst_initial <= 1e-14
    assert 0.0 <= worst_excess <= 1e-14


def test_loschmidt_echo_decays_faster_for_a_stronger_perturbation() -> None:
    """Monotone in the perturbation at fixed time: the property that defines it."""
    dimension = 256
    state = coherent_state(dimension=dimension, position=0.5, momentum=0.0)
    deltas = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    measured = {
        1.0: (1.000000, 0.999998, 0.999983, 0.999843, 0.998252, 0.984771),
        5.0: (0.999589, 0.996300, 0.959453, 0.679165, 0.020998, 0.004536),
        10.0: (0.999746, 0.997712, 0.974829, 0.793635, 0.073779, 0.031860),
    }
    for kick, expected in measured.items():
        reference = KickedRotor(dimension, kick)
        row = [
            float(
                loschmidt_echo(
                    reference,
                    KickedRotor(dimension, kick + delta),
                    state,
                    steps=20,
                    method="fft",
                ).values[20]
            )
            for delta in deltas
        ]
        assert row == pytest.approx(expected, abs=1e-5)
        assert all(row[index] > row[index + 1] for index in range(len(row) - 1))


def test_loschmidt_echo_separates_regular_and_chaotic_kick_strengths() -> None:
    """Three orders of magnitude at forty steps, on one unchanged perturbation.

    The ordering is *not* monotone in ``K`` -- see the docstring -- so what is
    pinned here is the separation between the regular block (``K <= 2``) and the
    chaotic block (``K >= 5``), not a sort order.
    """
    dimension = 256
    state = coherent_state(dimension=dimension, position=0.5, momentum=0.0)
    echoes = {
        kick: float(
            loschmidt_echo(
                KickedRotor(dimension, kick),
                KickedRotor(dimension, kick + 1e-2),
                state,
                steps=40,
                method="fft",
            ).values[40]
        )
        for kick in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    }
    assert echoes[0.2] == pytest.approx(0.92309, abs=1e-4)
    assert echoes[0.5] == pytest.approx(0.98830, abs=1e-4)
    assert echoes[1.0] == pytest.approx(0.99170, abs=1e-4)
    assert echoes[2.0] == pytest.approx(0.97403, abs=1e-4)
    assert echoes[5.0] == pytest.approx(1.3137e-03, rel=2e-3)
    assert echoes[10.0] == pytest.approx(5.8574e-03, rel=2e-3)
    assert (
        min(echoes[kick] for kick in (0.2, 0.5, 1.0, 2.0)) / max(echoes[5.0], echoes[10.0]) > 100.0
    )


def test_loschmidt_echo_accepts_batched_initial_states() -> None:
    """A batch is evolved as one call and agrees with the per-state calls."""
    generator = np.random.default_rng(3)
    reference = KickedRotor(128, 5.0)
    perturbed = KickedRotor(128, 5.05)
    batch = normalize_state(
        generator.normal(size=(2, 3, 128)) + 1j * generator.normal(size=(2, 3, 128))
    )
    batched = loschmidt_echo(reference, perturbed, batch, steps=6, method="fft")
    assert batched.values.shape == (2, 3, 7)
    assert batched.metadata.parameters["batch_shape"] == (2, 3)
    single = loschmidt_echo(reference, perturbed, batch[1, 2], steps=6, method="fft")
    assert np.max(np.abs(batched.values[1, 2] - single.values)) <= 1e-15


def test_loschmidt_echo_rejects_a_mismatched_pair_and_invalid_models() -> None:
    with pytest.raises(ValidationError, match="share one Hilbert-space dimension"):
        loschmidt_echo(
            KickedRotor(16, 3.0),
            KickedRotor(32, 3.0),
            basis_state(dimension=16, index=0),
            steps=2,
        )
    with pytest.raises(ValidationError, match="perturbed"):
        loschmidt_echo(
            KickedRotor(16, 3.0),
            object(),  # type: ignore[arg-type]
            basis_state(dimension=16, index=0),
            steps=2,
        )
    # The two models need not be the same class, only the same dimension.
    result = loschmidt_echo(
        QuantumBakerMap(16),
        KickedRotor(16, 3.0),
        basis_state(dimension=16, index=0),
        steps=4,
    )
    assert result.values.shape == (5,)
    reference_parameters = result.metadata.parameters["reference"]
    perturbed_parameters = result.metadata.parameters["perturbed"]
    assert isinstance(reference_parameters, Mapping)
    assert isinstance(perturbed_parameters, Mapping)
    assert reference_parameters["model"] == "QuantumBakerMap"
    assert perturbed_parameters["model"] == "KickedRotor"

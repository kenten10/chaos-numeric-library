from __future__ import annotations

import numpy as np
import pytest

from chaos_numerics.core import QuantumMap, ValidationError
from chaos_numerics.quantum import (
    BoundaryPhases,
    QuantumBakerMap,
    QuantumCatMap,
    SymmetrySector,
    basis_state,
    desymmetrize,
    eigenstates,
    evolve,
    normalize_state,
)
from chaos_numerics.spectral import (
    adjacent_gap_ratios,
    mean_gap_ratio_reference,
    prepare_eigenphases,
)


@pytest.mark.parametrize("dimension", [2, 4, 8, 16])
@pytest.mark.parametrize("model_type", [QuantumCatMap, QuantumBakerMap])
def test_quantum_cat_and_baker_are_unitary_quantum_maps(
    dimension: int,
    model_type: type[QuantumCatMap] | type[QuantumBakerMap],
) -> None:
    model = model_type(dimension)
    rng = np.random.default_rng(123 + dimension)
    state = normalize_state(rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension))

    assert isinstance(model, QuantumMap)
    # ``unitarity_defect`` alone would be tautological here: DenseUnitary
    # already rejects a defect above 1e-12 at construction time. Preserving the
    # Gram matrix of an independent random batch exercises the ``apply`` path
    # with a different contraction than ``U.H @ U``.
    rng2 = np.random.default_rng(4321 + dimension)
    batch = normalize_state(
        rng2.standard_normal((3, dimension)) + 1j * rng2.standard_normal((3, dimension))
    )
    np.testing.assert_allclose(
        model.apply(batch) @ model.apply(batch).conj().T,
        batch @ batch.conj().T,
        rtol=1e-13,
        atol=1e-13,
    )
    assert abs(np.linalg.norm(model.apply(state)) - 1.0) <= 1e-14
    np.testing.assert_allclose(
        model.as_linear_operator().rmatvec(model.apply(state)),
        state,
        rtol=1e-14,
        atol=1e-14,
    )


def test_quantum_cat_matches_independent_chirp_dft_factorization() -> None:
    dimension = 6
    model = QuantumCatMap(dimension, matrix=((2, 1), (1, 1)))
    indices = np.arange(dimension, dtype=np.float64)
    fourier = np.fft.fft(np.eye(dimension), axis=0, norm="ortho")
    input_chirp = np.exp(1j * np.pi * 2.0 * indices**2 / dimension)
    output_chirp = np.exp(1j * np.pi * indices**2 / dimension)
    expected = np.exp(-0.25j * np.pi) * output_chirp[:, None] * fourier * input_chirp[None, :]

    np.testing.assert_allclose(model.to_dense(), expected, rtol=1e-14, atol=1e-14)


def test_quantum_baker_matches_independent_small_direct_dft() -> None:
    dimension = 4
    model = QuantumBakerMap(dimension)

    def direct_fourier(size: int) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        result = np.empty((size, size), dtype=np.complex128)
        for momentum in range(size):
            for position in range(size):
                result[momentum, position] = np.exp(
                    -2j * np.pi * (momentum + 0.5) * (position + 0.5) / size
                ) / np.sqrt(size)
        return result

    full = direct_fourier(dimension)
    half = direct_fourier(dimension // 2)
    blocks = np.zeros((dimension, dimension), dtype=np.complex128)
    blocks[:2, :2] = half
    blocks[2:, 2:] = half

    np.testing.assert_allclose(model.to_dense(), full.conj().T @ blocks, atol=1e-14)


def test_quantum_cat_map_of_dimension_eight_has_exact_period_six() -> None:
    """Fix the cat quantization against an exactly solvable finite orbit.

    For ``N = 8`` and ``M = ((2, 1), (1, 1))`` the metaplectic representation is
    six-periodic, so ``U**6 = I`` exactly and the eigenphases must be the
    sixth roots of unity restricted to the multiplicities this quantization
    produces. Both statements are independent of the implementation and pin the
    chirp signs, the ``exp(-i pi / 4)`` Maslov prefactor, and the ordering.
    """
    model = QuantumCatMap(8)
    dense = model.to_dense()

    identity_defect = float(np.max(np.abs(np.linalg.matrix_power(dense, 6) - np.eye(8))))
    assert identity_defect <= 1e-14

    result = eigenstates(model)
    expected = np.pi * np.asarray(
        [-2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
    )
    np.testing.assert_allclose(result.eigenphases, expected, atol=1e-14)

    # ``degenerate_pairs == 3`` is exactly the three repeated phases above
    # (-2pi/3, 0, +2pi/3), so the snapshot now has a theoretical justification.
    assert result.metadata.parameters["degenerate_pairs"] == 3
    unique, counts = np.unique(np.round(expected, 12), return_counts=True)
    assert int(np.sum(counts - 1)) == 3
    assert unique.size == 5


def test_maps_connect_to_evolution_eigenstates_and_symmetry_diagnostics() -> None:
    model = QuantumCatMap(8)
    initial = basis_state(dimension=model.dimension, index=1)

    evolved = evolve(model, initial, steps=20)
    result = eigenstates(model)

    assert abs(np.linalg.norm(evolved.final_state) - 1.0) <= 1e-13
    assert float(np.max(result.residuals)) <= 1e-12
    symmetry_defects = result.metadata.parameters["symmetry_defects"]
    assert symmetry_defects["parity"] <= 1e-10  # type: ignore[index]
    assert [item.code for item in result.metadata.warnings] == ["eigenphase-degeneracy"]


@pytest.mark.parametrize(
    ("dimension", "expected"),
    [
        # Generic even dimensions: both sectors land on the COE reference.
        (700, (0.5211, 0.5324)),
        (802, (0.5386, 0.5568)),
        (900, (0.5093, 0.5310)),
        # Powers of two: desymmetrizing does not repair the arithmetic anomaly.
        (256, (0.3806, 0.4038)),
        (512, (0.4078, 0.4309)),
        (1024, (0.4389, 0.4559)),
    ],
)
def test_baker_desymmetrization_reaches_coe_only_away_from_powers_of_two(
    dimension: int, expected: tuple[float, float]
) -> None:
    """Desymmetrizing is necessary but not sufficient at ``N = 2**k``.

    The library's own advice -- split the parity sectors before comparing with
    RMT -- is only actionable if the split actually gets you there. It does at a
    generic even ``N`` and it does not at a power of two, where the sectors stay
    6 to 8 standard errors below the COE reference 0.5307 and barely move off the
    raw two-sector value of 0.42.

    Nothing here is random: the eigenphases of a fixed dimension are a
    deterministic function of the map, so the only knob is the choice of ``N``
    and the quoted values are exact to the digits shown. The ``+- 0.0035`` band
    is four times the ``complex128`` reproducibility of the eigensolver across
    BLAS builds, not a sampling allowance.
    """
    model = QuantumBakerMap(dimension)
    system = eigenstates(model, dense_limit=dimension)
    parity = model.symmetry_operators["parity"]

    raw = adjacent_gap_ratios(prepare_eigenphases(system, symmetry_sector="mixed-parity"))
    assert float(np.mean(raw.values)) == pytest.approx(0.42, abs=0.03)

    coe = mean_gap_ratio_reference("coe")
    means = []
    sectors: tuple[SymmetrySector, ...] = ("even", "odd")
    for sector, reference in zip(sectors, expected, strict=True):
        block = desymmetrize(system, parity, sector=sector)
        assert block.count == dimension // 2
        # The split itself is exact: expectation values are +/-1 to 2.3e-15.
        deviation = block.metadata.parameters["sector_expectation_deviation"]
        assert isinstance(deviation, float)
        assert deviation <= 1e-13
        ratios = adjacent_gap_ratios(prepare_eigenphases(block, symmetry_sector=sector))
        mean = float(np.mean(ratios.values))
        assert mean == pytest.approx(reference, abs=0.0035)
        means.append(mean)

    # The standard error of a sector mean is 0.27 / sqrt(N / 2).
    standard_error = 0.27 / np.sqrt(dimension / 2.0)
    deviations = [(mean - coe) / standard_error for mean in means]
    if dimension & (dimension - 1):
        # Worst measured excursion over the generic dimensions is 1.94 (N = 802).
        assert max(abs(value) for value in deviations) <= 2.5
    else:
        # Best measured power-of-two sector is 6.27 standard errors low (N = 1024).
        assert max(deviations) <= -4.0


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: QuantumCatMap(3), "even positive integer"),
        (lambda: QuantumCatMap(4, matrix=((2, 2), (1, 1))), "determinant 1"),
        (lambda: QuantumCatMap(4, matrix=((1, 2), (0, 1))), r"\+1 or -1"),
        (lambda: QuantumCatMap(4, boundary_phases=0.5), "only periodic"),
        (lambda: QuantumBakerMap(5), "even positive integer"),
        (lambda: QuantumBakerMap(4, boundary_phases=0.0), "anti-periodic"),
        (
            lambda: QuantumBakerMap(4, boundary_phases=BoundaryPhases(0.5, 0.0)),
            "anti-periodic",
        ),
    ],
)
def test_invalid_quantization_conditions_fail_before_floquet_construction(
    constructor: object, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        constructor()  # type: ignore[operator]

from __future__ import annotations

import numpy as np
import pytest

from chaos_numerics.core import QuantumMap, ValidationError
from chaos_numerics.quantum import (
    BoundaryPhases,
    QuantumBakerMap,
    QuantumCatMap,
    basis_state,
    eigenstates,
    evolve,
    normalize_state,
    unitarity_defect,
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
    assert unitarity_defect(model) <= 1e-12
    assert abs(np.linalg.norm(model.apply(state)) - 1.0) <= 2e-13
    np.testing.assert_allclose(
        model.as_linear_operator().rmatvec(model.apply(state)),
        state,
        rtol=5e-13,
        atol=5e-13,
    )


def test_quantum_cat_matches_independent_chirp_dft_factorization() -> None:
    dimension = 6
    model = QuantumCatMap(dimension, matrix=((2, 1), (1, 1)))
    indices = np.arange(dimension, dtype=np.float64)
    fourier = np.fft.fft(np.eye(dimension), axis=0, norm="ortho")
    input_chirp = np.exp(1j * np.pi * 2.0 * indices**2 / dimension)
    output_chirp = np.exp(1j * np.pi * indices**2 / dimension)
    expected = np.exp(-0.25j * np.pi) * output_chirp[:, None] * fourier * input_chirp[None, :]

    np.testing.assert_allclose(model.to_dense(), expected, rtol=5e-13, atol=5e-13)


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

    np.testing.assert_allclose(model.to_dense(), full.conj().T @ blocks, atol=5e-13)


def test_maps_connect_to_evolution_eigenstates_and_symmetry_diagnostics() -> None:
    model = QuantumCatMap(8)
    initial = basis_state(dimension=model.dimension, index=1)

    evolved = evolve(model, initial, steps=20)
    result = eigenstates(model)

    assert abs(np.linalg.norm(evolved) - 1.0) <= 5e-11
    assert float(np.max(result.residuals)) <= 1e-12
    assert result.metadata.parameters["degenerate_pairs"] == 3
    symmetry_defects = result.metadata.parameters["symmetry_defects"]
    assert symmetry_defects["parity"] <= 1e-10  # type: ignore[index]
    assert [item.code for item in result.metadata.warnings] == ["eigenphase-degeneracy"]


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: QuantumCatMap(3), "even positive integer"),
        (lambda: QuantumCatMap(4, matrix=((2, 2), (1, 1))), "determinant 1"),
        (lambda: QuantumCatMap(4, matrix=((1, 2), (0, 1))), r"\+1 or -1"),
        (lambda: QuantumCatMap(4, boundary_phase=0.5), "only periodic"),
        (lambda: QuantumBakerMap(5), "even positive integer"),
        (lambda: QuantumBakerMap(4, boundary_phase=0.0), "anti-periodic"),
        (
            lambda: QuantumBakerMap(4, boundary_phase=BoundaryPhases(0.5, 0.0)),
            "anti-periodic",
        ),
    ],
)
def test_invalid_quantization_conditions_fail_before_floquet_construction(
    constructor: object, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        constructor()  # type: ignore[operator]

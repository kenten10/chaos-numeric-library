from __future__ import annotations

import numpy as np
import pytest

from chaos_numerics.core import QuantumMap, ValidationError
from chaos_numerics.quantum import (
    BoundaryPhases,
    DenseUnitary,
    KickedRotor,
    basis_state,
    evolve,
    normalize_state,
    unitarity_defect,
)


@pytest.mark.parametrize("dimension", [7, 8, 15, 16])
@pytest.mark.parametrize(
    "phases",
    [BoundaryPhases(), BoundaryPhases(0.25, 0.375)],
)
def test_dense_and_fft_actions_match_for_odd_even_dimensions(
    dimension: int, phases: BoundaryPhases
) -> None:
    model = KickedRotor(dimension, 7.25, boundary_phase=phases)
    rng = np.random.default_rng(1000 + dimension)
    states = normalize_state(
        rng.standard_normal((10, dimension)) + 1j * rng.standard_normal((10, dimension))
    )

    np.testing.assert_allclose(
        model.apply_fft(states),
        model.apply_dense(states),
        rtol=5e-13,
        atol=5e-13,
    )
    assert isinstance(model, QuantumMap)
    assert model.parameters["fft_normalization"] == "ortho"


def test_dense_reference_is_unitary_and_linear_operator_has_correct_adjoint() -> None:
    model = KickedRotor(12, -2.5, boundary_phase=BoundaryPhases(0.1, 0.3))
    state = normalize_state(np.arange(12) + 1j * np.arange(12)[::-1])
    linear = model.as_linear_operator()

    assert unitarity_defect(model) <= 1e-12
    np.testing.assert_allclose(linear.matvec(state), model.apply_fft(state), atol=1e-13)
    np.testing.assert_allclose(
        linear.rmatvec(linear.matvec(state)),
        state,
        rtol=5e-13,
        atol=5e-13,
    )


def test_fft_evolution_preserves_norm_without_dense_materialization() -> None:
    model = KickedRotor(257, 8.0, boundary_phase=0.125)
    initial = basis_state(dimension=model.dimension, index=17)

    history = evolve(model, initial, steps=1000, method="fft", return_history=True)

    assert history.shape == (1001, model.dimension)
    assert float(np.max(np.abs(np.linalg.norm(history, axis=-1) - 1.0))) <= 5e-11
    np.testing.assert_allclose(model.apply(initial), model.apply_fft(initial), atol=1e-14)


def test_dense_and_fft_evolve_use_the_common_api() -> None:
    model = KickedRotor(16, 4.0, boundary_phase=0.2)
    state = normalize_state(np.linspace(1.0, 2.0, model.dimension) * (1.0 + 0.5j))

    dense = evolve(model, state, steps=5, method="dense", dense_limit=16)
    fft = evolve(model, state, steps=5, method="fft")

    np.testing.assert_allclose(fft, dense, rtol=5e-12, atol=5e-13)


def test_kicked_rotor_scalar_boundary_phase_is_common_twist() -> None:
    model = KickedRotor(8, 1.0, boundary_phase=1.25)

    assert model.boundary_phases == BoundaryPhases(position=0.25, momentum=0.25)
    assert model.effective_hbar == pytest.approx(2.0 * np.pi / 8)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0, 1.0, 0.0), "dimension must be positive"),
        ((4.0, 1.0, 0.0), "positive integer"),
        ((4, float("nan"), 0.0), "kick_strength must be a finite real"),
        ((4, 1.0, float("inf")), "position phase must be a finite real"),
        ((4, 1.0, True), "position phase must be a finite real"),
    ],
)
def test_kicked_rotor_validates_quantization_before_dense_allocation(
    args: tuple[object, object, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        KickedRotor(*args)  # type: ignore[arg-type]


def test_fft_method_requires_an_fft_capable_model() -> None:
    with pytest.raises(ValidationError, match="does not provide apply_fft"):
        evolve(
            DenseUnitary(np.eye(2, dtype=np.complex128)),
            basis_state(dimension=2, index=0),
            steps=1,
            method="fft",
        )

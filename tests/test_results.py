from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from chaos_numerics.core import (
    AnalysisResult,
    ConvergenceInfo,
    Diagnostic,
    EigenstateResult,
    ExperimentMetadata,
    Spectrum,
    Trajectory,
    ValidationError,
)


def floats(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    return np.asarray(value, dtype=np.float64)


def complexes(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    return np.asarray(value, dtype=np.complex128)


def test_single_trajectory_owns_readonly_arrays_and_has_compact_repr() -> None:
    source = floats([[1.0, 0.0], [0.5, 0.5]])
    initial = floats([1.0, 0.0])
    result = Trajectory(source, initial)

    source[0, 0] = 99.0
    initial[0] = 99.0

    assert result.shape == (2, 2)
    assert result.time_count == 2
    assert result.state_dim == 2
    assert not result.is_batched
    assert result.states[0, 0] == 1.0
    assert result.initial_state[0] == 1.0
    assert not result.states.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.states[0, 0] = 2.0
    with pytest.raises(FrozenInstanceError):
        result.states = floats([[0.0]])  # type: ignore[misc]

    representation = repr(result)
    assert "shape=(2, 2)" in representation
    assert "[[" not in representation
    assert len(representation) < 160


def test_batched_trajectory_validates_initial_shape_and_equality() -> None:
    states = floats(np.arange(24).reshape(2, 3, 4))
    initial = floats(np.arange(8).reshape(2, 4))

    result = Trajectory(states, initial)

    assert result.is_batched
    assert result == Trajectory(states, initial)
    assert result != Trajectory(states + 1.0, initial)
    with pytest.raises(ValidationError, match=r"initial_state must have shape \(2, 4\)"):
        Trajectory(states, floats([0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValidationError, match="trajectory states must have shape"):
        Trajectory(floats([1.0, 2.0]), floats([1.0, 2.0]))


def test_spectrum_stores_column_eigenvectors_and_residuals() -> None:
    result = Spectrum(
        eigenvalues=complexes([1.0, -1.0j]),
        eigenvectors=complexes([[1.0, 0.0], [0.0, 1.0]]),
        residuals=floats([1e-12, 2e-12]),
    )

    assert result.count == 2
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (2, 2)
    assert result == Spectrum(
        complexes([1.0, -1.0j]),
        complexes([[1.0, 0.0], [0.0, 1.0]]),
        floats([1e-12, 2e-12]),
    )
    with pytest.raises(ValidationError, match="eigenvectors must store eigenvectors in columns"):
        Spectrum(complexes([1.0, 2.0]), complexes([[1.0], [0.0]]))
    with pytest.raises(ValidationError, match="residuals must be non-negative"):
        Spectrum(complexes([1.0]), residuals=floats([-1.0]))


def test_eigenstate_result_validates_shapes() -> None:
    result = EigenstateResult(
        eigenphases=floats([0.0, np.pi]),
        eigenstates=complexes([[1.0, 0.0], [0.0, 1.0]]),
        residuals=floats([0.0, 1e-14]),
    )

    assert result.count == 2
    assert not result.eigenstates.flags.writeable
    with pytest.raises(ValidationError, match="eigenstates must store states in columns"):
        EigenstateResult(floats([0.0, 1.0]), complexes([[1.0], [0.0]]), floats([0.0, 0.0]))


def test_analysis_result_validates_optional_arrays() -> None:
    values = floats([[1.0, 2.0], [3.0, 4.0]])
    result = AnalysisResult(
        "lyapunov",
        values,
        uncertainty=floats([[0.1, 0.1], [0.2, 0.2]]),
        residuals=floats([[0.0, 0.0], [1e-12, 1e-12]]),
    )

    assert result == AnalysisResult(
        "lyapunov",
        values,
        uncertainty=floats([[0.1, 0.1], [0.2, 0.2]]),
        residuals=floats([[0.0, 0.0], [1e-12, 1e-12]]),
    )
    with pytest.raises(ValidationError, match="uncertainty must have shape"):
        AnalysisResult("lyapunov", values, uncertainty=floats([0.1, 0.2]))
    with pytest.raises(ValidationError, match="residuals must be non-negative"):
        AnalysisResult("lyapunov", floats([1.0]), residuals=floats([-0.1]))


def test_metadata_is_deeply_immutable_and_json_compatible() -> None:
    parameters: dict[str, object] = {"steps": np.int64(8), "method": ["rk4", {"dt": 0.1}]}
    metadata = ExperimentMetadata(
        parameters=parameters,
        precision="mixed",
        seed=np.int64(42),
        environment={"numpy": np.__version__},
        git_commit="abc123",
        warnings=(Diagnostic("near-singular", "Matrix is near singular"),),
    )
    parameters["steps"] = 999

    assert metadata.parameters["steps"] == 8
    with pytest.raises(TypeError):
        metadata.parameters["steps"] = 7  # type: ignore[index]
    json.dumps(metadata.to_dict(), allow_nan=False)
    with pytest.raises(ValidationError, match="store numerical arrays in the array payload"):
        ExperimentMetadata(parameters={"array": floats([1.0])})
    with pytest.raises(ValidationError, match="finite for JSON serialization"):
        ExperimentMetadata(parameters={"value": float("nan")})


def test_convergence_history_is_owned_readonly_and_serialized_separately() -> None:
    history = floats([1.0, 0.1, 0.01])
    convergence = ConvergenceInfo(
        converged=True,
        iterations=3,
        residual=0.01,
        tolerance=0.02,
        history=history,
        reason="tolerance reached",
    )
    history[0] = 99.0
    metadata = ExperimentMetadata(convergence=convergence)
    result = Trajectory(floats([[1.0], [0.5]]), floats([1.0]), metadata)

    assert convergence.history is not None
    assert convergence.history[0] == 1.0
    assert not convergence.history.flags.writeable
    arrays = result.array_payload()
    arrays["states"][0, 0] = 50.0
    arrays["convergence_history"][0] = 50.0
    assert result.states[0, 0] == 1.0
    assert convergence.history[0] == 1.0

    payload = result.metadata_payload()
    json.dumps(payload, allow_nan=False)
    assert payload["arrays"] == {
        "states": {"shape": [2, 1], "dtype": "float64"},
        "initial_state": {"shape": [1], "dtype": "float64"},
        "convergence_history": {"shape": [3], "dtype": "float64"},
    }
    metadata_payload = payload["metadata"]
    assert isinstance(metadata_payload, dict)
    convergence_payload = metadata_payload["convergence"]
    assert isinstance(convergence_payload, dict)
    assert convergence_payload["history_array_key"] == "convergence_history"


def test_result_arrays_reject_nonfinite_values_and_payloads_are_writable() -> None:
    with pytest.raises(ValidationError, match="finite"):
        AnalysisResult("invalid", floats([np.inf]))

    spectrum = Spectrum(complexes([1.0 + 0.0j]))
    arrays = spectrum.array_payload()
    arrays["eigenvalues"][0] = 2.0 + 0.0j
    assert arrays["eigenvalues"].flags.writeable
    assert spectrum.eigenvalues[0] == 1.0 + 0.0j

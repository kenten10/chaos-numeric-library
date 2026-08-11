from __future__ import annotations

import copy
import json
import pickle
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import get_args

import numpy as np
import pytest

from chaos_numerics.core import (
    AnalysisResult,
    AnyArray,
    ArrayPayload,
    ConvergenceInfo,
    Diagnostic,
    EigenstateResult,
    ExperimentMetadata,
    SerializableResult,
    Spectrum,
    Trajectory,
    ValidationError,
)
from chaos_numerics.quantum import QuantumEvolution


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


def _rich_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        parameters={"kick": 1.5, "matrix": ((2, 1), (1, 1)), "nested": {"tolerance": 1e-9}},
        precision="mixed",
        seed=7,
        environment={"numpy": "2.0.0", "thread_limits": {"OMP_NUM_THREADS": "4"}},
        git_commit="deadbeef",
        warnings=(Diagnostic("code", "message"),),
        convergence=ConvergenceInfo(
            converged=True,
            iterations=3,
            residual=1e-13,
            tolerance=1e-12,
            history=floats([1.0, 0.5, 1e-13]),
            reason="converged",
        ),
    )


def _containers() -> list[object]:
    metadata = _rich_metadata()
    return [
        metadata,
        metadata.convergence,
        Trajectory(floats([[1.0, 0.0], [0.5, 0.5]]), floats([1.0, 0.0]), metadata),
        Spectrum(
            complexes([1.0, 1.0j]),
            complexes([[1.0, 0.0], [0.0, 1.0]]),
            floats([0.0, 0.0]),
            metadata,
        ),
        EigenstateResult(
            floats([-1.0, 1.0]),
            complexes([[1.0, 0.0], [0.0, 1.0]]),
            floats([0.0, 0.0]),
            metadata,
        ),
        AnalysisResult("analysis", floats([1.0, 2.0]), floats([0.1, 0.1]), None, metadata),
        QuantumEvolution(complexes([1.0, 0.0]), complexes([[1.0, 0.0], [0.0, 1.0]]), metadata),
    ]


@pytest.mark.parametrize("container", _containers(), ids=lambda item: type(item).__name__)
def test_containers_round_trip_through_pickle_and_deepcopy(container: object) -> None:
    """Parallel sweeps need these objects to cross a process boundary.

    ``parameters`` and ``environment`` are stored as ``MappingProxyType``, which
    has no ``__reduce__``, so without explicit support every result container
    raised ``TypeError: cannot pickle 'mappingproxy' object`` and even
    ``copy.deepcopy`` failed. That closed off ``multiprocessing`` and ``joblib``.
    """
    restored = pickle.loads(pickle.dumps(container))
    copied = copy.deepcopy(container)

    assert restored == container
    assert copied == container
    assert type(restored) is type(container)


def test_pickled_results_keep_read_only_arrays_and_frozen_mappings() -> None:
    original = EigenstateResult(
        floats([-1.0, 1.0]),
        complexes([[1.0, 0.0], [0.0, 1.0]]),
        floats([0.0, 0.0]),
        _rich_metadata(),
    )

    restored = pickle.loads(pickle.dumps(original))

    for array in (restored.eigenphases, restored.eigenstates, restored.residuals):
        assert not array.flags.writeable
    assert restored.metadata.convergence is not None
    assert restored.metadata.convergence.history is not None
    assert not restored.metadata.convergence.history.flags.writeable
    assert isinstance(restored.metadata.parameters, MappingProxyType)
    assert isinstance(restored.metadata.environment, MappingProxyType)
    with pytest.raises(TypeError):
        restored.metadata.parameters["kick"] = 0.0  # type: ignore[index]
    assert restored.metadata.to_dict() == original.metadata.to_dict()


def test_payload_type_aliases_are_reachable_from_the_public_namespace() -> None:
    """``array_payload() -> ArrayPayload`` is a public signature, so the alias is public.

    ``py.typed`` is shipped, so a caller who writes a function taking or returning
    one of these payloads has to be able to name its type. Before these entries
    existed, the only import path was the private ``core._payload`` module.
    """
    import chaos_numerics.core as core

    assert "ArrayPayload" in core.__all__
    assert "AnyArray" in core.__all__
    assert core.ArrayPayload is ArrayPayload
    assert core.AnyArray is AnyArray

    # The alias really does describe what the containers return.
    payload: ArrayPayload = Trajectory(np.zeros((3, 2)), np.zeros(2)).array_payload()
    array: AnyArray = payload["states"]
    assert array.shape == (3, 2)


def test_eigensolver_argument_aliases_are_reachable_from_the_public_namespace() -> None:
    """``Side`` and ``OperatorLike`` appear in :func:`leading_eigenpairs`'s signature."""
    import chaos_numerics.operators as operators
    from chaos_numerics.operators import OperatorLike, Side

    assert "Side" in operators.__all__
    assert "OperatorLike" in operators.__all__
    assert get_args(Side) == ("right", "left")

    operator: OperatorLike = np.asarray([[1.0]])
    side: Side = "left"
    assert operators.leading_eigenpairs(operator, count=1, side=side).count == 1


def _every_result_container() -> list[SerializableResult]:
    """Build one instance of every public result container in the library."""
    from chaos_numerics.classical import (
        CatMap,
        StandardMap,
        find_periodic_orbits,
        poincare_section,
    )
    from chaos_numerics.operators import UniformPartition, build_ulam
    from chaos_numerics.quantum import (
        BoundaryPhases,
        KickedRotor,
        basis_state,
        coherent_state,
        eigenstates,
        evolve,
        husimi_distribution,
        wigner_distribution,
    )
    from chaos_numerics.spectral import (
        number_variance,
        prepare_eigenphases,
        spacing_distribution,
        unfold,
    )

    eigensystem = eigenstates(KickedRotor(64, 10.0, BoundaryPhases(0.25, 0.13)))
    prepared = prepare_eigenphases(eigensystem, symmetry_sector="both symmetries broken")
    unfolded = unfold(prepared, method="mean")
    partition = UniformPartition([[0.0, 1.0], [0.0, 1.0]], (4, 4), periodic=(True, True))
    return [
        Trajectory(states=np.zeros((3, 2)), initial_state=np.zeros(2)),
        Spectrum(
            eigenvalues=np.array([1.0 + 0.0j]),
            eigenvectors=np.array([[1.0 + 0.0j]]),
            residuals=np.zeros(1),
        ),
        eigensystem,
        find_periodic_orbits(CatMap(), np.array([[0.01, 0.02]]), period=1),
        poincare_section(StandardMap(0.5), samples=4, steps=10, seed=1),
        husimi_distribution(
            coherent_state(dimension=16, position=0.25, momentum=0.75), grid_shape=(8, 8)
        ),
        wigner_distribution(coherent_state(dimension=16, position=0.25, momentum=0.75)),
        # With the history, because ``history`` is the one optional array key and
        # the descriptor check below only bites when the key is actually shipped.
        evolve(
            KickedRotor(16, 5.0),
            basis_state(dimension=16, index=0),
            steps=3,
            return_history=True,
        ),
        prepared,
        unfolded,
        spacing_distribution(unfolded, bins=10, value_range=(0.0, 4.0)),
        number_variance(unfolded, np.linspace(0.5, 4.0, 5), samples=256, seed=0),
        build_ulam(CatMap(), partition, samples_per_cell=32, seed=1),
    ]


def test_every_result_container_is_serializable() -> None:
    """Persistence is a whole-library contract, not a privilege of the core types.

    The storage layer splits arrays from JSON. A container that does not implement
    the split cannot be saved, which is why this asserts the protocol over every
    public result type rather than over a hand-picked list.
    """
    containers = _every_result_container()
    assert len(containers) >= 13
    seen_types: set[str] = set()
    for container in containers:
        name = type(container).__name__
        assert isinstance(container, SerializableResult), name

        arrays = container.array_payload()
        assert arrays, f"{name} reported no arrays"
        for key, array in arrays.items():
            assert array.flags.writeable, f"{name}.{key} must be a writable copy"

        descriptor = container.metadata_payload()
        # Must be JSON-compatible and must not smuggle array contents into the
        # descriptor; json.dumps raises on ndarray, so this checks both at once.
        encoded = json.dumps(descriptor)
        assert descriptor["schema_version"] == 1
        result_type = descriptor["result_type"]
        assert isinstance(result_type, str) and result_type
        assert result_type not in seen_types, f"duplicate result_type {result_type}"
        seen_types.add(result_type)
        described = descriptor["arrays"]
        assert isinstance(described, dict)
        assert set(described) <= set(arrays), name
        for entry in described.values():
            assert set(entry) == {"shape", "dtype"}
        assert len(encoded) < 200_000, f"{name} descriptor looks like it holds data"


def _containers_with_metadata(metadata: ExperimentMetadata) -> list[SerializableResult]:
    """Construct one of each container directly, so the metadata can be chosen.

    `_every_result_container` builds its instances by calling library functions, which
    supply their own metadata; this helper exists only for checks that need a specific
    `ExperimentMetadata`.
    """
    from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

    from chaos_numerics.classical import PeriodicOrbitResult, PoincareSection
    from chaos_numerics.operators import UlamMatrix
    from chaos_numerics.quantum import HusimiResult, WignerResult
    from chaos_numerics.spectral import (
        PreparedEigenphases,
        SpacingDistributionResult,
        SpectralCurve,
        UnfoldedSpectrum,
    )

    pair = floats([0.0, 1.0])
    return [
        Trajectory(states=np.zeros((3, 2)), initial_state=np.zeros(2), metadata=metadata),
        Spectrum(
            eigenvalues=np.array([1.0 + 0.0j]),
            eigenvectors=np.array([[1.0 + 0.0j]]),
            residuals=np.zeros(1),
            metadata=metadata,
        ),
        AnalysisResult(values=pair, name="values", metadata=metadata),
        PoincareSection(points=np.zeros((2, 2)), coordinates=(0, 1), metadata=metadata),
        PeriodicOrbitResult(
            period=1,
            points=np.zeros((1, 2)),
            residuals=np.zeros(1),
            monodromy_matrices=np.zeros((1, 2, 2)),
            stability_multipliers=np.zeros((1, 2), dtype=np.complex128),
            metadata=metadata,
        ),
        HusimiResult(
            positions=pair,
            momenta=pair,
            values=np.zeros((2, 2)),
            raw_integral=np.array(1.0),
            metadata=metadata,
        ),
        WignerResult(
            positions=pair,
            momenta=pair,
            values=np.zeros((2, 2)),
            negative_weight=np.array(0.0),
            metadata=metadata,
        ),
        PreparedEigenphases(
            raw_phases=pair, phases=pair, spacings=pair, symmetry_sector="s", metadata=metadata
        ),
        UnfoldedSpectrum(
            raw_phases=pair,
            phases=pair,
            values=pair,
            spacings=pair,
            symmetry_sector="s",
            method="mean",
            metadata=metadata,
        ),
        SpacingDistributionResult(
            values=floats([0.0, 0.0]),
            bin_edges=floats([0.0, 1.0, 2.0]),
            sample_count=2,
            metadata=metadata,
        ),
        SpectralCurve(pair, floats([0.0, 0.0]), metadata=metadata),
        QuantumEvolution(
            final_state=np.array([1.0 + 0.0j, 0.0 + 0.0j]),
            history=np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]),
            metadata=metadata,
        ),
        UlamMatrix(csr_matrix(np.eye(2)), np.zeros(2), metadata),
    ]


def test_metadata_payload_describes_every_array_it_ships() -> None:
    """A payload key missing from the descriptor is a `KeyError` during persistence.

    The storage layer matches `array_payload()` against `metadata_payload()["arrays"]`,
    so a container that ships an array it does not describe fails with an exception
    that names neither the container nor the array. `convergence_history` is the one
    conditional entry, and the containers built above carry no convergence history, so
    it takes a metadata object with one to reach the asymmetry at all.
    """
    history = ExperimentMetadata(
        convergence=ConvergenceInfo(
            converged=True,
            iterations=1,
            residual=0.0,
            tolerance=1e-12,
            history=floats([1.0, 2.0]),
            reason="synthetic history for the descriptor check",
        )
    )
    containers = _containers_with_metadata(history)
    for container in containers:
        described = container.metadata_payload()["arrays"]
        assert isinstance(described, dict)
        shipped = set(container.array_payload())
        assert shipped <= set(described), (
            f"{type(container).__name__} ships {sorted(shipped - set(described))} "
            "without describing it"
        )
    # The guard is only meaningful if the history actually reached the payloads.
    assert any("convergence_history" in c.array_payload() for c in containers)


def test_array_payload_copies_are_independent_of_the_container() -> None:
    for container in _every_result_container():
        arrays = container.array_payload()
        name, first = next(iter(arrays.items()))
        if first.size == 0:
            continue
        flat = first.reshape(-1)
        original = flat[0]
        flat[0] = original + 1
        again = container.array_payload()[name].reshape(-1)
        assert again[0] == original, f"{type(container).__name__}.{name} is not copied"

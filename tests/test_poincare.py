from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from chaos_numerics.classical import (
    LogisticMap,
    PoincareSection,
    StandardMap,
    iterate,
    poincare_section,
)
from chaos_numerics.core import ExperimentMetadata, ValidationError


def floats(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    return np.asarray(value, dtype=np.float64)


def test_sampled_section_shape_bounds_and_recorded_metadata() -> None:
    section = poincare_section(StandardMap(0.3), samples=8, steps=100, seed=7)

    assert section.points.shape == (808, 2)
    assert section.count == 808
    assert section.coordinates == (0, 1)
    assert bool(np.all((section.points >= 0.0) & (section.points < 1.0)))
    parameters = section.metadata.parameters
    assert parameters["model"] == "StandardMap"
    assert parameters["model_parameters"]["kick_strength"] == 0.3  # type: ignore[index]
    assert parameters["steps"] == 100
    assert parameters["transient"] == 0
    assert parameters["trajectory_count"] == 8
    assert parameters["coordinates"] == (0, 1)
    assert section.metadata.seed == 7
    assert "count=808" in repr(section)


def test_unseeded_section_records_the_seed_it_actually_used() -> None:
    section = poincare_section(StandardMap(0.3), samples=4, steps=10)
    recorded = section.metadata.seed

    assert isinstance(recorded, int)
    assert section.metadata.parameters["seed"] == recorded
    replay = poincare_section(StandardMap(0.3), samples=4, steps=10, seed=recorded)
    np.testing.assert_array_equal(replay.points, section.points)


def test_same_seed_is_bit_identical_and_different_seeds_differ() -> None:
    first = poincare_section(StandardMap(1.1), samples=6, steps=20, seed=1234)
    same = poincare_section(StandardMap(1.1), samples=6, steps=20, seed=1234)
    other = poincare_section(StandardMap(1.1), samples=6, steps=20, seed=1235)

    np.testing.assert_array_equal(first.points, same.points)
    assert first == same
    assert not np.array_equal(first.points, other.points)


def test_transient_is_discarded_and_matches_a_direct_iterate_slice() -> None:
    model = StandardMap(0.7)
    starts = floats([[0.11, 0.22], [0.33, 0.44]])
    section = poincare_section(model, starts, steps=5, transient=3)
    reference = iterate(model, starts, steps=8).states[:, 3:, :].reshape(-1, 2)

    assert section.points.shape == (12, 2)
    np.testing.assert_array_equal(section.points, reference)


def test_initial_states_and_samples_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="exactly one of initial_states and samples"):
        poincare_section(StandardMap(0.3), steps=5)
    with pytest.raises(ValidationError, match="exactly one of initial_states and samples"):
        poincare_section(StandardMap(0.3), floats([0.1, 0.2]), steps=5, samples=3)
    with pytest.raises(ValidationError, match="seed applies only to the samples path"):
        poincare_section(StandardMap(0.3), floats([0.1, 0.2]), steps=5, seed=3)


def test_coordinate_selection_and_out_of_range_rejection() -> None:
    model = StandardMap(2.0)
    both = poincare_section(model, samples=3, steps=4, seed=11, coordinates=(0, 1))
    momentum = poincare_section(model, samples=3, steps=4, seed=11, coordinates=(1,))

    assert momentum.points.shape == (15, 1)
    assert momentum.coordinates == (1,)
    np.testing.assert_array_equal(momentum.points[:, 0], both.points[:, 1])
    with pytest.raises(ValidationError, match=r"coordinates must lie in \[0, 2\)"):
        poincare_section(model, samples=3, steps=4, seed=11, coordinates=(2,))
    with pytest.raises(ValidationError, match="at least one coordinate"):
        poincare_section(model, samples=3, steps=4, seed=11, coordinates=())


def test_one_dimensional_model_defaults_to_its_single_coordinate() -> None:
    section = poincare_section(LogisticMap(3.9), samples=5, steps=50, transient=100, seed=99)

    assert section.coordinates == (0,)
    assert section.points.shape == (255, 1)
    assert bool(np.all((section.points >= 0.0) & (section.points < 1.0)))


def test_points_are_read_only_and_array_payload_is_a_writable_copy() -> None:
    section = poincare_section(StandardMap(0.3), samples=2, steps=3, seed=5)

    assert not section.points.flags.writeable
    payload = section.array_payload()
    assert payload["points"].flags.writeable
    payload["points"][0, 0] = -12345.0
    assert float(section.points[0, 0]) != -12345.0
    descriptors = section.metadata_payload()
    assert descriptors["result_type"] == "poincare_section"
    assert descriptors["coordinates"] == [0, 1]
    assert descriptors["arrays"] == {"points": {"shape": [8, 2], "dtype": "float64"}}


@pytest.mark.parametrize(
    "roundtrip", [lambda item: pickle.loads(pickle.dumps(item)), copy.deepcopy]
)
def test_section_survives_pickle_and_deepcopy(roundtrip: object) -> None:
    section = poincare_section(StandardMap(1.3), samples=3, steps=7, seed=2024)
    assert callable(roundtrip)
    restored = roundtrip(section)

    assert isinstance(restored, PoincareSection)
    assert restored == section
    assert not restored.points.flags.writeable
    assert restored.metadata.seed == 2024


def test_section_container_validates_its_own_inputs() -> None:
    with pytest.raises(ValidationError, match="one column per coordinate"):
        PoincareSection(floats([[0.1, 0.2]]), (0,), ExperimentMetadata())
    with pytest.raises(ValidationError, match="ExperimentMetadata"):
        PoincareSection(floats([[0.1, 0.2]]), (0, 1), object())  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="steps must be positive"):
        poincare_section(StandardMap(0.3), samples=2, steps=0)
    with pytest.raises(ValidationError, match="samples must be positive"):
        poincare_section(StandardMap(0.3), samples=0, steps=2)
    with pytest.raises(ValidationError, match="ClassicalMap protocol"):
        poincare_section(object(), samples=2, steps=2)  # type: ignore[arg-type]

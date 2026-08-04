from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

from chaos_numerics.classical import CatMap
from chaos_numerics.core import ValidationError
from chaos_numerics.operators import UlamMatrix, UniformPartition, build_ulam
from chaos_numerics.operators.ulam import _build_ulam_dense_reference


class IdentityMap:
    @property
    def state_dim(self) -> int:
        return 1

    @property
    def is_periodic(self) -> tuple[bool]:
        return (False,)

    @property
    def bounds(self) -> tuple[tuple[float, float]]:
        return ((0.0, 1.0),)

    def step(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.asarray(state, dtype=np.float64)

    def jacobian(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        del state
        return np.ones((1, 1), dtype=np.float64)


class EscapingMap(IdentityMap):
    def step(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.asarray(state, dtype=np.float64) + 1.0


def test_identity_map_ulam_matrix_is_exact_identity() -> None:
    partition = UniformPartition(bounds=((0.0, 1.0),), shape=(5,))
    matrix = build_ulam(IdentityMap(), partition, samples_per_cell=16, seed=7)

    assert isinstance(matrix, UlamMatrix)
    assert matrix.format == "csr"
    np.testing.assert_array_equal(matrix.toarray(), np.eye(5))
    np.testing.assert_array_equal(matrix.escape_probabilities, np.zeros(5))
    assert not matrix.escape_probabilities.flags.writeable
    assert matrix.metadata.parameters["orientation"] == "P[target, source]"
    assert matrix.metadata.seed == 7


def test_cat_map_is_column_stochastic_and_nonnegative() -> None:
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(4, 4),
        periodic=(True, True),
    )
    matrix = build_ulam(CatMap(), partition, samples_per_cell=64, seed=11, batch_size=13)
    column_sums = np.asarray(matrix.sum(axis=0)).reshape(-1)

    np.testing.assert_allclose(column_sums, np.ones(partition.size), rtol=0.0, atol=1e-12)
    assert matrix.data.size > 0
    assert float(np.min(matrix.data)) >= 0.0
    assert float(np.max(matrix.data)) <= 1.0 + 1e-15


def test_seed_reproducibility_and_batch_size_do_not_change_counts() -> None:
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(3, 3),
        periodic=(True, True),
    )
    first = build_ulam(CatMap(), partition, samples_per_cell=31, seed=123, batch_size=31)
    second = build_ulam(CatMap(), partition, samples_per_cell=31, seed=123, batch_size=4)

    np.testing.assert_array_equal(first.indptr, second.indptr)
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.data, second.data)
    np.testing.assert_array_equal(first.escape_probabilities, second.escape_probabilities)


def test_csr_matches_independent_dense_reference() -> None:
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(2, 3),
        periodic=(True, True),
    )
    sparse = build_ulam(CatMap(), partition, samples_per_cell=23, seed=19, batch_size=5)
    dense, escape = _build_ulam_dense_reference(
        CatMap(),
        partition,
        samples_per_cell=23,
        seed=19,
    )

    np.testing.assert_allclose(sparse.toarray(), dense, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(sparse.escape_probabilities, escape)


def test_open_system_reports_column_mass_deficits() -> None:
    partition = UniformPartition(bounds=((0.0, 1.0),), shape=(4,))
    matrix = build_ulam(
        EscapingMap(),
        partition,
        samples_per_cell=8,
        seed=2,
        open_system=True,
    )
    column_sums = np.asarray(matrix.sum(axis=0)).reshape(-1)

    np.testing.assert_array_equal(matrix.toarray(), np.zeros((4, 4)))
    np.testing.assert_array_equal(matrix.escape_probabilities, np.ones(4))
    np.testing.assert_allclose(column_sums + matrix.escape_probabilities, np.ones(4))

    with pytest.raises(ValidationError, match="closed-system image escaped"):
        build_ulam(EscapingMap(), partition, samples_per_cell=8, seed=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples_per_cell": 0}, "samples_per_cell must be positive"),
        ({"samples_per_cell": 4, "batch_size": 0}, "batch_size must be positive"),
        ({"samples_per_cell": 4, "seed": True}, "seed must be a non-negative integer"),
        ({"samples_per_cell": 4, "open_system": 1}, "open_system must be a bool"),
    ],
)
def test_ulam_parameters_are_validated(kwargs: dict[str, object], message: str) -> None:
    partition = UniformPartition(bounds=((0.0, 1.0),), shape=(2,))
    with pytest.raises(ValidationError, match=message):
        build_ulam(IdentityMap(), partition, **kwargs)  # type: ignore[arg-type]


def test_model_partition_dimension_must_match() -> None:
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(2, 2),
        periodic=(True, True),
    )
    with pytest.raises(ValidationError, match="state_dim 1 must equal partition ndim 2"):
        build_ulam(IdentityMap(), partition, samples_per_cell=4, seed=1)

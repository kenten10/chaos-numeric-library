from __future__ import annotations

import copy
import json
import pickle
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore[import-untyped]

from chaos_numerics.classical import CatMap
from chaos_numerics.core import ExperimentMetadata, LinearOperatorLike, ValidationError
from chaos_numerics.core.results import SCHEMA_VERSION
from chaos_numerics.operators import UlamMatrix, UniformPartition, build_ulam
from chaos_numerics.operators.ulam import _build_ulam_dense_reference

# Every SciPy name a caller used to reach through inheritance. They are now
# reached through ``.matrix`` instead, and this list is the contract test for
# that: the frozen v0.1 surface of ``UlamMatrix`` is small and read-only.
FORWARDED_SCIPY_NAMES = (
    "astype",
    "conj",
    "copy",
    "data",
    "eliminate_zeros",
    "format",
    "indices",
    "indptr",
    "multiply",
    "power",
    "setdiag",
    "sort_indices",
    "sqrt",
    "sum",
    "sum_duplicates",
    "T",
    "tocsc",
    "transpose",
)


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


class LeakyDoublingMap(IdentityMap):
    """``x -> 2x mod 1`` observed on the sub-domain ``[0, 0.75)``.

    Mass that lands in the hole ``[0.75, 1)`` leaves the partition, so some
    columns are strictly substochastic and the escape vector is partly nonzero
    with fractional entries rather than the all-or-nothing pattern of
    :class:`EscapingMap`.
    """

    def step(self, state: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.mod(2.0 * np.asarray(state, dtype=np.float64), 1.0)


class HalfOpenIntervals:
    """A ``Partition`` that is not a :class:`RectangularPartition`.

    ``build_ulam`` reaches its vectorized escape mask through the concrete
    rectangular partitions; a user-defined partition only promises the
    :class:`~chaos_numerics.core.Partition` protocol, whose ``locate`` signals an
    out-of-domain point by raising. This class is that second contract: equal
    cells over ``[0, 1)`` with no batch admission test of its own.
    """

    def __init__(self, cells: int) -> None:
        self._cells = cells

    @property
    def ndim(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._cells,)

    @property
    def size(self) -> int:
        return self._cells

    def locate(self, points: ArrayLike, /) -> np.ndarray[tuple[int, ...], np.dtype[np.intp]]:
        values = np.asarray(points, dtype=np.float64)[..., 0]
        if bool(np.any((values < 0.0) | (values >= 1.0))):
            raise ValidationError("points coordinate 0 must lie in [0.0, 1.0)")
        return np.floor(values * self._cells).astype(np.intp)

    def ravel_index(
        self, multi_index: ArrayLike, /
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.intp]]:
        return np.asarray(multi_index, dtype=np.intp)[..., 0]

    def unravel_index(
        self, flat_index: ArrayLike, /
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.intp]]:
        return np.asarray(flat_index, dtype=np.intp)[..., None]

    def cell_bounds(self, index: int, /) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        return np.asarray([[index / self._cells, (index + 1) / self._cells]], dtype=np.float64)

    def sample(
        self, index: int, count: int, *, seed: int | None = None
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        lower, upper = (float(bound) for bound in self.cell_bounds(index)[0])
        unit = np.random.default_rng(seed).random((count, 1), dtype=np.float64)
        return np.asarray(lower + unit * (upper - lower), dtype=np.float64)


def leaky_matrix() -> UlamMatrix:
    """An open-system matrix with a partly nonzero escape vector and real structure."""
    partition = UniformPartition(bounds=((0.0, 0.75),), shape=(7,))
    return build_ulam(
        LeakyDoublingMap(),
        partition,
        samples_per_cell=64,
        seed=3,
        open_system=True,
    )


def test_identity_map_ulam_matrix_is_exact_identity() -> None:
    partition = UniformPartition(bounds=((0.0, 1.0),), shape=(5,))
    matrix = build_ulam(IdentityMap(), partition, samples_per_cell=16, seed=7)

    assert isinstance(matrix, UlamMatrix)
    assert matrix.matrix.format == "csr"
    assert matrix.shape == (5, 5)
    assert matrix.dtype == np.dtype(np.float64)
    assert matrix.nnz == 5
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
    column_sums = np.asarray(matrix.matrix.sum(axis=0)).reshape(-1)

    np.testing.assert_allclose(column_sums, np.ones(partition.size), rtol=0.0, atol=1e-12)
    assert matrix.nnz > 0
    assert float(np.min(matrix.matrix.data)) >= 0.0
    assert float(np.max(matrix.matrix.data)) <= 1.0 + 1e-15


def test_seed_reproducibility_and_batch_size_do_not_change_counts() -> None:
    """The three CSR arrays and the escape vector replay bit for bit from a seed."""
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(3, 3),
        periodic=(True, True),
    )
    first = build_ulam(CatMap(), partition, samples_per_cell=31, seed=123, batch_size=31)
    second = build_ulam(CatMap(), partition, samples_per_cell=31, seed=123, batch_size=4)

    np.testing.assert_array_equal(first.matrix.indptr, second.matrix.indptr)
    np.testing.assert_array_equal(first.matrix.indices, second.matrix.indices)
    np.testing.assert_array_equal(first.matrix.data, second.matrix.data)
    np.testing.assert_array_equal(first.escape_probabilities, second.escape_probabilities)

    # The same guarantee read through the persistence contract rather than the
    # stored matrix, because that is the form a saved experiment replays from.
    for key, value in first.array_payload().items():
        np.testing.assert_array_equal(value, second.array_payload()[key], err_msg=key)


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
    column_sums = np.asarray(matrix.matrix.sum(axis=0)).reshape(-1)

    np.testing.assert_array_equal(matrix.toarray(), np.zeros((4, 4)))
    np.testing.assert_array_equal(matrix.escape_probabilities, np.ones(4))
    np.testing.assert_allclose(column_sums + matrix.escape_probabilities, np.ones(4))
    assert matrix.nnz == 0

    with pytest.raises(ValidationError, match="closed-system image escaped"):
        build_ulam(EscapingMap(), partition, samples_per_cell=8, seed=2)


def test_leaky_partition_records_fractional_escape() -> None:
    matrix = leaky_matrix()
    column_sums = np.asarray(matrix.matrix.sum(axis=0)).reshape(-1)
    escape = matrix.escape_probabilities

    assert float(np.sum(escape)) > 0.0
    assert bool(np.any(escape == 0.0))
    assert bool(np.any((escape > 0.0) & (escape < 1.0)))
    np.testing.assert_allclose(column_sums + escape, np.ones(matrix.shape[1]), atol=1e-12)
    assert matrix.metadata.parameters["open_system"] is True


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


def test_stored_arrays_are_read_only_and_the_operator_is_frozen() -> None:
    """The mutation holes that inheriting ``csr_matrix`` used to leave open."""
    matrix = leaky_matrix()

    with pytest.raises(ValueError, match="read-only"):
        matrix.matrix.data[0] = 12345.0
    with pytest.raises(ValueError, match="read-only"):
        matrix.matrix.indices[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        matrix.matrix.indptr[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        matrix.escape_probabilities[0] = 0.5
    with pytest.raises(FrozenInstanceError):
        matrix.escape_probabilities = np.zeros(matrix.shape[1])  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        matrix.matrix = csr_matrix(np.eye(matrix.shape[0]))  # type: ignore[misc]

    # Item assignment into a stored entry was the other way to make the
    # diagnostics describe a matrix that no longer existed.
    with pytest.raises(ValueError, match="read-only"):
        matrix.matrix[0, 0] = 0.5


def test_scipy_operations_are_reached_through_the_matrix_attribute() -> None:
    """The forwarded surface is small; SciPy arithmetic lives on ``.matrix``."""
    matrix = leaky_matrix()

    for name in FORWARDED_SCIPY_NAMES:
        assert not hasattr(matrix, name), name
    assert not isinstance(matrix, csr_matrix)

    # ``Any`` because SciPy ships no stubs, so operator results are untyped.
    product: Any = matrix.matrix @ matrix.matrix
    total: Any = matrix.matrix + matrix.matrix
    assert not isinstance(product, UlamMatrix)
    assert not isinstance(total, UlamMatrix)
    np.testing.assert_allclose(product.toarray(), matrix.toarray() @ matrix.toarray())

    # Diagnostics do not follow the product, and nothing pretends otherwise: the
    # caller keeps the escape vector it needs and re-wraps deliberately.
    escape = matrix.escape_probabilities
    rewrapped = UlamMatrix(product, escape, matrix.metadata)
    np.testing.assert_array_equal(rewrapped.escape_probabilities, escape)
    assert rewrapped.metadata.seed == matrix.metadata.seed


def test_matvec_and_matmul_agree_with_the_stored_matrix() -> None:
    matrix = leaky_matrix()
    vector = np.linspace(0.1, 1.3, matrix.shape[1])

    expected = np.asarray(matrix.matrix @ vector)
    np.testing.assert_allclose(matrix.matvec(vector), expected, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(matrix @ vector, expected, rtol=0.0, atol=0.0)
    # ``LinearOperatorLike`` fixes the matvec dtype; ``@`` keeps SciPy's own.
    assert matrix.matvec(vector).dtype == np.dtype(np.complex128)
    assert np.asarray(matrix @ vector).dtype == np.dtype(np.float64)

    block = np.eye(matrix.shape[1], 3)
    np.testing.assert_allclose(matrix @ block, np.asarray(matrix.matrix @ block))

    with pytest.raises(ValidationError, match="vector must have shape"):
        matrix.matvec(np.ones(matrix.shape[1] + 1))


def test_ulam_matrix_satisfies_the_linear_operator_protocol() -> None:
    """``isinstance`` used to be ``False`` because ``csr_matrix`` has no ``matvec``."""
    matrix = leaky_matrix()
    operator: LinearOperatorLike = matrix

    assert isinstance(matrix, LinearOperatorLike)
    assert operator.shape == (7, 7)
    assert operator.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        operator.matvec(np.ones(7)),
        np.asarray(matrix.matrix @ np.ones(7)),
    )


def test_array_payload_is_writable_and_independent() -> None:
    matrix = leaky_matrix()
    payload = matrix.array_payload()

    assert set(payload) == {"data", "indices", "indptr", "shape", "escape_probabilities"}
    for name, array in payload.items():
        assert array.flags.writeable, name

    original_data = np.array(matrix.matrix.data, copy=True)
    original_escape = np.array(matrix.escape_probabilities, copy=True)
    payload["data"][0] = 12345.0
    payload["escape_probabilities"][0] = 0.5
    np.testing.assert_array_equal(matrix.matrix.data, original_data)
    np.testing.assert_array_equal(matrix.escape_probabilities, original_escape)


def test_array_payload_reconstructs_the_stored_matrix() -> None:
    matrix = leaky_matrix()
    payload = matrix.array_payload()
    rebuilt = csr_matrix(
        (payload["data"], payload["indices"], payload["indptr"]),
        shape=tuple(int(extent) for extent in payload["shape"]),
    )

    np.testing.assert_array_equal(rebuilt.toarray(), matrix.toarray())
    np.testing.assert_array_equal(
        UlamMatrix(
            rebuilt,
            np.asarray(payload["escape_probabilities"], dtype=np.float64),
            matrix.metadata,
        ).toarray(),
        matrix.toarray(),
    )


def test_metadata_payload_describes_arrays_without_their_contents() -> None:
    matrix = leaky_matrix()
    payload = matrix.metadata_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["result_type"] == "ulam_matrix"
    assert payload["shape"] == [7, 7]
    arrays = payload["arrays"]
    assert isinstance(arrays, dict)
    assert set(arrays) == {"data", "indices", "indptr", "shape", "escape_probabilities"}
    assert arrays["escape_probabilities"] == {"shape": [7], "dtype": "float64"}
    assert arrays["data"] == {"shape": [matrix.nnz], "dtype": "float64"}
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["seed"] == 3

    # A metadata payload that cannot be serialized as JSON is carrying arrays.
    json.dumps(payload)


def test_pickle_and_deepcopy_round_trip_preserves_diagnostics_and_immutability() -> None:
    original = leaky_matrix()
    clones = (pickle.loads(pickle.dumps(original)), copy.deepcopy(original))
    for clone in clones:
        assert isinstance(clone, UlamMatrix)
        assert clone.matrix.format == "csr"
        np.testing.assert_array_equal(clone.toarray(), original.toarray())
        np.testing.assert_array_equal(clone.escape_probabilities, original.escape_probabilities)
        assert not clone.escape_probabilities.flags.writeable
        assert not clone.matrix.data.flags.writeable
        assert clone.metadata.parameters == original.metadata.parameters
        assert clone.metadata.seed == original.metadata.seed
        assert clone == original


def test_construction_normalizes_any_sparse_input_to_canonical_csr() -> None:
    duplicated = coo_matrix(
        (
            np.asarray([0.25, 0.25, 1.0], dtype=np.float64),
            (np.asarray([0, 0, 1]), np.asarray([0, 0, 1])),
        ),
        shape=(2, 2),
    )
    matrix = UlamMatrix(duplicated, np.asarray([0.5, 0.0]))

    assert matrix.matrix.format == "csr"
    assert matrix.nnz == 2
    np.testing.assert_array_equal(matrix.toarray(), np.asarray([[0.5, 0.0], [0.0, 1.0]]))
    assert not matrix.matrix.data.flags.writeable
    assert matrix.metadata == ExperimentMetadata(precision="float64")


@pytest.mark.parametrize(
    ("matrix", "escape", "message"),
    [
        (np.eye(2), np.zeros(2), "must be a SciPy sparse matrix"),
        (csr_matrix((2, 3), dtype=np.float64), np.zeros(3), "must be square"),
        (csr_matrix(np.eye(2)), np.zeros(3), r"must have shape \(2,\)"),
        (csr_matrix(np.eye(2)), np.asarray([0.0, 1.5]), r"finite values in \[0, 1\]"),
        (csr_matrix(np.asarray([[-1.0, 0.0], [0.0, 1.0]])), np.zeros(2), "non-negative"),
        (csr_matrix(np.asarray([[0.7, 0.0], [0.7, 1.0]])), np.zeros(2), "column-substochastic"),
        (csr_matrix(np.eye(2, dtype=np.complex128)), np.zeros(2), "real numeric values"),
    ],
)
def test_ulam_matrix_construction_is_validated(
    matrix: Any,
    escape: Any,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        UlamMatrix(matrix, escape)


def test_metadata_must_be_experiment_metadata() -> None:
    with pytest.raises(ValidationError, match="must be an ExperimentMetadata instance"):
        UlamMatrix(csr_matrix(np.eye(2)), np.zeros(2), {"seed": 1})  # type: ignore[arg-type]


def test_repr_stays_compact_for_large_operators() -> None:
    matrix = leaky_matrix()
    text = repr(matrix)

    assert text.startswith("UlamMatrix(shape=(7, 7), dtype='float64', nnz=")
    assert "escape_max=" in text
    assert len(text) < 200


def test_equality_compares_structure_escape_and_metadata() -> None:
    matrix = leaky_matrix()
    same = leaky_matrix()
    other_escape = UlamMatrix(matrix.matrix, np.zeros(matrix.shape[1]), matrix.metadata)

    assert matrix == same
    assert matrix != other_escape
    assert matrix != matrix.matrix


def test_open_system_matches_the_point_at_a_time_reference_exactly() -> None:
    """The vectorized escape mask must not change a single sampled count.

    ``build_ulam(open_system=True)`` used to call ``locate`` once per sampled
    point inside a ``try``/``except``; it now computes the escape mask for a whole
    batch and calls ``locate`` once on the surviving rows.
    ``_build_ulam_dense_reference`` is still the point-at-a-time loop, so it is the
    regression guard for that rewrite. ``samples_per_cell`` is a power of two, so
    the reference's repeated ``+= 1.0 / samples`` is exact and the comparison can
    be an equality rather than a tolerance.
    """
    partition = UniformPartition(bounds=((0.0, 0.75),), shape=(7,))
    sparse = build_ulam(
        LeakyDoublingMap(),
        partition,
        samples_per_cell=64,
        seed=3,
        batch_size=9,
        open_system=True,
    )
    dense, escape = _build_ulam_dense_reference(
        LeakyDoublingMap(),
        partition,
        samples_per_cell=64,
        seed=3,
        open_system=True,
    )

    assert float(np.sum(escape)) > 0.0
    np.testing.assert_array_equal(sparse.toarray(), dense)
    np.testing.assert_array_equal(sparse.escape_probabilities, escape)


def test_open_system_on_a_fully_periodic_partition_loses_no_mass() -> None:
    """A periodic coordinate has no outside, so ``open_system=True`` changes nothing.

    This is the branch of the escape mask that skips a periodic coordinate: every
    finite image wraps back into the domain, so the matrix has to stay
    column-stochastic and identical to the closed build.
    """
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(4, 4),
        periodic=(True, True),
    )
    opened = build_ulam(
        CatMap(), partition, samples_per_cell=64, seed=11, batch_size=13, open_system=True
    )
    closed = build_ulam(CatMap(), partition, samples_per_cell=64, seed=11, batch_size=13)

    np.testing.assert_array_equal(opened.escape_probabilities, np.zeros(partition.size))
    np.testing.assert_array_equal(opened.matrix.data, closed.matrix.data)
    np.testing.assert_array_equal(opened.matrix.indices, closed.matrix.indices)
    np.testing.assert_array_equal(opened.matrix.indptr, closed.matrix.indptr)


def test_open_system_supports_a_user_defined_partition() -> None:
    """A partition that only implements the protocol still counts escapes.

    The batch escape mask is an internal capability of the rectangular partitions.
    Everything else falls back to asking ``locate`` one point at a time, which is
    all the public :class:`~chaos_numerics.core.Partition` protocol offers, and
    this pins that both routes agree on the same map.
    """
    foreign = HalfOpenIntervals(4)
    rectangular = UniformPartition(bounds=((0.0, 1.0),), shape=(4,))
    kwargs = {"samples_per_cell": 32, "seed": 21, "open_system": True}

    escaping = build_ulam(EscapingMap(), foreign, **kwargs)  # type: ignore[arg-type]
    np.testing.assert_array_equal(escaping.escape_probabilities, np.ones(4))
    assert escaping.nnz == 0

    identity = build_ulam(IdentityMap(), foreign, **kwargs)  # type: ignore[arg-type]
    reference = build_ulam(IdentityMap(), rectangular, **kwargs)  # type: ignore[arg-type]
    np.testing.assert_array_equal(identity.toarray(), reference.toarray())
    np.testing.assert_array_equal(identity.escape_probabilities, reference.escape_probabilities)

    with pytest.raises(ValidationError, match="closed-system image escaped"):
        build_ulam(EscapingMap(), foreign, samples_per_cell=8, seed=1)


@pytest.mark.parametrize(
    ("model", "partition", "message"),
    [
        (
            object(),
            UniformPartition(bounds=((0.0, 1.0),), shape=(2,)),
            r"model must implement the ClassicalMap protocol; got object; missing "
            r"bounds, is_periodic, jacobian, state_dim, step",
        ),
        (
            IdentityMap(),
            object(),
            r"partition must implement the Partition protocol; got object; missing "
            r"cell_bounds, locate, ndim, ravel_index, sample, shape, size, unravel_index",
        ),
    ],
)
def test_build_ulam_reports_protocol_violations_not_attribute_errors(
    model: Any,
    partition: Any,
    message: str,
) -> None:
    """``validate_protocol`` promises this of every public entry point.

    ``build_ulam`` used to read ``model.state_dim`` straight away, so a foreign
    object surfaced as ``AttributeError: 'object' object has no attribute
    'state_dim'`` -- exactly the bare ``AttributeError`` the helper's docstring
    says never happens -- and a foreign partition was never checked at all.
    """
    with pytest.raises(ValidationError, match=message):
        build_ulam(model, partition, samples_per_cell=4, seed=1)


def test_model_partition_dimension_must_match() -> None:
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(2, 2),
        periodic=(True, True),
    )
    with pytest.raises(ValidationError, match="state_dim 1 must equal partition ndim 2"):
        build_ulam(IdentityMap(), partition, samples_per_cell=4, seed=1)

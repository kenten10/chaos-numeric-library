from __future__ import annotations

import numpy as np
import pytest

from chaos_numerics.core import Partition, ValidationError
from chaos_numerics.operators import RectangularPartition, UniformPartition


def floats(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    return np.asarray(value, dtype=np.float64)


def test_uniform_partition_properties_and_protocol() -> None:
    partition = UniformPartition(
        bounds=((0.0, 1.0), (-1.0, 1.0)),
        shape=(2, 4),
        periodic=(True, False),
    )

    assert partition.ndim == 2
    assert partition.shape == (2, 4)
    assert partition.size == 8
    assert partition.bounds == ((0.0, 1.0), (-1.0, 1.0))
    assert isinstance(partition, Partition)
    assert not partition.edges[0].flags.writeable


def test_internal_cut_is_right_owned_and_nonperiodic_upper_is_outside() -> None:
    partition = UniformPartition(bounds=((0.0, 1.0),), shape=(4,))
    left_of_cut = np.nextafter(0.25, 0.0)

    assert int(partition.locate(floats([left_of_cut]))) == 0
    assert int(partition.locate(floats([0.25]))) == 1
    assert int(partition.locate(floats([0.0]))) == 0
    with pytest.raises(ValidationError, match=r"must lie in \[0.0, 1.0\)"):
        partition.locate(floats([1.0]))


def test_periodic_boundary_wraps_upper_and_adjacent_values() -> None:
    partition = UniformPartition(
        bounds=((0.0, 1.0),),
        shape=(4,),
        periodic=(True,),
    )

    assert int(partition.locate(floats([1.0]))) == 0
    assert int(partition.locate(floats([-0.1]))) == 3
    assert int(partition.locate(floats([np.nextafter(1.0, 0.0)]))) == 3
    assert int(partition.locate(floats([np.nextafter(1.0, np.inf)]))) == 0
    assert int(partition.locate(floats([1.0 + 1_000_000.0]))) == 0


def test_periodic_wrap_keeps_values_a_rounding_error_below_the_lower_edge() -> None:
    """``np.mod(-1e-17, 1.0)`` rounds up to exactly ``1.0`` and escaped the domain.

    ``locate`` reduces periodic coordinates with ``wrap_into_half_open`` so that a
    point one rounding error below the lower edge lands in the first cell instead
    of raising for an index one past the last one.
    """
    square = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(4, 4),
        periodic=(True, True),
    )
    np.testing.assert_array_equal(
        square.locate(floats([[-1e-17, 0.5]])),
        np.asarray([2], dtype=np.intp),
    )

    unit = UniformPartition(bounds=((0.0, 1.0),), shape=(4,), periodic=(True,))
    assert int(unit.locate(floats([-1e-17]))) == 0
    assert int(unit.locate(floats([np.nextafter(0.0, -np.inf)]))) == 0

    shifted = UniformPartition(bounds=((-2.0, 3.0),), shape=(5,), periodic=(True,))
    assert int(shifted.locate(floats([-2.0 - 1e-17]))) == 0
    assert int(shifted.locate(floats([-2.0]))) == 0
    assert int(shifted.locate(floats([3.0]))) == 0


def test_two_dimensional_batch_locate_uses_c_order() -> None:
    partition = UniformPartition(
        bounds=((0.0, 1.0), (0.0, 1.0)),
        shape=(2, 3),
    )
    points = floats([[[0.1, 0.1], [0.1, 0.5]], [[0.6, 0.1], [0.6, 0.9]]])

    located = partition.locate(points)

    np.testing.assert_array_equal(located, np.asarray([[0, 1], [3, 5]], dtype=np.intp))
    assert located.dtype == np.dtype(np.intp)


def test_flat_multi_index_round_trip_is_exhaustive() -> None:
    partition = UniformPartition(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(3, 4))
    flat = np.arange(partition.size, dtype=np.intp)

    multi = partition.unravel_index(flat)

    np.testing.assert_array_equal(partition.ravel_index(multi), flat)
    np.testing.assert_array_equal(multi[0], np.asarray([0, 0], dtype=np.intp))
    np.testing.assert_array_equal(multi[-1], np.asarray([2, 3], dtype=np.intp))
    with pytest.raises(ValidationError, match="coordinate 1"):
        partition.ravel_index(np.asarray([0, 4], dtype=np.intp))
    with pytest.raises(ValidationError, match=r"\[0, 12\)"):
        partition.unravel_index(np.asarray(12, dtype=np.intp))


def test_rectangular_nonuniform_edges_and_cell_bounds() -> None:
    partition = RectangularPartition(
        edges=(floats([0.0, 0.2, 1.0]), floats([-1.0, 0.0, 0.5, 1.0])),
    )

    assert partition.shape == (2, 3)
    assert int(partition.locate(floats([0.2, 0.5]))) == 5
    np.testing.assert_array_equal(
        partition.cell_bounds(5),
        floats([[0.2, 1.0], [0.5, 1.0]]),
    )


def test_seeded_samples_are_reproducible_and_inside_every_cell() -> None:
    partition = UniformPartition(
        bounds=((-2.0, 2.0), (10.0, 11.0)),
        shape=(3, 2),
    )
    for index in range(partition.size):
        first = partition.sample(index, 64, seed=100 + index)
        second = partition.sample(index, 64, seed=100 + index)
        bounds = partition.cell_bounds(index)

        np.testing.assert_array_equal(first, second)
        assert first.shape == (64, 2)
        assert bool(np.all(first >= bounds[:, 0]))
        assert bool(np.all(first < bounds[:, 1]))
        np.testing.assert_array_equal(partition.locate(first), np.full(64, index, dtype=np.intp))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UniformPartition(bounds=((0.0, 1.0),), shape=(0,)),
        lambda: UniformPartition(bounds=((1.0, 0.0),), shape=(2,)),
        lambda: UniformPartition(bounds=((0.0, 1.0),), shape=(2,), periodic=(True, False)),
        lambda: RectangularPartition(edges=(floats([0.0, 0.5, 0.5, 1.0]),)),
        lambda: RectangularPartition(edges=(floats([0.0]),)),
    ],
)
def test_partition_construction_is_validated(factory: object) -> None:
    with pytest.raises(ValidationError):
        factory()  # type: ignore[operator]


def test_point_and_sampling_inputs_are_validated() -> None:
    partition = UniformPartition(bounds=((0.0, 1.0),), shape=(2,))

    with pytest.raises(ValidationError, match="trailing dimension 1"):
        partition.locate(floats([0.1, 0.2]))
    with pytest.raises(ValidationError, match="finite"):
        partition.locate(floats([np.nan]))
    with pytest.raises(ValidationError, match="count must be positive"):
        partition.sample(0, 0)
    with pytest.raises(ValidationError, match="seed must be a non-negative integer"):
        partition.sample(0, 2, seed=True)


@pytest.mark.parametrize(
    ("periodic", "points"),
    [
        ((False,), [[-1.5], [-1e-18], [0.0], [0.25], [0.5], [0.999], [1.0], [1.25], [2.0]]),
        ((True,), [[-1.5], [-1e-18], [0.0], [0.25], [0.5], [0.999], [1.0], [1.25], [2.0]]),
        (
            (True, False),
            [[0.5, 0.5], [1.5, 0.5], [0.5, 1.0], [1.5, -0.25], [-0.5, 0.999], [2.25, 1.75]],
        ),
        (
            (False, True),
            [[0.5, 0.5], [1.5, 0.5], [0.5, 1.0], [1.0, 2.5], [-1e-18, -1e-18], [0.999, -3.5]],
        ),
    ],
)
def test_inside_mask_agrees_with_locate_point_by_point(
    periodic: tuple[bool, ...],
    points: list[list[float]],
) -> None:
    """The batch admission test must accept exactly what ``locate`` accepts.

    ``build_ulam(open_system=True)`` uses ``_inside_mask`` to decide which sampled
    images escaped and then calls ``locate`` once on the rest, so any disagreement
    between the two is either a miscounted escape or a ``ValidationError`` raised
    from a code path that is supposed to tolerate escapes. Random sampling almost
    never lands on a domain edge, which is why the agreement is pinned here on the
    endpoints and wrap-around values directly rather than through a built matrix.
    """
    ndim = len(periodic)
    partition = UniformPartition(bounds=((0.0, 1.0),) * ndim, shape=(4,) * ndim, periodic=periodic)
    batch = floats(points)

    def accepted(point: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> bool:
        try:
            partition.locate(point)
        except ValidationError:
            return False
        return True

    expected = np.asarray([accepted(point) for point in batch])
    mask = partition._inside_mask(batch)

    assert bool(np.any(expected)), "case must accept at least one point"
    assert all(periodic) or not bool(np.all(expected)), "a bounded axis must reject something"
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(
        partition.locate(batch[mask]),
        np.asarray([int(partition.locate(point)) for point in batch[mask]]),
    )


def test_inside_mask_rejects_non_finite_coordinates_like_locate_does() -> None:
    """``locate`` refuses a non-finite point regardless of periodicity, so the mask does."""
    partition = UniformPartition(bounds=((0.0, 1.0),), shape=(4,), periodic=(True,))
    batch = floats([[0.5], [np.nan], [np.inf], [-np.inf]])

    np.testing.assert_array_equal(
        partition._inside_mask(batch), np.asarray([True, False, False, False])
    )
    with pytest.raises(ValidationError, match="finite"):
        partition.locate(batch)

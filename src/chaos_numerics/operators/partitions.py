"""One- and two-dimensional rectangular partitions with C-order indices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from chaos_numerics.core._validation import as_float_array, validate_bounds, validate_shape
from chaos_numerics.core.exceptions import ValidationError
from chaos_numerics.core.types import ArrayLike, FloatArray, IndexArray


@dataclass(frozen=True, slots=True, eq=False, init=False)
class RectangularPartition:
    """A partition defined by strictly increasing coordinate-edge arrays.

    ``locate`` returns C-order flat indices. Exact internal cuts belong to the
    positive/right cell. Non-periodic upper endpoints are outside the domain;
    periodic upper endpoints wrap to the lower endpoint.
    """

    edges: tuple[FloatArray, ...]
    periodic: tuple[bool, ...]

    def __init__(
        self,
        edges: Sequence[ArrayLike],
        periodic: Sequence[bool] | None = None,
    ) -> None:
        edge_arrays = tuple(_edge_array(edge, coordinate=index) for index, edge in enumerate(edges))
        if len(edge_arrays) not in {1, 2}:
            raise ValidationError(f"partition must have 1 or 2 dimensions; got {len(edge_arrays)}")
        flags = _periodic_flags(periodic, ndim=len(edge_arrays))
        object.__setattr__(self, "edges", edge_arrays)
        object.__setattr__(self, "periodic", flags)

    @property
    def ndim(self) -> int:
        return len(self.edges)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(edge.size - 1) for edge in self.edges)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.intp))

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((float(edge[0]), float(edge[-1])) for edge in self.edges)

    def locate(self, points: ArrayLike, /) -> IndexArray:
        """Locate scalar or batched points and return C-order flat indices."""
        values = as_float_array(points, name="points", trailing_dim=self.ndim, copy=True)
        multi = np.empty(values.shape, dtype=np.intp)
        for coordinate, (edge, periodic) in enumerate(zip(self.edges, self.periodic, strict=True)):
            lower = edge[0]
            upper = edge[-1]
            coordinate_values = values[..., coordinate]
            if periodic:
                coordinate_values = lower + np.mod(coordinate_values - lower, upper - lower)
            elif bool(np.any((coordinate_values < lower) | (coordinate_values >= upper))):
                raise ValidationError(
                    f"points coordinate {coordinate} must lie in [{lower}, {upper})"
                )
            indices = np.searchsorted(edge, coordinate_values, side="right") - 1
            multi[..., coordinate] = indices.astype(np.intp, copy=False)
        return self.ravel_index(multi)

    def ravel_index(self, multi_index: ArrayLike, /) -> IndexArray:
        """Convert ``(..., ndim)`` multi-indices to C-order flat indices."""
        indices = _index_array(multi_index, name="multi_index")
        if indices.ndim == 0 or indices.shape[-1] != self.ndim:
            raise ValidationError(
                f"multi_index must have trailing dimension {self.ndim}; got {indices.shape}"
            )
        flat = np.zeros(indices.shape[:-1], dtype=np.intp)
        for coordinate, extent in enumerate(self.shape):
            current = indices[..., coordinate]
            if bool(np.any((current < 0) | (current >= extent))):
                raise ValidationError(
                    f"multi_index coordinate {coordinate} must lie in [0, {extent})"
                )
            flat = flat * extent + current
        return flat

    def unravel_index(self, flat_index: ArrayLike, /) -> IndexArray:
        """Convert scalar or batched flat indices to ``(..., ndim)`` indices."""
        flat = _index_array(flat_index, name="flat_index")
        if bool(np.any((flat < 0) | (flat >= self.size))):
            raise ValidationError(f"flat_index must lie in [0, {self.size})")
        remainder = flat.copy()
        result = np.empty((*flat.shape, self.ndim), dtype=np.intp)
        for coordinate in range(self.ndim - 1, -1, -1):
            extent = self.shape[coordinate]
            result[..., coordinate] = remainder % extent
            remainder //= extent
        return result

    def cell_bounds(self, index: int, /) -> FloatArray:
        """Return one flat-indexed cell's ``(ndim, 2)`` half-open bounds."""
        flat = _scalar_index(index, name="index", upper=self.size)
        multi = self.unravel_index(np.asarray(flat, dtype=np.intp))
        result = np.empty((self.ndim, 2), dtype=np.float64)
        for coordinate, edge in enumerate(self.edges):
            cell = int(multi[coordinate])
            result[coordinate] = (edge[cell], edge[cell + 1])
        return result

    def sample(self, index: int, count: int, *, seed: int | None = None) -> FloatArray:
        """Draw local-RNG uniform samples from one half-open cell."""
        sample_count = _positive_int(count, name="count")
        if seed is not None:
            seed = _nonnegative_int(seed, name="seed")
        bounds = self.cell_bounds(index)
        generator = np.random.default_rng(seed)
        unit = generator.random((sample_count, self.ndim), dtype=np.float64)
        return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RectangularPartition):
            return NotImplemented
        return self.periodic == other.periodic and all(
            np.array_equal(left, right) for left, right in zip(self.edges, other.edges, strict=True)
        )


class UniformPartition(RectangularPartition):
    """An equal-width rectangular partition constructed from bounds and shape."""

    __slots__ = ()

    def __init__(
        self,
        bounds: ArrayLike,
        shape: Sequence[int | np.integer[Any]],
        periodic: Sequence[bool] | None = None,
    ) -> None:
        partition_shape = validate_shape(shape, name="shape")
        if len(partition_shape) not in {1, 2}:
            raise ValidationError(
                f"UniformPartition must have 1 or 2 dimensions; got {len(partition_shape)}"
            )
        partition_bounds = validate_bounds(
            bounds,
            state_dim=len(partition_shape),
            name="bounds",
        )
        edges = tuple(
            np.linspace(lower, upper, cells + 1, dtype=np.float64)
            for (lower, upper), cells in zip(partition_bounds, partition_shape, strict=True)
        )
        flags = None if periodic is None else tuple(periodic)
        super().__init__(edges=edges, periodic=flags)


def _edge_array(value: ArrayLike, *, coordinate: int) -> FloatArray:
    edge = as_float_array(value, name=f"edges[{coordinate}]", ndim=1, copy=True)
    if edge.size < 2:
        raise ValidationError(f"edges[{coordinate}] must contain at least two values")
    if bool(np.any(np.diff(edge) <= 0.0)):
        raise ValidationError(f"edges[{coordinate}] must be strictly increasing")
    edge.setflags(write=False)
    return edge


def _periodic_flags(value: Sequence[bool] | None, *, ndim: int) -> tuple[bool, ...]:
    if value is None:
        return (False,) * ndim
    flags = tuple(value)
    if len(flags) != ndim or not all(isinstance(flag, (bool, np.bool_)) for flag in flags):
        raise ValidationError(f"periodic must contain {ndim} boolean flags")
    return tuple(bool(flag) for flag in flags)


def _index_array(value: ArrayLike, *, name: str) -> IndexArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} could not be converted to an integer array") from error
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValidationError(f"{name} must contain integers; got dtype {raw.dtype}")
    if raw.dtype.kind == "u" and bool(np.any(raw > np.iinfo(np.intp).max)):
        raise ValidationError(f"{name} entries must fit in np.intp")
    return raw.astype(np.intp, copy=False)


def _scalar_index(value: object, *, name: str, upper: int) -> int:
    result = _nonnegative_int(value, name=name)
    if result >= upper:
        raise ValidationError(f"{name} must lie in [0, {upper})")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValidationError(f"{name} must be positive; got 0")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValidationError(f"{name} must be non-negative; got {result}")
    return result


__all__ = ["RectangularPartition", "UniformPartition"]

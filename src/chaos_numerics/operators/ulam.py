"""Monte Carlo Ulam discretization with column-stochastic orientation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix, issparse  # type: ignore[import-untyped]

from chaos_numerics.core import (
    ClassicalMap,
    ExperimentMetadata,
    Partition,
    ValidationError,
)
from chaos_numerics.core._payload import (
    ArrayPayload,
    add_convergence_history,
    metadata_payload,
)
from chaos_numerics.core._validation import (
    as_complex_array,
    as_float_array,
    validate_protocol,
)
from chaos_numerics.core.types import ArrayLike, ComplexArray, FloatArray, OperatorShape
from chaos_numerics.operators.partitions import RectangularPartition

# Column sums are float64 sums of at most ``cells`` exact multiples of
# ``1 / samples_per_cell``, so their rounding error is orders of magnitude below
# this bound; the bound exists to reject genuinely superstochastic input, not to
# absorb sampling noise.
_COLUMN_SUM_TOLERANCE = 1e-9


def _float_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(precision="float64")


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class UlamMatrix:
    """Immutable column-stochastic transition matrix with escape diagnostics.

    ``matrix`` stores ``P[target, source] = Pr(target | source)`` as canonical
    CSR, ``escape_probabilities[source]`` is the sampled mass that left the
    partition from that column -- so a nonzero entry is the signature of an open
    system -- and ``metadata`` records how the matrix was built.

    The dataclass is frozen and the three CSR arrays and ``escape_probabilities``
    are read-only, so an operator and its diagnostics can never disagree. Use
    :meth:`array_payload` when writable arrays are needed.

    The read-only flags stop in-place writes, including ``operator.matrix[i, j] =
    x`` on a stored entry. They cannot stop a caller from rebinding the arrays of
    the stored SciPy object -- ``operator.matrix.data = ...``, or a SciPy method
    that rebuilds them such as ``setdiag`` at a position with no stored entry --
    any more than a read-only NumPy array survives ``setflags(write=True)``.
    ``.matrix`` is read-only storage by contract; mutating it through those
    routes leaves ``escape_probabilities`` and ``metadata`` describing a matrix
    that no longer exists, which is the whole reason not to do it.

    Only a small read-only surface is forwarded from the stored matrix:
    :attr:`shape`, :attr:`dtype`, :attr:`nnz`, :meth:`matvec`, :meth:`toarray`,
    and ``@`` against a dense vector or matrix. Everything else -- ``sum``,
    ``copy``, ``astype``, ``multiply``, ``+``, ``-``, scalar multiplication,
    transposition, ``format`` -- is reached through ``.matrix``, which keeps the
    frozen v0.1 surface of this class small instead of promising the whole SciPy
    sparse API.

    Perform SciPy operations through ``.matrix``. Diagnostics do not propagate
    through those operations, so read what you need off the operator first and
    carry it yourself::

        escape = operator.escape_probabilities  # keep the diagnostics you need
        product = operator.matrix @ operator.matrix  # a plain SciPy csr_matrix

    A zeroed escape vector on a derived operator never means the system became
    closed; it means nobody carried the diagnostics forward. Note also that the
    escape vector of a product is not the escape vector of either operand, so
    re-wrapping a derived matrix with the original vector is wrong unless the
    derived quantity really does describe the same one-step mass loss.
    """

    matrix: csr_matrix
    escape_probabilities: FloatArray
    metadata: ExperimentMetadata = field(default_factory=_float_metadata)

    def __post_init__(self) -> None:
        matrix = _frozen_csr(self.matrix)
        columns = int(matrix.shape[1])
        escape = as_float_array(
            self.escape_probabilities,
            name="escape_probabilities",
            ndim=1,
            copy=True,
        )
        if escape.shape != (columns,):
            raise ValidationError(
                f"escape_probabilities must have shape ({columns},); got {escape.shape}"
            )
        if bool(np.any((escape < 0.0) | (escape > 1.0))):
            raise ValidationError("escape_probabilities must be finite values in [0, 1]")
        if not isinstance(self.metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        escape.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "escape_probabilities", escape)

    @property
    def shape(self) -> OperatorShape:
        """Two-dimensional operator shape ``(cells, cells)``."""
        rows, columns = self.matrix.shape
        return (int(rows), int(columns))

    @property
    def dtype(self) -> np.dtype[Any]:
        """Element dtype of the stored transition probabilities, always ``float64``."""
        return cast("np.dtype[Any]", np.dtype(self.matrix.dtype))

    @property
    def nnz(self) -> int:
        """Number of stored transitions."""
        return int(self.matrix.nnz)

    def matvec(self, vector: ArrayLike, /) -> ComplexArray:
        """Apply the operator to one vector.

        This is the :class:`~chaos_numerics.core.LinearOperatorLike` entry point,
        and that protocol fixes the result dtype at ``complex128``. Use ``@`` or
        ``.matrix`` when the natural ``float64`` product is wanted; the values
        are identical either way.
        """
        values = as_complex_array(vector, name="vector", ndim=1)
        if values.shape != (self.shape[1],):
            raise ValidationError(f"vector must have shape ({self.shape[1]},); got {values.shape}")
        return cast(ComplexArray, np.asarray(self.matrix @ values, dtype=np.complex128))

    def toarray(self) -> FloatArray:
        """Return an independent writable dense copy of the transition matrix."""
        return cast(FloatArray, np.asarray(self.matrix.toarray(), dtype=np.float64))

    def __matmul__(self, other: Any) -> Any:
        """Multiply by a dense vector or matrix; identical to ``self.matrix @ other``.

        Only the left-hand form is forwarded. Write ``vector @ operator.matrix``
        for the right-hand product.
        """
        return self.matrix @ other

    def __reduce__(self) -> tuple[type[UlamMatrix], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only.

        The ``__setstate__`` generated for a ``slots=True`` dataclass restores the
        fields directly, bypassing ``__post_init__``, and NumPy drops
        ``writeable=False`` when an array is pickled. Without this hook a
        ``pickle`` or :func:`copy.deepcopy` round trip would hand back a mutable
        operator whose diagnostics could then drift from its values.
        """
        return (self.__class__, (self.matrix, self.escape_probabilities, self.metadata))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UlamMatrix):
            return NotImplemented
        return (
            self.shape == other.shape
            and np.array_equal(self.matrix.data, other.matrix.data)
            and np.array_equal(self.matrix.indices, other.matrix.indices)
            and np.array_equal(self.matrix.indptr, other.matrix.indptr)
            and np.array_equal(self.escape_probabilities, other.escape_probabilities)
            and self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        converged = (
            self.metadata.convergence.converged if self.metadata.convergence is not None else None
        )
        return (
            f"UlamMatrix(shape={self.shape}, dtype={self.dtype.name!r}, nnz={self.nnz}, "
            f"escape_max={float(np.max(self.escape_probabilities)):.3g}, "
            f"warnings={len(self.metadata.warnings)}, converged={converged})"
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable arrays ready for NPZ-like persistence.

        ``csr_matrix((data, indices, indptr), shape=tuple(shape))`` reconstructs
        the stored matrix exactly.
        """
        payload: ArrayPayload = {
            "data": self.matrix.data.copy(),
            "indices": self.matrix.indices.copy(),
            "indptr": self.matrix.indptr.copy(),
            "shape": np.asarray(self.shape, dtype=np.intp),
            "escape_probabilities": self.escape_probabilities.copy(),
        }
        add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return JSON metadata and array descriptors without array contents."""
        arrays: ArrayPayload = {
            "data": self.matrix.data,
            "indices": self.matrix.indices,
            "indptr": self.matrix.indptr,
            "shape": np.asarray(self.shape, dtype=np.intp),
            "escape_probabilities": self.escape_probabilities,
        }
        add_convergence_history(arrays, self.metadata, copy=False)
        return metadata_payload("ulam_matrix", self.metadata, arrays, shape=list(self.shape))


def _frozen_csr(value: Any) -> Any:
    """Return an owned canonical CSR copy whose three arrays are read-only.

    Copying is what makes the immutability real: freezing the caller's arrays in
    place would silently change an object the caller still owns. Canonicalizing
    before freezing is required rather than optional, because ``sum_duplicates``
    and ``sort_indices`` write into ``data`` and can therefore never run on a
    frozen matrix.
    """
    if not issparse(value):
        raise ValidationError(
            "matrix must be a SciPy sparse matrix; wrap a dense array with "
            f"scipy.sparse.csr_matrix first, got {type(value).__name__}"
        )
    shape = tuple(int(extent) for extent in value.shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValidationError(f"matrix must be square and two-dimensional; got shape {value.shape}")
    if shape[0] < 1:
        raise ValidationError("matrix dimension must be positive")

    source = value.tocsr()
    prepared = csr_matrix(
        (
            as_float_array(source.data, name="matrix data", ndim=1, copy=True),
            np.array(source.indices, copy=True),
            np.array(source.indptr, copy=True),
        ),
        shape=shape,
    )
    prepared.sum_duplicates()
    prepared.sort_indices()
    if prepared.nnz and float(np.min(prepared.data)) < 0.0:
        raise ValidationError(
            "matrix must contain only non-negative transition probabilities; got "
            f"minimum {float(np.min(prepared.data))}"
        )
    # Column stochasticity itself is not required: an open system loses mass, and
    # ``escape_probabilities`` is exactly what records it. Substochasticity is,
    # because a column that sums above one is not a probability distribution over
    # targets under any reading, and it survives the documented workflow of
    # combining operators through ``.matrix`` (a product of substochastic
    # matrices stays substochastic).
    column_sums = np.asarray(prepared.sum(axis=0), dtype=np.float64).reshape(-1)
    largest = float(np.max(column_sums))
    if largest > 1.0 + _COLUMN_SUM_TOLERANCE:
        raise ValidationError(
            "matrix must be column-substochastic: no column sum may exceed one, but the "
            f"largest column sum is {largest}"
        )

    matrix = csr_matrix(
        (
            np.array(prepared.data, copy=True),
            np.array(prepared.indices, copy=True),
            np.array(prepared.indptr, copy=True),
        ),
        shape=shape,
    )
    matrix.data.setflags(write=False)
    matrix.indices.setflags(write=False)
    matrix.indptr.setflags(write=False)
    return matrix


def build_ulam(
    model: ClassicalMap,
    partition: Partition,
    *,
    samples_per_cell: int,
    seed: int | None = None,
    batch_size: int | None = None,
    open_system: bool = False,
) -> UlamMatrix:
    """Build ``P[target, source]`` as CSR from seeded cell sampling.

    Each source cell owns an independently derived child seed. ``batch_size``
    controls only model-evaluation memory and does not change sampled points.
    """
    samples, chunk_size, effective_seed = _validate_build(
        model,
        partition,
        samples_per_cell=samples_per_cell,
        seed=seed,
        batch_size=batch_size,
        open_system=open_system,
    )
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    escape = np.zeros(partition.size, dtype=np.float64)

    for source in range(partition.size):
        points = partition.sample(
            source,
            samples,
            seed=_child_seed(effective_seed, source),
        )
        counts = np.zeros(partition.size, dtype=np.int64)
        escaped = 0
        for start in range(0, samples, chunk_size):
            images = _step_batch(model, points[start : start + chunk_size], partition.ndim)
            if open_system:
                escaped += _accumulate_open(partition, images, counts)
            else:
                try:
                    targets = partition.locate(images).reshape(-1)
                except ValidationError as error:
                    raise ValidationError(
                        f"closed-system image escaped the partition from source cell {source}; "
                        "pass open_system=True to record mass loss"
                    ) from error
                counts += np.bincount(targets, minlength=partition.size)
        nonzero = np.flatnonzero(counts)
        rows.extend(int(target) for target in nonzero)
        columns.extend([source] * int(nonzero.size))
        data.extend(float(counts[target] / samples) for target in nonzero)
        escape[source] = escaped / samples

    metadata = ExperimentMetadata(
        parameters={
            "model": type(model).__name__,
            "model_parameters": _model_parameters(model),
            "partition": type(partition).__name__,
            "partition_shape": partition.shape,
            "orientation": "P[target, source]",
            "samples_per_cell": samples,
            "batch_size": chunk_size,
            "open_system": open_system,
            "seed_derivation": "numpy.SeedSequence(root, spawn_key=(source_cell,))",
            "parallelization_boundary": "source cells are independent; v0.1 executes serially",
        },
        seed=effective_seed,
    )
    # ``UlamMatrix`` canonicalizes and freezes its own copy of the CSR arrays, so
    # no sum_duplicates/sort_indices call is needed here.
    matrix = csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (np.asarray(rows, dtype=np.intp), np.asarray(columns, dtype=np.intp)),
        ),
        shape=(partition.size, partition.size),
    )
    return UlamMatrix(matrix, escape, metadata)


def _build_ulam_dense_reference(
    model: ClassicalMap,
    partition: Partition,
    *,
    samples_per_cell: int,
    seed: int,
    open_system: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Small independent loop reference used only by correctness tests."""
    samples, _, effective_seed = _validate_build(
        model,
        partition,
        samples_per_cell=samples_per_cell,
        seed=seed,
        batch_size=1,
        open_system=open_system,
    )
    matrix = np.zeros((partition.size, partition.size), dtype=np.float64)
    escape = np.zeros(partition.size, dtype=np.float64)
    for source in range(partition.size):
        points = partition.sample(source, samples, seed=_child_seed(effective_seed, source))
        for point in points:
            image = _step_batch(model, point[None, :], partition.ndim)[0]
            try:
                target = int(partition.locate(image))
            except ValidationError as error:
                if not open_system:
                    raise ValidationError(
                        f"closed-system image escaped the partition from source cell {source}"
                    ) from error
                escape[source] += 1.0 / samples
            else:
                matrix[target, source] += 1.0 / samples
    return matrix, escape


def _accumulate_open(
    partition: Partition,
    images: FloatArray,
    counts: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
) -> int:
    """Bin one batch of images into ``counts`` and return how many escaped.

    An open system has to distinguish "this point left the partition" from "this
    batch is invalid", and :meth:`Partition.locate` reports both the same way: it
    raises as soon as any point is out of domain. Asking it one point at a time is
    the only thing the protocol alone allows, and it made ``open_system=True``
    roughly forty times slower than the closed path at 32x32 cells.

    A :class:`~chaos_numerics.operators.RectangularPartition` can answer the
    admission test for a whole batch at once, so the escape mask is computed
    vectorized and ``locate`` is called once on the surviving rows. Because both
    routes evaluate the same elementwise arithmetic, the counts are identical bit
    for bit; the per-point loop remains for a user-defined ``Partition``, which
    exposes no batch admission test.
    """
    size = int(partition.size)
    if isinstance(partition, RectangularPartition):
        inside = partition._inside_mask(images)
        surviving = int(np.count_nonzero(inside))
        if surviving:
            targets = partition.locate(images[inside]).reshape(-1)
            counts += np.bincount(targets, minlength=size)
        return int(images.shape[0]) - surviving

    escaped = 0
    for image in images:
        try:
            target = int(partition.locate(image))
        except ValidationError:
            escaped += 1
        else:
            counts[target] += 1
    return escaped


def _step_batch(model: ClassicalMap, points: FloatArray, dimension: int) -> FloatArray:
    images = as_float_array(
        model.step(points),
        name="model step result",
        ndim=2,
        trailing_dim=dimension,
    )
    if images.shape != points.shape:
        raise ValidationError(
            f"model step must preserve sample shape {points.shape}; got {images.shape}"
        )
    return images


def _validate_build(
    model: ClassicalMap,
    partition: Partition,
    *,
    samples_per_cell: object,
    seed: object,
    batch_size: object,
    open_system: object,
) -> tuple[int, int, int]:
    # Both protocol checks come first so that a foreign object is named as a
    # protocol violation here rather than as an AttributeError from the
    # ``model.state_dim`` read one line below.
    validate_protocol(model, ClassicalMap, name="model")
    validate_protocol(partition, Partition, name="partition")
    if model.state_dim != partition.ndim:
        raise ValidationError(
            f"model state_dim {model.state_dim} must equal partition ndim {partition.ndim}"
        )
    if partition.size <= 0 or len(partition.shape) != partition.ndim:
        raise ValidationError("partition shape, size, and ndim are inconsistent or empty")
    samples = _positive_int(samples_per_cell, name="samples_per_cell")
    chunk = samples if batch_size is None else _positive_int(batch_size, name="batch_size")
    if seed is None:
        effective_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    else:
        effective_seed = _nonnegative_int(seed, name="seed")
    if not isinstance(open_system, bool):
        raise ValidationError(f"open_system must be a bool; got {open_system!r}")
    return samples, min(chunk, samples), effective_seed


def _child_seed(root_seed: int, source: int) -> int:
    sequence = np.random.SeedSequence(root_seed, spawn_key=(source,))
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _model_parameters(model: ClassicalMap) -> dict[str, object]:
    parameters = getattr(model, "parameters", None)
    return dict(parameters) if isinstance(parameters, Mapping) else {}


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


__all__ = ["UlamMatrix", "build_ulam"]

"""Monte Carlo Ulam discretization with column-stochastic orientation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from chaos_numerics.core import ClassicalMap, ExperimentMetadata, Partition, ValidationError
from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.types import FloatArray


class UlamMatrix(csr_matrix):  # type: ignore[misc]
    """CSR transition matrix with escape diagnostics and build metadata."""

    escape_probabilities: FloatArray
    metadata: ExperimentMetadata

    def __init__(
        self,
        arg1: Any,
        shape: tuple[int, int] | None = None,
        *,
        escape_probabilities: FloatArray | None = None,
        metadata: ExperimentMetadata | None = None,
    ) -> None:
        super().__init__(arg1, shape=shape, dtype=np.float64)
        escape = (
            np.zeros(self.shape[1], dtype=np.float64)
            if escape_probabilities is None
            else np.asarray(escape_probabilities, dtype=np.float64).copy()
        )
        if escape.shape != (self.shape[1],):
            raise ValidationError(
                f"escape_probabilities must have shape ({self.shape[1]},); got {escape.shape}"
            )
        if not bool(np.all(np.isfinite(escape))) or bool(np.any((escape < 0.0) | (escape > 1.0))):
            raise ValidationError("escape_probabilities must be finite values in [0, 1]")
        if metadata is not None and not isinstance(metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        escape.setflags(write=False)
        self.escape_probabilities = escape
        self.metadata = ExperimentMetadata() if metadata is None else metadata


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
                for image in images:
                    try:
                        target = int(partition.locate(image))
                    except ValidationError:
                        escaped += 1
                    else:
                        counts[target] += 1
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
    matrix = UlamMatrix(
        (
            np.asarray(data, dtype=np.float64),
            (np.asarray(rows, dtype=np.intp), np.asarray(columns, dtype=np.intp)),
        ),
        shape=(partition.size, partition.size),
        escape_probabilities=escape,
        metadata=metadata,
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


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

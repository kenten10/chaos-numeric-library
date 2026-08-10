"""Poincare-section point clouds for discrete classical maps.

A discrete map is already a section of the underlying flow, so the "section" of a
map is the scatter of visited states: many trajectories, many times, projected
onto a few coordinates and flattened into one ``(n_points, k)`` array ready for
``matplotlib.pyplot.scatter``. Trajectory generation itself is delegated to
:func:`~chaos_numerics.classical.iterate`; this module only chooses the initial
ensemble, discards the transient, and reshapes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from chaos_numerics.classical.trajectories import iterate
from chaos_numerics.core import ClassicalMap, ExperimentMetadata, ValidationError
from chaos_numerics.core._validation import as_float_array, validate_protocol
from chaos_numerics.core.results import SCHEMA_VERSION, ArrayPayload
from chaos_numerics.core.types import ArrayLike, FloatArray

_RESULT_TYPE = "poincare_section"


@dataclass(frozen=True, slots=True, eq=False)
class PoincareSection:
    """Immutable flattened section points and the coordinates they were taken from.

    ``points`` has shape ``(n_points, len(coordinates))`` and is read-only, like
    every other array exposed by a result type in this library;
    :meth:`array_payload` hands back writable copies.
    """

    points: FloatArray
    coordinates: tuple[int, ...]
    metadata: ExperimentMetadata

    def __post_init__(self) -> None:
        points = _readonly_float(self.points, name="section points")
        coordinates = tuple(int(index) for index in self.coordinates)
        if points.shape[1] != len(coordinates):
            raise ValidationError(
                f"section points must have one column per coordinate; got shape {points.shape} "
                f"for coordinates {coordinates}"
            )
        if not isinstance(self.metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def count(self) -> int:
        """Number of stored section points."""
        return int(self.points.shape[0])

    def __reduce__(self) -> tuple[type[PoincareSection], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only."""
        return (self.__class__, (self.points, self.coordinates, self.metadata))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoincareSection):
            return NotImplemented
        return (
            np.array_equal(self.points, other.points)
            and self.coordinates == other.coordinates
            and self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        return (
            f"PoincareSection(count={self.count}, coordinates={self.coordinates}, "
            f"warnings={len(self.metadata.warnings)})"
        )

    def _arrays(self) -> ArrayPayload:
        arrays: ArrayPayload = {"points": self.points}
        convergence = self.metadata.convergence
        if convergence is not None and convergence.history is not None:
            arrays["convergence_history"] = convergence.history
        return arrays

    def array_payload(self) -> ArrayPayload:
        """Return independent writable arrays ready for NPZ-like persistence."""
        return {name: array.copy() for name, array in self._arrays().items()}

    def metadata_payload(self) -> dict[str, object]:
        """Return JSON metadata and array descriptors without array contents."""
        descriptors: dict[str, object] = {
            name: {"shape": list(array.shape), "dtype": array.dtype.name}
            for name, array in self._arrays().items()
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "result_type": _RESULT_TYPE,
            "metadata": self.metadata.to_dict(),
            "arrays": descriptors,
            "coordinates": list(self.coordinates),
        }


def poincare_section(
    model: ClassicalMap,
    initial_states: ArrayLike | None = None,
    *,
    steps: int,
    samples: int | None = None,
    transient: int = 0,
    coordinates: Sequence[int] | None = None,
    seed: int | None = None,
) -> PoincareSection:
    """Collect the states visited by an ensemble into one flattened point cloud.

    Supply either ``initial_states`` -- shape ``(state_dim,)`` or
    ``(count, state_dim)`` -- or ``samples``, which draws that many independent
    uniform points from ``model.bounds`` with a reproducible ``seed``. The two are
    mutually exclusive because they answer the same question in incompatible ways.

    ``transient`` steps are discarded, then ``steps + 1`` states per trajectory are
    recorded, counting the state reached at the end of the transient. The result
    therefore holds ``count * (steps + 1)`` rows, in trajectory-major order with
    time increasing inside each trajectory.

    ``coordinates`` selects the projected columns and defaults to the first two
    coordinates, or the only one when ``state_dim == 1``. The seed actually used is
    always recorded in ``metadata.seed``, so a section drawn without an explicit
    seed can still be reproduced exactly.
    """
    validate_protocol(model, ClassicalMap, name="model")
    recorded_steps = _nonnegative_int(steps, name="steps")
    if recorded_steps == 0:
        raise ValidationError("steps must be positive; a section of one time is not a section")
    discarded = _nonnegative_int(transient, name="transient")
    dimension = _model_dimension(model)
    columns = _coordinates(coordinates, dimension=dimension)
    starts, effective_seed = _initial_ensemble(
        model,
        initial_states,
        samples=samples,
        seed=seed,
        dimension=dimension,
    )

    trajectory = iterate(model, starts, steps=discarded + recorded_steps, include_initial=True)
    selected = trajectory.states[:, discarded:, list(columns)]
    points = np.ascontiguousarray(selected.reshape(-1, len(columns)), dtype=np.float64)

    metadata = ExperimentMetadata(
        parameters={
            "model": type(model).__name__,
            "model_parameters": _model_parameters(model),
            "steps": recorded_steps,
            "transient": discarded,
            "trajectory_count": int(starts.shape[0]),
            "coordinates": columns,
            "seed": effective_seed,
            "initial_states": "supplied" if initial_states is not None else "uniform_in_bounds",
            "flattening": "trajectory-major, time increasing inside each trajectory",
        },
        precision="float64",
        seed=effective_seed,
    )
    return PoincareSection(points=points, coordinates=columns, metadata=metadata)


def _initial_ensemble(
    model: ClassicalMap,
    initial_states: ArrayLike | None,
    *,
    samples: object,
    seed: object,
    dimension: int,
) -> tuple[FloatArray, int | None]:
    if (initial_states is None) == (samples is None):
        raise ValidationError(
            "poincare_section needs exactly one of initial_states and samples: pass "
            "initial_states to place the ensemble yourself, or samples to draw that many "
            "uniform points from model.bounds; passing both would silently discard one of them"
        )
    if initial_states is not None:
        if seed is not None:
            raise ValidationError(
                "seed applies only to the samples path; supplied initial_states involve no "
                "randomness, so a seed here would be recorded without being used"
            )
        starts = as_float_array(
            initial_states,
            name="initial_states",
            trailing_dim=dimension,
            copy=True,
        )
        if starts.ndim == 1:
            starts = starts[None, :]
        if starts.ndim != 2 or starts.shape[0] == 0:
            raise ValidationError(
                "initial_states must have shape (state_dim,) or (count, state_dim) with at "
                f"least one state; got {starts.shape}"
            )
        return starts, None

    count = _positive_int(samples, name="samples")
    effective_seed = (
        int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
        if seed is None
        else _nonnegative_int(seed, name="seed")
    )
    generator = np.random.default_rng(effective_seed)
    unit = generator.random((count, dimension), dtype=np.float64)
    lower = np.asarray([bound[0] for bound in model.bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in model.bounds], dtype=np.float64)
    if lower.shape != (dimension,) or not bool(np.all(np.isfinite(lower) & np.isfinite(upper))):
        raise ValidationError("model bounds must be finite and match state_dim")
    return lower + unit * (upper - lower), effective_seed


def _coordinates(value: Sequence[int] | None, *, dimension: int) -> tuple[int, ...]:
    if value is None:
        return tuple(range(min(2, dimension)))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(f"coordinates must be a sequence of integers; got {value!r}")
    columns = tuple(_nonnegative_int(index, name="coordinates entry") for index in value)
    if not columns:
        raise ValidationError("coordinates must select at least one coordinate")
    outside = [index for index in columns if index >= dimension]
    if outside:
        raise ValidationError(
            f"coordinates must lie in [0, {dimension}) for this model; got {outside}"
        )
    return columns


def _model_dimension(model: ClassicalMap) -> int:
    dimension = model.state_dim
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise ValidationError(f"model state_dim must be a positive integer; got {dimension!r}")
    result = int(dimension)
    if result <= 0:
        raise ValidationError(f"model state_dim must be positive; got {result}")
    if len(model.bounds) != result:
        raise ValidationError("model bounds must match state_dim")
    return result


def _model_parameters(model: ClassicalMap) -> dict[str, object]:
    parameters = getattr(model, "parameters", None)
    return dict(parameters) if isinstance(parameters, Mapping) else {}


def _readonly_float(value: FloatArray, *, name: str) -> FloatArray:
    array = as_float_array(value, name=name, ndim=2, copy=True)
    array.setflags(write=False)
    return array


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


__all__ = ["PoincareSection", "poincare_section"]

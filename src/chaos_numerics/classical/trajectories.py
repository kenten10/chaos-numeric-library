"""Trajectory construction for built-in and user-defined classical maps."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from chaos_numerics.core import ClassicalMap, ExperimentMetadata, Trajectory, ValidationError
from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.types import ArrayLike, FloatArray


def iterate(
    model: ClassicalMap,
    initial_state: ArrayLike,
    *,
    steps: int,
    include_initial: bool = True,
) -> Trajectory:
    """Apply ``model`` repeatedly in ``O(steps * batch_size)`` time.

    The returned time axis is inserted immediately before the trailing state
    axis: ``(time, state_dim)`` for one state and ``(*batch, time, state_dim)``
    for a batch. Periodic initial coordinates are wrapped into their documented
    half-open bounds before being stored.
    """
    count = _nonnegative_int(steps, name="steps")
    if not isinstance(include_initial, bool):
        raise ValidationError(f"include_initial must be a bool; got {include_initial!r}")
    if count == 0 and not include_initial:
        raise ValidationError("steps must be positive when include_initial is False")

    dimension = _model_dimension(model)
    current = as_float_array(
        initial_state,
        name="initial_state",
        trailing_dim=dimension,
        copy=True,
    )
    current = _normalize_initial(model, current)
    normalized_initial = current.copy()
    stored_count = count + int(include_initial)
    states = np.empty((*current.shape[:-1], stored_count, dimension), dtype=np.float64)

    output_index = 0
    if include_initial:
        states[..., output_index, :] = current
        output_index += 1
    for _ in range(count):
        next_state = as_float_array(
            model.step(current),
            name="model step result",
            ndim=current.ndim,
            trailing_dim=dimension,
        )
        if next_state.shape != current.shape:
            raise ValidationError(
                f"model step must preserve state shape {current.shape}; got {next_state.shape}"
            )
        current = next_state
        states[..., output_index, :] = current
        output_index += 1

    parameters: dict[str, object] = {
        "model": type(model).__name__,
        "steps": count,
        "include_initial": include_initial,
    }
    model_parameters = getattr(model, "parameters", None)
    if isinstance(model_parameters, Mapping):
        parameters["model_parameters"] = dict(model_parameters)
    metadata = ExperimentMetadata(parameters=parameters, precision="float64")
    return Trajectory(states=states, initial_state=normalized_initial, metadata=metadata)


def _normalize_initial(model: ClassicalMap, state: FloatArray) -> FloatArray:
    bounds = model.bounds
    periodic = model.is_periodic
    if len(bounds) != model.state_dim or len(periodic) != model.state_dim:
        raise ValidationError("model bounds and is_periodic must match state_dim")
    result = state.copy()
    for coordinate, ((lower, upper), is_periodic) in enumerate(zip(bounds, periodic, strict=True)):
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValidationError(f"model bounds[{coordinate}] must be finite and increasing")
        values = result[..., coordinate]
        if is_periodic:
            result[..., coordinate] = lower + np.mod(values - lower, upper - lower)
        elif bool(np.any((values < lower) | (values >= upper))):
            raise ValidationError(
                f"initial_state coordinate {coordinate} must lie in [{lower}, {upper})"
            )
    return result


def _model_dimension(model: ClassicalMap) -> int:
    dimension = model.state_dim
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise ValidationError(f"model state_dim must be a positive integer; got {dimension!r}")
    result = int(dimension)
    if result <= 0:
        raise ValidationError(f"model state_dim must be positive; got {result}")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValidationError(f"{name} must be non-negative; got {result}")
    return result


__all__ = ["iterate"]

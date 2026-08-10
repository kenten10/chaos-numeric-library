"""Trajectory construction for built-in and user-defined classical maps."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from chaos_numerics.core import ClassicalMap, ExperimentMetadata, Trajectory, ValidationError
from chaos_numerics.core._validation import (
    as_float_array,
    validate_protocol,
    wrap_into_half_open,
)
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

    Notes
    -----
    The initial state is fully validated, but inside the loop only the dtype and
    shape of each step result are checked; finiteness is verified once, on the
    whole stored array, after the loop. Nothing escapes that check -- every state
    a step produced is stored, so a ``nan`` or ``inf`` returned by a user-defined
    map at any step is still detected even if a later step would have swallowed it
    (``1 / inf``). What is given up is the step number: the error reports that a
    non-finite state appeared somewhere between the initial state and ``steps``
    steps later, not where. Per-step validation cost about 6 µs of fixed overhead
    per step, which dominated short-state trajectories; call ``model.step``
    directly if a per-step diagnosis is worth that price.
    """
    # The protocol check comes first so that a foreign object is named as a
    # protocol violation here rather than as an AttributeError from the
    # ``model.state_dim`` read further down.
    validate_protocol(model, ClassicalMap, name="model")
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
    expected_shape = current.shape
    for _ in range(count):
        current = _step(model, current, expected_shape=expected_shape, dimension=dimension)
        states[..., output_index, :] = current
        output_index += 1
    _validate_trajectory_finite(states, steps=count)

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


def _step(
    model: ClassicalMap,
    current: FloatArray,
    *,
    expected_shape: tuple[int, ...],
    dimension: int,
) -> FloatArray:
    """Advance one step, checking only what the storing assignment cannot.

    A well-behaved map returns a ``float64`` array of the incoming shape and takes
    the fast path with no validation call at all. Anything else falls through to
    :func:`~chaos_numerics.core._validation.as_float_array` so that a list, an
    integer array, or a reshaped result still gets a named error rather than a
    broadcasting failure -- but with ``finite=False``, because finiteness is
    checked once on the whole trajectory instead of once per step.
    """
    result = model.step(current)
    if type(result) is np.ndarray and result.dtype == np.float64 and result.shape == expected_shape:
        return result
    coerced = as_float_array(
        result,
        name="model step result",
        ndim=len(expected_shape),
        trailing_dim=dimension,
        finite=False,
    )
    if coerced.shape != expected_shape:
        raise ValidationError(
            f"model step must preserve state shape {expected_shape}; got {coerced.shape}"
        )
    return coerced


def _validate_trajectory_finite(states: FloatArray, *, steps: int) -> None:
    if bool(np.all(np.isfinite(states))):
        return
    raise ValidationError(
        f"model step produced a non-finite state within {steps} steps of the initial state: "
        "the stored trajectory contains nan or inf. The step at which it first appeared is "
        "not reported, because the trajectory is validated once at the end rather than once "
        "per step; call model.step directly on the initial state to locate it"
    )


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
            result[..., coordinate] = wrap_into_half_open(values, lower=lower, upper=upper)
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

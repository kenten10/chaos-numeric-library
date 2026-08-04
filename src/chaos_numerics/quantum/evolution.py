"""Common dense and matrix-free quantum evolution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import numpy as np

from chaos_numerics.core import NumericalError, QuantumMap, ValidationError
from chaos_numerics.core._validation import as_complex_array
from chaos_numerics.core.types import ArrayLike, ComplexArray
from chaos_numerics.quantum.states import quantum_state
from chaos_numerics.quantum.unitary import _dense_representation


def evolve(
    model: QuantumMap,
    initial_state: ArrayLike,
    *,
    steps: int,
    method: Literal["auto", "dense", "matrix_free", "fft"] = "auto",
    return_history: bool = False,
    include_initial: bool = True,
    norm_tolerance: float = 5e-11,
    dense_limit: int = 256,
) -> ComplexArray:
    """Evolve scalar or batched normalized states through one common API."""
    count = _nonnegative_int(steps, name="steps")
    if method not in {"auto", "dense", "matrix_free", "fft"}:
        raise ValidationError(f"invalid evolution method {method!r}")
    if not isinstance(return_history, bool) or not isinstance(include_initial, bool):
        raise ValidationError("return_history and include_initial must be bool values")
    tolerance = _nonnegative_float(norm_tolerance, name="norm_tolerance")
    state = quantum_state(initial_state, dimension=model.dimension, normalized=True)
    initial_norms = np.linalg.norm(state, axis=-1)
    dense = _dense_representation(model, dense_limit=dense_limit) if method == "dense" else None
    action = model.apply
    if method == "fft":
        fft_action = getattr(model, "apply_fft", None)
        if not callable(fft_action):
            raise ValidationError(
                f"invalid evolution method 'fft' for {type(model).__name__}: "
                "model does not provide apply_fft"
            )
        action = cast(Callable[[ArrayLike], ComplexArray], fft_action)
    history: list[ComplexArray] = []
    if return_history and include_initial:
        history.append(state.copy())
    for _ in range(count):
        candidate = state @ dense.T if dense is not None else action(state)
        next_state = as_complex_array(
            candidate,
            name="quantum map result",
            ndim=state.ndim,
            trailing_dim=model.dimension,
            copy=True,
        )
        if next_state.shape != state.shape:
            raise ValidationError(
                f"quantum map must preserve state shape {state.shape}; got {next_state.shape}"
            )
        drift = np.abs(np.linalg.norm(next_state, axis=-1) - initial_norms)
        if bool(np.any(drift > tolerance)):
            raise NumericalError(
                f"quantum-state norm drift {float(np.max(drift))} exceeds {tolerance}"
            )
        state = next_state
        if return_history:
            history.append(state.copy())
    if not return_history:
        return state
    if history:
        return np.stack(history, axis=-2)
    return np.empty((*state.shape[:-1], 0, state.shape[-1]), dtype=np.complex128)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValidationError(f"{name} must be non-negative; got {result}")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite non-negative number; got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValidationError(f"{name} must be a finite non-negative number; got {value!r}")
    return result


__all__ = ["evolve"]

"""Common dense and matrix-free quantum evolution.

``EvolutionMethod`` is exported because it appears in the signature of
:func:`evolve`, and a library that ships ``py.typed`` has to let callers annotate
their own wrappers with the same type. The accepted values are:

``"auto"``
    Use the model's own ``apply``, which is the matrix-free path for every model
    that has one. This is the default and the right answer almost always.
``"dense"``
    Materialize an ``O(N**2)`` reference matrix and multiply by it, subject to
    ``dense_limit``. Faster per step at small ``N``, but pays an ``O(N**3)``
    assembly first.
``"matrix_free"``
    Accepted for symmetry with ``"dense"``; it selects the same route as
    ``"auto"``.
``"fft"``
    Require the model's ``apply_fft``. Raises :class:`ValidationError` for a model
    that does not provide one, which is the point: it is an assertion that the
    ``O(N log N)`` algorithm really is what ran.

:class:`QuantumEvolution` is the container :func:`evolve` returns. It exists so
that the return type does not change shape with an argument and so that the
reproducibility metadata -- which route actually ran, and what the norm audit
saw -- survives the call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast, get_args

import numpy as np

from chaos_numerics.core import ExperimentMetadata, NumericalError, QuantumMap, ValidationError
from chaos_numerics.core._payload import (
    ArrayPayload,
    add_convergence_history,
    metadata_payload,
)
from chaos_numerics.core._validation import as_complex_array, validate_protocol
from chaos_numerics.core.types import ArrayLike, ComplexArray
from chaos_numerics.quantum.states import quantum_state
from chaos_numerics.quantum.unitary import DEFAULT_DENSE_LIMIT, _dense_representation

EvolutionMethod: TypeAlias = Literal["auto", "dense", "matrix_free", "fft"]
_EVOLUTION_METHODS: tuple[str, ...] = get_args(EvolutionMethod)
_NORM_CHECK_INTERVAL = 16
"""Number of steps between norm audits; see :func:`evolve` for the contract."""


def _complex_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(precision="complex128")


@dataclass(frozen=True, slots=True, eq=False)
class QuantumEvolution:
    """The final state of an evolution, optionally its history, and its metadata.

    :func:`evolve` used to return a bare array whose *shape* depended on
    ``return_history``: ``(*batch, N)`` for one flag value and
    ``(*batch, T, N)`` for the other. A caller writing generic code then had to
    branch on an argument to know what it was holding, and none of the
    reproducibility metadata -- which route ran, whether the norm audit saw any
    drift -- survived the call at all. This container fixes both:
    :attr:`final_state` is always present and always ``(*batch, N)``, and
    :attr:`history` is ``None`` unless it was asked for.

    Same contract as every other result container in the library: read-only
    owned arrays, value-based ``__eq__``, a ``__reduce__`` that rebuilds through
    ``__init__`` so a ``pickle`` or ``deepcopy`` round trip keeps the arrays
    read-only, and the :class:`~chaos_numerics.core.SerializableResult` split
    into an array payload and a JSON descriptor.

    ``metadata.parameters`` carries what the caller can no longer reconstruct
    from the arrays: ``method`` (the route that actually ran, with ``"auto"``
    already resolved) alongside ``requested_method``, plus ``steps``,
    ``dimension``, ``include_initial``, ``return_history``, ``norm_tolerance``,
    ``dense_limit``, and the audit result as ``norm_audits`` (how many steps
    were checked) and ``maximum_norm_drift`` (the worst drift any of them saw).
    A run that reports ``norm_audits = 0`` was never checked, which is only
    possible at ``steps = 0``.

    Arrays are copied on construction, as everywhere else in the library, so
    building one from a buffer you still hold costs a second copy of it.
    """

    final_state: ComplexArray
    history: ComplexArray | None = None
    metadata: ExperimentMetadata = field(default_factory=_complex_metadata)

    def __post_init__(self) -> None:
        final_state = as_complex_array(self.final_state, name="final_state", copy=True)
        if final_state.ndim < 1 or final_state.shape[-1] == 0:
            raise ValidationError(
                f"final_state must have a non-empty trailing state axis; got {final_state.shape}"
            )
        history = None
        if self.history is not None:
            history = as_complex_array(self.history, name="history", copy=True)
            expected = (*final_state.shape[:-1], history.shape[-2], final_state.shape[-1])
            if history.ndim != final_state.ndim + 1 or history.shape != expected:
                raise ValidationError(
                    "history must have shape (*batch, time, dimension) matching final_state "
                    f"{final_state.shape}; got {history.shape}"
                )
            history.setflags(write=False)
        if not isinstance(self.metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        final_state.setflags(write=False)
        object.__setattr__(self, "final_state", final_state)
        object.__setattr__(self, "history", history)

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension of the evolved states."""
        return int(self.final_state.shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch axes, empty for a single state."""
        return tuple(int(size) for size in self.final_state.shape[:-1])

    @property
    def time_count(self) -> int:
        """Number of stored times, or ``0`` when no history was requested."""
        return 0 if self.history is None else int(self.history.shape[-2])

    def __reduce__(self) -> tuple[type[QuantumEvolution], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only."""
        return (self.__class__, (self.final_state, self.history, self.metadata))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuantumEvolution):
            return NotImplemented
        mine, theirs = self.history, other.history
        if (mine is None) != (theirs is None):
            return False
        histories_equal = mine is None or theirs is None or bool(np.array_equal(mine, theirs))
        return (
            bool(np.array_equal(self.final_state, other.final_state))
            and histories_equal
            and self.metadata == other.metadata
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable copies of every stored array."""
        payload: ArrayPayload = {"final_state": self.final_state.copy()}
        if self.history is not None:
            payload["history"] = self.history.copy()
        add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return the JSON descriptor, with array shapes but no array contents.

        The key set here must be exactly the key set of :meth:`array_payload`,
        including the optional ``history`` and the convergence history: the
        storage layer looks each descriptor up in the archive, so a descriptor
        that names an array the payload omits fails with a ``KeyError`` rather
        than a :class:`~chaos_numerics.core.ValidationError`.
        """
        arrays: ArrayPayload = {"final_state": self.final_state}
        if self.history is not None:
            arrays["history"] = self.history
        add_convergence_history(arrays, self.metadata, copy=False)
        return metadata_payload("quantum_evolution", self.metadata, arrays)


def evolve(
    model: QuantumMap,
    initial_state: ArrayLike,
    *,
    steps: int,
    method: EvolutionMethod = "auto",
    return_history: bool = False,
    include_initial: bool = True,
    norm_tolerance: float = 5e-11,
    dense_limit: int = DEFAULT_DENSE_LIMIT,
) -> QuantumEvolution:
    """Evolve scalar or batched normalized states through one common API.

    Returns a :class:`QuantumEvolution` whose ``final_state`` is always
    ``(*batch, N)`` and whose ``history`` is ``(*batch, T, N)`` when
    ``return_history=True`` and ``None`` otherwise. The return **type** does not
    depend on the arguments, only the presence of the optional history does::

        result = evolve(model, state, steps=100, method="fft")
        result.final_state  # always there
        result.history  # None here
        result.metadata.parameters  # method actually run, norm audit result

    ``method="auto"`` uses the model's own ``apply``, which is the matrix-free
    path for every model that has one. ``method="dense"`` materializes an
    ``O(N**2)`` reference matrix instead; it is measurably faster per step for
    small ``N`` but pays an ``O(N**3)`` assembly first, so it is an explicit
    opt-in rather than the automatic choice. The route that actually ran is
    recorded as ``metadata.parameters["method"]`` with ``"auto"`` already
    resolved, so a result read back from disk says which algorithm produced it.

    Norm auditing: the norm drift ``|| ||psi_t|| - ||psi_0|| ||`` must satisfy
    ``norm_tolerance`` at every audited step, and exceeding it raises
    :class:`NumericalError`. The first step, every
    ``_NORM_CHECK_INTERVAL``-th step, and the final step are audited; the
    tolerance is never relaxed at an audited step, so a drifting or
    non-unitary map is still rejected, just possibly a few steps later than
    the first excursion. The audit is reported rather than merely enforced:
    ``metadata.parameters`` carries ``norm_audits`` and the worst
    ``maximum_norm_drift`` any audit saw, which is what turns "it did not
    raise" into a number a reader can compare against ``norm_tolerance``.

    ``dense_limit`` applies to ``method="dense"`` only and defaults to
    :data:`~chaos_numerics.quantum.DEFAULT_DENSE_LIMIT`. The limit is a
    deliberate guard, not a capability boundary: raise it only knowing that a
    dense reference costs ``O(N**2)`` memory and ``O(N**3)`` time. Measured
    ``complex128`` assembly plus diagonalization cost is about 0.5 s / 4 MiB at
    ``N = 512``, 2.2 s / 17 MiB at ``N = 1024``, and 12 s / 67 MiB at
    ``N = 2048``.
    """
    validate_protocol(model, QuantumMap, name="model")
    count = _nonnegative_int(steps, name="steps")
    if method not in _EVOLUTION_METHODS:
        candidates = ", ".join(repr(name) for name in _EVOLUTION_METHODS)
        raise ValidationError(f"invalid evolution method {method!r}; expected one of {candidates}")
    if not isinstance(return_history, bool) or not isinstance(include_initial, bool):
        raise ValidationError("return_history and include_initial must be bool values")
    tolerance = _nonnegative_float(norm_tolerance, name="norm_tolerance")
    state = quantum_state(initial_state, dimension=model.dimension, normalized=True)
    initial_norms = np.linalg.norm(state, axis=-1)
    dense = _dense_representation(model, dense_limit=dense_limit) if method == "dense" else None
    resolved = "matrix_free" if method in ("auto", "matrix_free") else method
    action = model.apply
    if method == "fft":
        fft_action = getattr(model, "apply_fft", None)
        if not callable(fft_action):
            raise ValidationError(
                f"invalid evolution method 'fft' for {type(model).__name__}: "
                "model does not provide apply_fft"
            )
        action = cast(Callable[[ArrayLike], ComplexArray], fft_action)

    stored = count + (1 if include_initial else 0) if return_history else 0
    history: ComplexArray | None = None
    if return_history:
        history = np.empty((*state.shape[:-1], stored, model.dimension), dtype=np.complex128)
        if include_initial:
            history[..., 0, :] = state
    offset = 1 if include_initial else 0

    expected_shape = (*initial_norms.shape, model.dimension)
    audits = 0
    worst_drift = 0.0
    for index in range(count):
        candidate = state @ dense.T if dense is not None else action(state)
        if index == 0:
            # Validate dtype, shape, and finiteness once. A map that satisfies
            # the contract on its first output keeps satisfying it, and the
            # audited norm checks below reject later blow-ups, so repeating the
            # full scan and a defensive copy on every step is pure overhead.
            state = as_complex_array(
                candidate,
                name="quantum map result",
                ndim=state.ndim,
                trailing_dim=model.dimension,
                copy=False,
            )
        else:
            state = np.asarray(candidate, dtype=np.complex128)
        if state.shape != expected_shape:
            raise ValidationError(
                f"quantum map must preserve state shape {expected_shape}; got {state.shape}"
            )
        if index in (0, count - 1) or (index + 1) % _NORM_CHECK_INTERVAL == 0:
            drift = np.abs(np.linalg.norm(state, axis=-1) - initial_norms)
            # Written as a negated ``<=`` so that NaN norms are rejected too.
            if bool(np.any(~(drift <= tolerance))):
                raise NumericalError(
                    f"quantum-state norm drift {float(np.max(drift))} exceeds {tolerance}"
                )
            audits += 1
            worst_drift = max(worst_drift, float(np.max(drift)))
        if history is not None:
            history[..., index + offset, :] = state

    metadata = ExperimentMetadata(
        parameters={
            "dimension": int(model.dimension),
            "steps": count,
            "method": resolved,
            "requested_method": method,
            "return_history": return_history,
            "include_initial": include_initial,
            "batch_shape": list(state.shape[:-1]),
            "norm_tolerance": tolerance,
            "norm_check_interval": _NORM_CHECK_INTERVAL,
            "norm_audits": audits,
            "maximum_norm_drift": worst_drift,
            "dense_limit": int(dense_limit),
            "model": _model_parameters(model),
        },
        precision="complex128",
    )
    return QuantumEvolution(state, history, metadata)


def _model_parameters(model: QuantumMap) -> dict[str, object]:
    """Return the model's JSON parameters, or just its type when it has none."""
    parameters = getattr(model, "parameters", None)
    if isinstance(parameters, dict):
        return {"model": type(model).__name__, **parameters}
    return {"model": type(model).__name__, "dimension": int(model.dimension)}


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


__all__ = ["EvolutionMethod", "QuantumEvolution", "evolve"]

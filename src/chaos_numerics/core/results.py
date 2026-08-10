"""Immutable numerical result containers with owned read-only arrays.

Constructors copy input arrays into canonical dtypes. Public arrays are read-only;
``array_payload`` returns writable copies for persistence or downstream mutation.
Metadata payloads contain only JSON-compatible values and array descriptors, never
the arrays themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chaos_numerics.core._payload import (
    SCHEMA_VERSION,
    AnyArray,
    ArrayPayload,
)
from chaos_numerics.core._payload import (
    add_convergence_history as _add_convergence_history,
)
from chaos_numerics.core._payload import (
    metadata_payload as _metadata_payload,
)
from chaos_numerics.core._validation import as_complex_array, as_float_array
from chaos_numerics.core.exceptions import ValidationError
from chaos_numerics.core.metadata import ExperimentMetadata
from chaos_numerics.core.types import ComplexArray, FloatArray


def _float_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(precision="float64")


def _complex_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(precision="complex128")


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Trajectory:
    """Single or batched classical states with trailing coordinate axis."""

    states: FloatArray
    initial_state: FloatArray
    metadata: ExperimentMetadata = field(default_factory=_float_metadata)

    def __post_init__(self) -> None:
        states = _readonly_float(self.states, name="trajectory states")
        if states.ndim < 2:
            raise ValidationError(
                "trajectory states must have shape (time, state_dim) or "
                f"(*batch, time, state_dim); got {states.shape}"
            )
        if states.shape[-2] == 0 or states.shape[-1] == 0:
            raise ValidationError(
                f"trajectory states cannot have empty time/state axes; got {states.shape}"
            )

        expected_initial_shape = (*states.shape[:-2], states.shape[-1])
        initial = _readonly_float(self.initial_state, name="initial_state")
        if initial.shape != expected_initial_shape:
            raise ValidationError(
                f"initial_state must have shape {expected_initial_shape}; got {initial.shape}"
            )
        _validate_metadata(self.metadata)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "initial_state", initial)

    @property
    def shape(self) -> tuple[int, ...]:
        """Trajectory array shape."""
        return self.states.shape

    @property
    def state_dim(self) -> int:
        """Number of state coordinates."""
        return int(self.states.shape[-1])

    @property
    def time_count(self) -> int:
        """Number of stored times."""
        return int(self.states.shape[-2])

    @property
    def is_batched(self) -> bool:
        """Whether one or more batch axes precede time and state."""
        return self.states.ndim > 2

    def __reduce__(self) -> tuple[type[Trajectory], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only."""
        return (self.__class__, (self.states, self.initial_state, self.metadata))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trajectory):
            return NotImplemented
        return (
            np.array_equal(self.states, other.states)
            and np.array_equal(self.initial_state, other.initial_state)
            and self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        return _compact_repr(
            "Trajectory",
            array=self.states,
            details=f"batched={self.is_batched}",
            metadata=self.metadata,
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable arrays ready for NPZ-like persistence."""
        payload: ArrayPayload = {
            "states": self.states.copy(),
            "initial_state": self.initial_state.copy(),
        }
        _add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return JSON metadata and array descriptors without array contents."""
        arrays: ArrayPayload = {"states": self.states, "initial_state": self.initial_state}
        _add_convergence_history(arrays, self.metadata, copy=False)
        return _metadata_payload("trajectory", self.metadata, arrays)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Spectrum:
    """Eigenvalues with optional column eigenvectors and normalized residuals."""

    eigenvalues: ComplexArray
    eigenvectors: ComplexArray | None = None
    residuals: FloatArray | None = None
    metadata: ExperimentMetadata = field(default_factory=_complex_metadata)

    def __post_init__(self) -> None:
        eigenvalues = _readonly_complex(self.eigenvalues, name="eigenvalues", ndim=1)
        eigenvectors = None
        if self.eigenvectors is not None:
            eigenvectors = _readonly_complex(self.eigenvectors, name="eigenvectors", ndim=2)
            if eigenvectors.shape[1] != eigenvalues.size:
                raise ValidationError(
                    "eigenvectors must store eigenvectors in columns with shape "
                    f"(dimension, {eigenvalues.size}); got {eigenvectors.shape}"
                )

        residuals = _optional_residuals(self.residuals, count=eigenvalues.size)
        _validate_metadata(self.metadata)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "residuals", residuals)

    @property
    def count(self) -> int:
        """Number of stored eigenvalues."""
        return self.eigenvalues.size

    def __reduce__(self) -> tuple[type[Spectrum], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only."""
        return (
            self.__class__,
            (self.eigenvalues, self.eigenvectors, self.residuals, self.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Spectrum):
            return NotImplemented
        return (
            np.array_equal(self.eigenvalues, other.eigenvalues)
            and _optional_array_equal(self.eigenvectors, other.eigenvectors)
            and _optional_array_equal(self.residuals, other.residuals)
            and self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        dimension = self.eigenvectors.shape[0] if self.eigenvectors is not None else None
        return _compact_repr(
            "Spectrum",
            array=self.eigenvalues,
            details=f"count={self.count}, dimension={dimension}",
            metadata=self.metadata,
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable arrays ready for persistence."""
        payload: ArrayPayload = {"eigenvalues": self.eigenvalues.copy()}
        if self.eigenvectors is not None:
            payload["eigenvectors"] = self.eigenvectors.copy()
        if self.residuals is not None:
            payload["residuals"] = self.residuals.copy()
        _add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return JSON metadata and array descriptors without array contents."""
        arrays: ArrayPayload = {"eigenvalues": self.eigenvalues}
        if self.eigenvectors is not None:
            arrays["eigenvectors"] = self.eigenvectors
        if self.residuals is not None:
            arrays["residuals"] = self.residuals
        _add_convergence_history(arrays, self.metadata, copy=False)
        return _metadata_payload("spectrum", self.metadata, arrays)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class EigenstateResult:
    """Circular eigenphases, column eigenstates, and per-state residuals."""

    eigenphases: FloatArray
    eigenstates: ComplexArray
    residuals: FloatArray
    metadata: ExperimentMetadata = field(default_factory=_complex_metadata)

    def __post_init__(self) -> None:
        eigenphases = _readonly_float(self.eigenphases, name="eigenphases", ndim=1)
        eigenstates = _readonly_complex(self.eigenstates, name="eigenstates", ndim=2)
        if eigenstates.shape[1] != eigenphases.size:
            raise ValidationError(
                "eigenstates must store states in columns with shape "
                f"(dimension, {eigenphases.size}); got {eigenstates.shape}"
            )
        residuals = _optional_residuals(self.residuals, count=eigenphases.size)
        assert residuals is not None
        _validate_metadata(self.metadata)
        object.__setattr__(self, "eigenphases", eigenphases)
        object.__setattr__(self, "eigenstates", eigenstates)
        object.__setattr__(self, "residuals", residuals)

    @property
    def count(self) -> int:
        """Number of stored eigenstates."""
        return self.eigenphases.size

    def __reduce__(self) -> tuple[type[EigenstateResult], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only."""
        return (
            self.__class__,
            (self.eigenphases, self.eigenstates, self.residuals, self.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EigenstateResult):
            return NotImplemented
        return (
            np.array_equal(self.eigenphases, other.eigenphases)
            and np.array_equal(self.eigenstates, other.eigenstates)
            and np.array_equal(self.residuals, other.residuals)
            and self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        return _compact_repr(
            "EigenstateResult",
            array=self.eigenphases,
            details=f"count={self.count}, dimension={self.eigenstates.shape[0]}",
            metadata=self.metadata,
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable arrays ready for persistence."""
        payload: ArrayPayload = {
            "eigenphases": self.eigenphases.copy(),
            "eigenstates": self.eigenstates.copy(),
            "residuals": self.residuals.copy(),
        }
        _add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return JSON metadata and array descriptors without array contents."""
        arrays: ArrayPayload = {
            "eigenphases": self.eigenphases,
            "eigenstates": self.eigenstates,
            "residuals": self.residuals,
        }
        _add_convergence_history(arrays, self.metadata, copy=False)
        return _metadata_payload("eigenstate", self.metadata, arrays)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class AnalysisResult:
    """Named real analysis values with optional uncertainty and residual arrays."""

    name: str
    values: FloatArray
    uncertainty: FloatArray | None = None
    residuals: FloatArray | None = None
    metadata: ExperimentMetadata = field(default_factory=_float_metadata)

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValidationError("analysis name must be a non-empty trimmed string")
        values = _readonly_float(self.values, name="analysis values")
        uncertainty = _optional_same_shape(
            self.uncertainty,
            values=values,
            name="uncertainty",
            nonnegative=True,
        )
        residuals = _optional_same_shape(
            self.residuals,
            values=values,
            name="residuals",
            nonnegative=True,
        )
        _validate_metadata(self.metadata)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "residuals", residuals)

    def __reduce__(self) -> tuple[type[AnalysisResult], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only."""
        return (
            self.__class__,
            (self.name, self.values, self.uncertainty, self.residuals, self.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnalysisResult):
            return NotImplemented
        return (
            self.name == other.name
            and np.array_equal(self.values, other.values)
            and _optional_array_equal(self.uncertainty, other.uncertainty)
            and _optional_array_equal(self.residuals, other.residuals)
            and self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        return _compact_repr(
            "AnalysisResult",
            array=self.values,
            details=f"name={self.name!r}",
            metadata=self.metadata,
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable arrays ready for persistence."""
        payload: ArrayPayload = {"values": self.values.copy()}
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty.copy()
        if self.residuals is not None:
            payload["residuals"] = self.residuals.copy()
        _add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return JSON metadata and array descriptors without array contents."""
        arrays: ArrayPayload = {"values": self.values}
        if self.uncertainty is not None:
            arrays["uncertainty"] = self.uncertainty
        if self.residuals is not None:
            arrays["residuals"] = self.residuals
        _add_convergence_history(arrays, self.metadata, copy=False)
        return _metadata_payload("analysis", self.metadata, arrays, name=self.name)


def _readonly_float(value: FloatArray, *, name: str, ndim: int | None = None) -> FloatArray:
    array = as_float_array(value, name=name, ndim=ndim, copy=True)
    array.setflags(write=False)
    return array


def _readonly_complex(value: ComplexArray, *, name: str, ndim: int) -> ComplexArray:
    array = as_complex_array(value, name=name, ndim=ndim, copy=True)
    array.setflags(write=False)
    return array


def _optional_residuals(value: FloatArray | None, *, count: int) -> FloatArray | None:
    if value is None:
        return None
    residuals = _readonly_float(value, name="residuals", ndim=1)
    if residuals.shape != (count,):
        raise ValidationError(f"residuals must have shape ({count},); got {residuals.shape}")
    if np.any(residuals < 0.0):
        raise ValidationError("residuals must be non-negative")
    return residuals


def _optional_same_shape(
    value: FloatArray | None,
    *,
    values: FloatArray,
    name: str,
    nonnegative: bool,
) -> FloatArray | None:
    if value is None:
        return None
    array = _readonly_float(value, name=name)
    if array.shape != values.shape:
        raise ValidationError(f"{name} must have shape {values.shape}; got {array.shape}")
    if nonnegative and np.any(array < 0.0):
        raise ValidationError(f"{name} must be non-negative")
    return array


def _optional_array_equal(left: AnyArray | None, right: AnyArray | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(left, right))


def _validate_metadata(metadata: ExperimentMetadata) -> None:
    if not isinstance(metadata, ExperimentMetadata):
        raise ValidationError("metadata must be an ExperimentMetadata instance")


def _compact_repr(
    class_name: str,
    *,
    array: AnyArray,
    details: str,
    metadata: ExperimentMetadata,
) -> str:
    converged = metadata.convergence.converged if metadata.convergence is not None else None
    return (
        f"{class_name}(shape={array.shape}, dtype={array.dtype.name!r}, {details}, "
        f"warnings={len(metadata.warnings)}, converged={converged})"
    )


# ``SCHEMA_VERSION``, ``AnyArray``, and ``ArrayPayload`` are defined in
# ``core._payload`` and re-exported here because this module is where the four
# result containers live and where the rest of the library already imports them
# from. Listing them keeps that path explicit under ``implicit_reexport = false``.
__all__ = [
    "SCHEMA_VERSION",
    "AnalysisResult",
    "AnyArray",
    "ArrayPayload",
    "EigenstateResult",
    "Spectrum",
    "Trajectory",
]

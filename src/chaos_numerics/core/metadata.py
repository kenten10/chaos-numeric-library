"""Immutable metadata and diagnostics shared by numerical results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np

from chaos_numerics.core._validation import (
    as_float_array,
    validate_finite_float,
    validate_nonnegative_int,
    validate_trimmed_string,
)
from chaos_numerics.core.exceptions import ValidationError
from chaos_numerics.core.types import FloatArray

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | MappingProxyType[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A stable warning code and its human-readable explanation."""

    code: str
    message: str
    category: str = "numerical"

    def __post_init__(self) -> None:
        for field_name in ("code", "message", "category"):
            validate_trimmed_string(getattr(self, field_name), name=f"diagnostic {field_name}")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation."""
        return {"code": self.code, "message": self.message, "category": self.category}


@dataclass(frozen=True, slots=True, eq=False)
class ConvergenceInfo:
    """Termination status and optional finite convergence history."""

    converged: bool
    iterations: int
    residual: float | None = None
    tolerance: float | None = None
    history: FloatArray | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "iterations",
            validate_nonnegative_int(self.iterations, name="convergence iterations"),
        )

        for field_name in ("residual", "tolerance"):
            value = getattr(self, field_name)
            if value is None:
                continue
            object.__setattr__(
                self,
                field_name,
                validate_finite_float(value, name=f"convergence {field_name}", minimum=0.0),
            )

        if self.reason is not None:
            validate_trimmed_string(self.reason, name="convergence reason")
        if self.history is not None:
            object.__setattr__(
                self,
                "history",
                _readonly_float(self.history, name="convergence history"),
            )

    def __reduce__(
        self,
    ) -> tuple[type[ConvergenceInfo], tuple[object, ...]]:
        """Round-trip through ``__init__`` so the history stays read-only.

        NumPy drops the ``writeable=False`` flag when an array is pickled, so
        rebuilding through the constructor is what keeps the documented
        immutability after ``pickle`` or ``copy.deepcopy``.
        """
        return (
            self.__class__,
            (
                self.converged,
                self.iterations,
                self.residual,
                self.tolerance,
                self.history,
                self.reason,
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConvergenceInfo):
            return NotImplemented
        return (
            self.converged == other.converged
            and self.iterations == other.iterations
            and self.residual == other.residual
            and self.tolerance == other.tolerance
            and _optional_array_equal(self.history, other.history)
            and self.reason == other.reason
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON metadata, leaving history in the array payload."""
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "reason": self.reason,
            "history_array_key": "convergence_history" if self.history is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    """Reproducibility, precision, diagnostics, and convergence for a result."""

    parameters: Mapping[str, object] = field(default_factory=dict)
    precision: str = "float64"
    seed: int | np.integer[Any] | None = None
    environment: Mapping[str, object] = field(default_factory=dict)
    git_commit: str | None = None
    warnings: tuple[Diagnostic, ...] = ()
    convergence: ConvergenceInfo | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters, path="parameters"))
        object.__setattr__(
            self, "environment", _freeze_mapping(self.environment, path="environment")
        )
        if self.precision not in {"float64", "complex128", "mixed"}:
            raise ValidationError(
                "precision must be one of 'float64', 'complex128', or 'mixed'; "
                f"got {self.precision!r}"
            )
        if self.seed is not None:
            if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
                self.seed, (int, np.integer)
            ):
                raise ValidationError(f"seed must be an integer or None; got {self.seed!r}")
            object.__setattr__(self, "seed", int(self.seed))
        if self.git_commit is not None:
            validate_trimmed_string(self.git_commit, name="git_commit")

        warnings = tuple(self.warnings)
        if not all(isinstance(item, Diagnostic) for item in warnings):
            raise ValidationError("warnings must contain only Diagnostic instances")
        object.__setattr__(self, "warnings", warnings)
        if self.convergence is not None and not isinstance(self.convergence, ConvergenceInfo):
            raise ValidationError("convergence must be ConvergenceInfo or None")

    def __reduce__(
        self,
    ) -> tuple[type[ExperimentMetadata], tuple[object, ...]]:
        """Round-trip through ``__init__`` because ``mappingproxy`` cannot pickle.

        ``parameters`` and ``environment`` are stored as :class:`MappingProxyType`
        for immutability, and that type has no ``__reduce__``. Writing plain
        dictionaries into the pickle state and re-freezing them in
        ``__post_init__`` makes results usable with ``pickle``,
        ``copy.deepcopy``, ``multiprocessing``, and ``joblib``.
        """
        return (
            self.__class__,
            (
                _thaw_mapping(self.parameters),
                self.precision,
                self.seed,
                _thaw_mapping(self.environment),
                self.git_commit,
                self.warnings,
                self.convergence,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible dictionary without numerical arrays."""
        return {
            "parameters": _thaw_mapping(self.parameters),
            "precision": self.precision,
            "seed": self.seed,
            "environment": _thaw_mapping(self.environment),
            "git_commit": self.git_commit,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "convergence": self.convergence.to_dict() if self.convergence is not None else None,
        }


def _freeze_mapping(value: object, *, path: str) -> MappingProxyType[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be a mapping with string keys")

    frozen: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValidationError(f"{path} keys must be strings; got {key!r}")
        frozen[key] = _freeze_json(item, path=f"{path}.{key}")
    return MappingProxyType(frozen)


def _freeze_json(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValidationError(f"{path} must be finite for JSON serialization")
        return numeric
    if isinstance(value, Mapping):
        return _freeze_mapping(value, path=path)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise ValidationError(
        f"{path} must be JSON-compatible metadata, not {type(value).__name__}; "
        "store numerical arrays in the array payload"
    )


def _thaw_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {key: _thaw_json(item) for key, item in value.items()}


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _thaw_mapping(value)
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _readonly_float(value: FloatArray, *, name: str) -> FloatArray:
    array = as_float_array(value, name=name, copy=True)
    array.setflags(write=False)
    return array


def _optional_array_equal(left: FloatArray | None, right: FloatArray | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(left, right))


__all__ = ["ConvergenceInfo", "Diagnostic", "ExperimentMetadata"]

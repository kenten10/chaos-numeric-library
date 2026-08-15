"""Declarative experiment configuration and deterministic parameter grids."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from chaos_numerics.core import ExperimentMetadata, ValidationError
from chaos_numerics.core._validation import validate_trimmed_string


def _validated_mapping(value: Mapping[str, object], *, name: str) -> MappingProxyType[str, object]:
    try:
        validated = ExperimentMetadata(parameters=value).parameters
    except ValidationError as error:
        raise ValidationError(f"invalid {name}: {error}") from error
    return MappingProxyType({key: _thaw(item) for key, item in validated.items()})


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Experiment:
    """A JSON-compatible declaration of one model and analysis."""

    model: str
    analysis: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    seed: int | None = None
    strict_reproducibility: bool = False

    def __post_init__(self) -> None:
        for name in ("model", "analysis"):
            validate_trimmed_string(getattr(self, name), name=name)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValidationError(f"seed must be an integer or None; got {self.seed!r}")
        if not isinstance(self.strict_reproducibility, bool):
            raise ValidationError("strict_reproducibility must be a bool")
        object.__setattr__(
            self,
            "parameters",
            _validated_mapping(self.parameters, name="experiment parameters"),
        )

    def __reduce__(self) -> tuple[type[Experiment], tuple[object, ...]]:
        """Rebuild through ``__init__`` because ``mappingproxy`` cannot pickle."""
        return (
            self.__class__,
            (
                self.model,
                self.analysis,
                {key: _thaw(value) for key, value in self.parameters.items()},
                self.seed,
                self.strict_reproducibility,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "model": self.model,
            "analysis": self.analysis,
            "parameters": {key: _thaw(value) for key, value in self.parameters.items()},
            "seed": self.seed,
            "strict_reproducibility": self.strict_reproducibility,
        }


def cartesian_grid(**parameters: Iterable[object]) -> tuple[dict[str, object], ...]:
    """Return the deterministic Cartesian product of named parameter values."""
    names = tuple(parameters)
    values = tuple(tuple(items) for items in parameters.values())
    if any(not items for items in values):
        return ()
    rows = tuple(
        dict(zip(names, combination, strict=True)) for combination in itertools.product(*values)
    )
    return _validate_grid(rows)


def zip_grid(**parameters: Iterable[object]) -> tuple[dict[str, object], ...]:
    """Zip equally sized named parameter sequences into deterministic jobs."""
    names = tuple(parameters)
    values = tuple(tuple(items) for items in parameters.values())
    lengths = {len(items) for items in values}
    if len(lengths) > 1:
        raise ValidationError("zip_grid parameter sequences must have equal lengths")
    return _validate_grid(
        tuple(
            dict(zip(names, combination, strict=True)) for combination in zip(*values, strict=True)
        )
    )


def _validate_grid(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(dict(_validated_mapping(row, name="grid parameters")) for row in rows)


__all__ = ["Experiment", "cartesian_grid", "zip_grid"]

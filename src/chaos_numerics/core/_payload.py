"""Shared persistence payload helpers for every domain result container.

Splitting arrays from JSON is a whole-library convention, not a `core` detail: the
storage layer writes numerical arrays to NPZ and everything else to JSON, and it
can only do that if each result exposes the split the same way. Keeping the two
builders here means a domain container implements
:class:`~chaos_numerics.core.SerializableResult` by delegating rather than by
copying the descriptor format, which is how the format drifted before.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np

from chaos_numerics.core.metadata import ExperimentMetadata

AnyArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
"""A NumPy array of any shape and any dtype.

The persistence split carries ``float64``, ``complex128``, and index arrays side
by side in one mapping, so the element type cannot be narrowed at this boundary.
Individual result attributes stay precisely typed as
:data:`~chaos_numerics.core.FloatArray` or
:data:`~chaos_numerics.core.ComplexArray`; only the payload is heterogeneous."""

ArrayPayload: TypeAlias = dict[str, AnyArray]
"""The return type of ``array_payload()`` on every result container.

Maps an array name to an independent writable array, ready to be written to NPZ.
It is part of the public surface because ``py.typed`` is shipped: a caller who
annotates a function that consumes or produces one of these payloads needs a name
for the type that does not come from a private module."""

# Bumped only when the on-disk layout changes, never for a new result type.
SCHEMA_VERSION = 1


def add_convergence_history(
    payload: ArrayPayload,
    metadata: ExperimentMetadata,
    *,
    copy: bool = True,
) -> None:
    """Add the convergence history to an array payload when one was recorded."""
    if metadata.convergence is not None and metadata.convergence.history is not None:
        history = metadata.convergence.history
        payload["convergence_history"] = history.copy() if copy else history


def metadata_payload(
    result_type: str,
    metadata: ExperimentMetadata,
    arrays: Mapping[str, AnyArray],
    **extra: Any,
) -> dict[str, object]:
    """Return the JSON-compatible descriptor for one result.

    Array contents never appear here; each entry is reduced to its shape and dtype
    so that the descriptor stays small and the numbers live in the archive.
    """
    descriptors = {
        name: {"shape": list(array.shape), "dtype": array.dtype.name}
        for name, array in arrays.items()
    }
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "result_type": result_type,
        "metadata": metadata.to_dict(),
        "arrays": descriptors,
    }
    payload.update(extra)
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "AnyArray",
    "ArrayPayload",
    "add_convergence_history",
    "metadata_payload",
]

"""Runtime validation kept separate from static protocol definitions."""

from collections.abc import Sequence
from typing import Any, Generic, Protocol, TypeAlias, cast

import numpy as np

from chaos_numerics.core.exceptions import ValidationError
from chaos_numerics.core.types import ArrayLike, ComplexArray, FloatArray, OperatorArray

IntegerLike: TypeAlias = int | np.integer[Any]


def as_float_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
    trailing_dim: int | None = None,
    finite: bool = True,
    copy: bool = False,
) -> FloatArray:
    """Return a validated ``float64`` array without guessing its axis order."""
    raw = _as_array(value, name=name)
    if raw.dtype.kind not in "iuf":
        raise ValidationError(f"{name} must contain real numeric values; got dtype {raw.dtype}")

    array = cast(FloatArray, raw.astype(np.float64, copy=copy))
    _validate_array_shape(array, name=name, ndim=ndim, trailing_dim=trailing_dim)
    _validate_finite(array, name=name, finite=finite)
    return array


def as_complex_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
    trailing_dim: int | None = None,
    finite: bool = True,
    copy: bool = False,
) -> ComplexArray:
    """Return a validated ``complex128`` array."""
    raw = _as_array(value, name=name)
    if raw.dtype.kind not in "iufc":
        raise ValidationError(f"{name} must contain numeric values; got dtype {raw.dtype}")

    array = cast(ComplexArray, raw.astype(np.complex128, copy=copy))
    _validate_array_shape(array, name=name, ndim=ndim, trailing_dim=trailing_dim)
    _validate_finite(array, name=name, finite=finite)
    return array


def as_operator_array(
    value: ArrayLike,
    *,
    name: str = "operator",
    dimension: int | None = None,
    finite: bool = True,
    copy: bool = False,
) -> OperatorArray:
    """Return a square ``complex128`` dense operator with an optional dimension."""
    array = as_complex_array(
        value,
        name=name,
        ndim=2,
        finite=finite,
        copy=copy,
    )
    if array.shape[0] != array.shape[1]:
        raise ValidationError(f"{name} must be square; got shape {array.shape}")
    if dimension is not None:
        expected = validate_positive_int(dimension, name="dimension")
        if array.shape != (expected, expected):
            raise ValidationError(
                f"{name} must have shape ({expected}, {expected}); got {array.shape}"
            )
    return array


def validate_bounds(
    bounds: ArrayLike,
    *,
    state_dim: int,
    name: str = "bounds",
) -> FloatArray:
    """Validate bounds in coordinate order with shape ``(state_dim, 2)``."""
    dimension = validate_positive_int(state_dim, name="state_dim")
    array = as_float_array(bounds, name=name, ndim=2, trailing_dim=2)
    if array.shape != (dimension, 2):
        raise ValidationError(f"{name} must have shape ({dimension}, 2); got {array.shape}")
    invalid = np.flatnonzero(array[:, 0] >= array[:, 1])
    if invalid.size:
        coordinates = ", ".join(str(int(index)) for index in invalid)
        raise ValidationError(
            f"{name} lower bounds must be less than upper bounds; invalid coordinates: "
            f"{coordinates}"
        )
    return array


def validate_shape(
    shape: Sequence[IntegerLike],
    *,
    name: str = "shape",
    ndim: int | None = None,
) -> tuple[int, ...]:
    """Validate a non-empty tuple of positive integer extents."""
    if not shape:
        raise ValidationError(f"{name} must contain at least one dimension")
    result = tuple(
        validate_positive_int(extent, name=f"{name}[{index}]") for index, extent in enumerate(shape)
    )
    if ndim is not None and len(result) != ndim:
        raise ValidationError(f"{name} must contain {ndim} dimensions; got {len(result)}")
    return result


def validate_positive_int(value: IntegerLike, *, name: str) -> int:
    """Validate a positive built-in or NumPy integer, excluding booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a positive integer; got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValidationError(f"{name} must be positive; got {result}")
    return result


def validate_protocol(value: object, protocol: type, *, name: str) -> None:
    """Reject objects that do not structurally implement a runtime protocol.

    Every public entry point uses this so that a missing attribute surfaces as
    :class:`ValidationError` naming the protocol, never as a bare
    ``AttributeError`` raised deep inside an algorithm.
    """
    if not isinstance(value, protocol):
        missing = sorted(
            member for member in _protocol_members(protocol) if not hasattr(value, member)
        )
        detail = f"; missing {', '.join(missing)}" if missing else ""
        raise ValidationError(
            f"{name} must implement the {protocol.__name__} protocol; "
            f"got {type(value).__name__}{detail}"
        )


def _protocol_members(protocol: type) -> set[str]:
    """Return the member names a protocol requires.

    CPython exposes this as ``__protocol_attrs__``, but only from 3.12 onwards, and
    it is private in any case. Deriving the names from the class body keeps the
    error message identical on every supported interpreter instead of silently
    dropping the useful half of it on 3.11.

    The derivation keeps only names without a leading underscore. Every name a
    protocol class body carries that is *not* one of its members is either a
    dunder or an implementation flag such as ``_is_protocol``, and each new
    CPython adds more of them -- 3.12 added ``__non_callable_proto_members__``,
    3.13 added ``__firstlineno__`` and ``__static_attributes__``, 3.14 added
    ``__annotate_func__`` and ``__annotations_cache__``. An allowlist by shape
    does not have to grow with them, and no protocol in this library requires a
    private or dunder member.
    """
    cached = getattr(protocol, "__protocol_attrs__", None)
    if isinstance(cached, (set, frozenset)):
        return set(cached)
    members: set[str] = set()
    for base in protocol.__mro__:
        if base in (object, Protocol, Generic):
            continue
        names = (*getattr(base, "__annotations__", {}), *vars(base))
        members.update(name for name in names if not name.startswith("_"))
    return members


def wrap_into_half_open(
    values: FloatArray,
    *,
    lower: float,
    upper: float,
) -> FloatArray:
    """Reduce values into ``[lower, upper)`` without the ``np.mod`` endpoint leak.

    ``np.mod(-1e-17, 1.0)`` rounds to exactly ``1.0``, so the plain modulo can
    return the excluded upper endpoint for inputs a rounding error below
    ``lower``. Every periodic coordinate in the library is reduced through this
    helper so that the documented half-open bounds hold exactly.
    """
    width = upper - lower
    reduced = lower + np.mod(values - lower, width)
    return cast(FloatArray, np.where(reduced >= upper, lower, reduced))


def _as_array(value: ArrayLike, *, name: str) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"{name} could not be converted to a regular numeric array"
        ) from error


def _validate_array_shape(
    array: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    *,
    name: str,
    ndim: int | None,
    trailing_dim: int | None,
) -> None:
    if array.ndim == 0:
        raise ValidationError(f"{name} must have at least one dimension; got scalar input")
    if ndim is not None and array.ndim != ndim:
        raise ValidationError(f"{name} must have {ndim} dimensions; got shape {array.shape}")
    if trailing_dim is not None:
        expected = validate_positive_int(trailing_dim, name="trailing_dim")
        if array.shape[-1] != expected:
            raise ValidationError(
                f"{name} must have trailing dimension {expected}; got shape {array.shape}"
            )


def _validate_finite(
    array: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    *,
    name: str,
    finite: bool,
) -> None:
    if finite and not bool(np.all(np.isfinite(array))):
        raise ValidationError(f"{name} must contain only finite values")

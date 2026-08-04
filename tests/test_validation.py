"""Tests for runtime validation independent from static protocols."""

import numpy as np
import pytest

from chaos_numerics.core import ValidationError
from chaos_numerics.core._validation import (
    as_complex_array,
    as_float_array,
    as_operator_array,
    validate_bounds,
    validate_positive_int,
    validate_shape,
)


def test_float_array_promotes_numeric_input_and_preserves_batch_axes() -> None:
    states = as_float_array([[1, 2], [3, 4]], name="states", trailing_dim=2)

    assert states.dtype == np.float64
    assert states.shape == (2, 2)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([True, False], "real numeric values"),
        ([1 + 2j, 3 + 4j], "real numeric values"),
        (["one", "two"], "real numeric values"),
        (1.0, "at least one dimension"),
        ([1.0, np.inf], "finite values"),
    ],
)
def test_float_array_rejects_invalid_dtype_rank_and_values(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        as_float_array(value, name="state")  # type: ignore[arg-type]


def test_float_array_rejects_wrong_trailing_dimension() -> None:
    with pytest.raises(ValidationError, match=r"trailing dimension 2.*\(3,\)"):
        as_float_array([1.0, 2.0, 3.0], name="state", trailing_dim=2)


def test_complex_array_accepts_real_input_and_uses_complex128() -> None:
    state = as_complex_array([1.0, 0.0], name="state", trailing_dim=2)

    assert state.dtype == np.complex128
    np.testing.assert_array_equal(state, np.array([1.0, 0.0], dtype=np.complex128))


def test_operator_validation_rejects_nonsquare_and_wrong_dimension() -> None:
    with pytest.raises(ValidationError, match=r"must be square.*\(2, 3\)"):
        as_operator_array(np.ones((2, 3)), name="floquet")

    with pytest.raises(ValidationError, match=r"shape \(3, 3\).*"):
        as_operator_array(np.eye(2), name="floquet", dimension=3)


def test_bounds_validation_reports_invalid_coordinate() -> None:
    with pytest.raises(ValidationError, match="invalid coordinates: 1"):
        validate_bounds([[0.0, 1.0], [2.0, 2.0]], state_dim=2)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_positive_integer_validation_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValidationError, match=r"positive integer|must be positive"):
        validate_positive_int(value, name="dimension")  # type: ignore[arg-type]


def test_shape_validation_accepts_numpy_integers_and_rejects_wrong_rank() -> None:
    assert validate_shape((np.int64(2), 3), ndim=2) == (2, 3)

    with pytest.raises(ValidationError, match="must contain 2 dimensions"):
        validate_shape((2, 3, 4), ndim=2)

"""Tests for runtime validation independent from static protocols."""

from collections.abc import Callable

import numpy as np
import pytest

from chaos_numerics.core import (
    ClassicalMap,
    ConvergenceInfo,
    ExperimentMetadata,
    Flow,
    LinearOperatorLike,
    Partition,
    QuantumMap,
    SerializableResult,
    ValidationError,
)
from chaos_numerics.core._validation import (
    _protocol_members,
    as_complex_array,
    as_float_array,
    as_operator_array,
    validate_bounds,
    validate_positive_int,
    validate_shape,
)


def floats(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    return np.asarray(value, dtype=np.float64)


def complexes(value: object) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    return np.asarray(value, dtype=np.complex128)


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

    with pytest.raises(ValidationError, match="at least one dimension"):
        validate_shape(())


def test_operator_validation_accepts_a_square_operator_of_the_declared_dimension() -> None:
    operator = as_operator_array([[0, 1], [1, 0]], name="floquet", dimension=2)

    assert operator.dtype == np.complex128
    assert operator.shape == (2, 2)


def test_bounds_validation_rejects_the_wrong_number_of_coordinates() -> None:
    with pytest.raises(ValidationError, match=r"bounds must have shape \(2, 2\); got \(1, 2\)"):
        validate_bounds([[0.0, 1.0]], state_dim=2)


def test_ragged_input_is_a_validation_error_not_a_numpy_error() -> None:
    """``np.asarray`` raises on a ragged nested sequence; the boundary translates it."""
    with pytest.raises(ValidationError, match="could not be converted to a regular numeric array"):
        as_float_array([[1.0, 2.0], [3.0]], name="states")


# Spelled out rather than derived, so that the 3.11 path is checked against the
# protocol definitions and not against the code under test.
EXPECTED_PROTOCOL_MEMBERS: dict[str, set[str]] = {
    "ClassicalMap": {"state_dim", "is_periodic", "bounds", "step", "jacobian"},
    "Flow": {"state_dim", "vector_field", "jacobian"},
    "LinearOperatorLike": {"shape", "dtype", "matvec"},
    "Partition": {
        "ndim",
        "shape",
        "size",
        "cell_bounds",
        "locate",
        "ravel_index",
        "unravel_index",
        "sample",
    },
    "QuantumMap": {"dimension", "apply", "as_linear_operator"},
    "SerializableResult": {"metadata", "array_payload", "metadata_payload"},
}


@pytest.mark.parametrize(
    "protocol",
    [ClassicalMap, Flow, LinearOperatorLike, Partition, QuantumMap, SerializableResult],
)
def test_protocol_members_are_derived_identically_without_cpythons_cache(
    protocol: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python 3.11 path must name the same members as ``__protocol_attrs__``.

    ``__protocol_attrs__`` does not exist on 3.11, so ``validate_protocol`` derives
    the member names from the class body instead. That fallback is dead code on
    every newer interpreter, which is exactly how it could rot into producing a
    different -- or empty -- ``missing ...`` list on the one interpreter that
    still uses it. Replacing the cache with a non-set forces the derivation.
    """
    derived = _protocol_members(protocol)
    assert derived, f"no members derived for {protocol.__name__}"
    assert not any(name.startswith("_") for name in derived), derived

    cached = getattr(protocol, "__protocol_attrs__", None)
    if cached is None:
        # Python 3.11 has no cache, which is the interpreter the fallback exists
        # for. There is nothing to cross-check against, so compare with the members
        # spelled out in the protocol body instead.
        assert derived == EXPECTED_PROTOCOL_MEMBERS[protocol.__name__]
        return

    expected = set(cached)
    assert derived == expected
    monkeypatch.setattr(protocol, "__protocol_attrs__", None)
    assert _protocol_members(protocol) == expected


def test_public_constructors_raise_validation_error_rather_than_leaking() -> None:
    """A wrong *type* must fail the same way a wrong *value* does.

    ``ValidationError`` subclasses ``ValueError`` precisely so that a caller can
    write ``except ValueError``. Six constructors used the idiom
    ``if not value or value.strip() != value``, which short-circuits only for falsy
    values, so any truthy non-string reached ``.strip()`` and left with an
    ``AttributeError`` -- caught by neither ``except ValueError`` nor
    ``except ChaosNumericsError``. Two more leaked an ``AssertionError`` and a bare
    ``KeyError``. Each is checked here through ``except ValueError`` as well, because
    that is the contract the exception hierarchy advertises.
    """
    from chaos_numerics.core import AnalysisResult, Diagnostic, EigenstateResult
    from chaos_numerics.experiment import Experiment

    cases: list[tuple[str, Callable[[], object]]] = [
        ("AnalysisResult name", lambda: AnalysisResult(5, np.ones(2))),  # type: ignore[arg-type]
        ("Experiment model", lambda: Experiment(5, "trajectory")),  # type: ignore[arg-type]
        ("Experiment analysis", lambda: Experiment("standard_map", 5)),  # type: ignore[arg-type]
        ("Diagnostic code", lambda: Diagnostic(5, "message")),  # type: ignore[arg-type]
        ("ConvergenceInfo reason", lambda: ConvergenceInfo(True, 1, 0.0, 1e-9, None, 5)),  # type: ignore[arg-type]
        ("metadata git_commit", lambda: ExperimentMetadata(git_commit=5)),  # type: ignore[arg-type]
        (
            "EigenstateResult residuals",
            # The declared type says required; the runtime used to accept None anyway.
            lambda: EigenstateResult(floats([0.1, 0.2]), complexes(np.eye(2)), None),  # type: ignore[arg-type]
        ),
        ("ConvergenceInfo iterations float", lambda: ConvergenceInfo(True, 2.5)),  # type: ignore[arg-type]
        ("ConvergenceInfo iterations str", lambda: ConvergenceInfo(True, "3")),  # type: ignore[arg-type]
        ("ConvergenceInfo residual str", lambda: ConvergenceInfo(True, 1, "abc")),  # type: ignore[arg-type]
    ]
    for label, call in cases:
        with pytest.raises(ValidationError):
            call()
        # The advertised catch pattern has to work too.
        with pytest.raises(ValueError):
            call()
        assert label

    # A float that happens to be integral is still rejected: silently storing
    # ``iterations=2.5`` as ``2`` was a wrong answer where an error belongs.
    with pytest.raises(ValidationError, match="non-negative integer"):
        ConvergenceInfo(True, 3.0)  # type: ignore[arg-type]
    assert ConvergenceInfo(True, 3).iterations == 3


def test_missing_required_analysis_parameter_names_itself() -> None:
    """``values.pop("steps")`` used to raise a ``KeyError`` whose message was ``'steps'``.

    Persisted by a sweep that becomes ``{"type": "KeyError", "message": "'steps'"}``
    in the manifest, which says nothing about which analysis wanted it.
    """
    from chaos_numerics.experiment import Experiment, run_experiment

    for analysis in ("trajectory", "lyapunov_spectrum", "largest_lyapunov_exponent"):
        with pytest.raises(ValidationError, match=r"requires the parameter 'steps'"):
            run_experiment(Experiment(model="standard_map", analysis=analysis, seed=1))

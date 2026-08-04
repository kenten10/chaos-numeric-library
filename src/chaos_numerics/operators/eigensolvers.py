"""Leading eigensystems and invariant densities for Ulam operators."""

from __future__ import annotations

import warnings as python_warnings
from typing import Any, Literal

import numpy as np
from scipy.sparse import issparse  # type: ignore[import-untyped]
from scipy.sparse.linalg import (  # type: ignore[import-untyped]
    ArpackNoConvergence,
    eigs,
)
from scipy.sparse.linalg import (
    norm as sparse_norm,
)

from chaos_numerics.core import (
    AnalysisResult,
    ConvergenceError,
    ConvergenceInfo,
    ConvergenceWarning,
    Diagnostic,
    ExperimentMetadata,
    NumericalError,
    Spectrum,
    ValidationError,
)


def leading_eigenpairs(
    operator: Any,
    *,
    count: int = 6,
    side: Literal["right", "left"] = "right",
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
    strict: bool = False,
) -> Spectrum:
    """Return eigenpairs ordered by decreasing eigenvalue modulus.

    Right vectors satisfy ``P @ v = lambda * v``. Left vectors are stored as
    columns and satisfy ``P.H @ w = conj(lambda) * w``.
    """
    dimension = _operator_dimension(operator)
    pair_count = _positive_int(count, name="count")
    if pair_count > dimension:
        raise ValidationError(f"count must not exceed operator dimension {dimension}")
    if side not in {"right", "left"}:
        raise ValidationError(f"side must be 'right' or 'left'; got {side!r}")
    solver_tolerance = _positive_float(tolerance, name="tolerance")
    if max_iterations is not None:
        max_iterations = _positive_int(max_iterations, name="max_iterations")
    if not isinstance(strict, bool):
        raise ValidationError(f"strict must be a bool; got {strict!r}")

    solved_operator = operator.T.conjugate() if side == "left" else operator
    method = "dense" if pair_count >= dimension - 1 else "arpack"
    partial = False
    if method == "dense":
        dense = _dense_array(solved_operator)
        values, vectors = np.linalg.eig(dense)
    else:
        initial = np.linspace(1.0, 2.0, dimension, dtype=np.float64)
        initial /= np.linalg.norm(initial)
        try:
            values, vectors = eigs(
                solved_operator,
                k=pair_count,
                which="LM",
                tol=solver_tolerance,
                maxiter=max_iterations,
                v0=initial,
            )
        except ArpackNoConvergence as error:
            values = error.eigenvalues
            vectors = error.eigenvectors
            partial = True
            if values is None or vectors is None or len(values) == 0:
                raise ConvergenceError("ARPACK returned no converged eigenpairs") from error

    values = np.asarray(values, dtype=np.complex128)
    vectors = np.asarray(vectors, dtype=np.complex128)
    if side == "left":
        values = np.conjugate(values)
    order = _eigenvalue_order(values)[:pair_count]
    values = values[order]
    vectors = _normalize_columns(vectors[:, order])
    residuals = _eigenpair_residuals(operator, values, vectors, side=side)
    residual_limit = 1e-12 if method == "dense" else max(solver_tolerance, 1e-9)
    converged = (
        not partial and values.size == pair_count and bool(np.all(residuals <= residual_limit))
    )
    reason = (
        "all independently evaluated residuals satisfy tolerance"
        if converged
        else "partial eigenpairs or independently evaluated residuals exceed tolerance"
    )
    diagnostics: tuple[Diagnostic, ...] = ()
    if not converged:
        message = f"leading eigenpairs did not fully converge: {reason}"
        if strict:
            raise ConvergenceError(message)
        python_warnings.warn(message, ConvergenceWarning, stacklevel=2)
        diagnostics = (Diagnostic("eigenpairs-not-converged", message, "convergence"),)

    convergence = ConvergenceInfo(
        converged=converged,
        iterations=0,
        residual=float(np.max(residuals)),
        tolerance=residual_limit,
        history=residuals,
        reason=reason,
    )
    metadata = ExperimentMetadata(
        parameters={
            "side": side,
            "ordering": "descending absolute eigenvalue",
            "normalization": "unit L2 norm; largest component positive real",
            "method": method,
            "requested_count": pair_count,
            "returned_count": int(values.size),
            "requested_tolerance": solver_tolerance,
            "max_iterations": max_iterations,
            "iteration_count_available": False,
        },
        precision="complex128",
        warnings=diagnostics,
        convergence=convergence,
    )
    return Spectrum(values, vectors, residuals, metadata)


def stationary_density(
    operator: Any,
    *,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
    strict: bool = True,
) -> AnalysisResult:
    """Return the normalized right eigenvector whose eigenvalue is nearest one."""
    dimension = _operator_dimension(operator)
    spectrum = leading_eigenpairs(
        operator,
        count=min(4, dimension),
        side="right",
        tolerance=tolerance,
        max_iterations=max_iterations,
        strict=strict,
    )
    index = int(np.argmin(np.abs(spectrum.eigenvalues - 1.0)))
    eigenvalue = spectrum.eigenvalues[index]
    assert spectrum.eigenvectors is not None
    vector = spectrum.eigenvectors[:, index]
    total = np.sum(vector)
    if abs(total) <= np.finfo(np.float64).tiny:
        raise NumericalError("stationary eigenvector has zero total mass")
    normalized = vector / total
    imaginary_norm = float(np.linalg.norm(np.imag(normalized)))
    density = np.real(normalized).astype(np.float64)
    if imaginary_norm > 1e-12:
        raise NumericalError(
            f"stationary density has non-negligible imaginary norm {imaginary_norm}"
        )
    if float(np.min(density)) < -1e-12:
        raise NumericalError(f"stationary density has negative entry {float(np.min(density))}")
    density /= np.sum(density)
    invariance = float(np.linalg.norm(_matvec(operator, density) - density, ord=1))
    eigenvalue_error = float(abs(eigenvalue - 1.0))
    limit = max(_positive_float(tolerance, name="tolerance"), 1e-10)
    eigenpairs_converged = (
        spectrum.metadata.convergence is not None and spectrum.metadata.convergence.converged
    )
    converged = eigenpairs_converged and invariance <= limit and eigenvalue_error <= limit
    if not converged and strict:
        raise ConvergenceError(
            f"stationary density failed invariance/eigenvalue tolerance: "
            f"invariance={invariance}, eigenvalue_error={eigenvalue_error}"
        )
    diagnostics = spectrum.metadata.warnings
    if not converged and not strict:
        message = "stationary density did not satisfy invariance or eigenvalue tolerance"
        python_warnings.warn(message, ConvergenceWarning, stacklevel=2)
        diagnostics = (*diagnostics, Diagnostic("stationary-density-not-converged", message))
    convergence = ConvergenceInfo(
        converged=converged,
        iterations=0,
        residual=max(invariance, eigenvalue_error),
        tolerance=limit,
        reason="invariance and eigenvalue-one checks" if converged else "checks exceeded tolerance",
    )
    metadata = ExperimentMetadata(
        parameters={
            "eigenvalue_real": float(np.real(eigenvalue)),
            "eigenvalue_imag": float(np.imag(eigenvalue)),
            "normalization": "real density with sum one",
            "invariance_l1": invariance,
        },
        precision="float64",
        warnings=diagnostics,
        convergence=convergence,
    )
    return AnalysisResult("stationary_density", density, metadata=metadata)


def spectral_gap(
    operator_or_spectrum: Any,
    *,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
    strict: bool = True,
) -> AnalysisResult:
    """Return the Markov spectral gap ``1 - abs(lambda_2)``."""
    spectrum = (
        operator_or_spectrum
        if isinstance(operator_or_spectrum, Spectrum)
        else leading_eigenpairs(
            operator_or_spectrum,
            count=2,
            tolerance=tolerance,
            max_iterations=max_iterations,
            strict=strict,
        )
    )
    if spectrum.count < 2:
        raise ValidationError("spectral_gap requires at least two eigenvalues")
    stationary_index = int(np.argmin(np.abs(spectrum.eigenvalues - 1.0)))
    leading = spectrum.eigenvalues[stationary_index]
    remaining = np.delete(spectrum.eigenvalues, stationary_index)
    subleading = remaining[int(np.argmax(np.abs(remaining)))]
    leading_error = float(abs(leading - 1.0))
    limit = max(_positive_float(tolerance, name="tolerance"), 1e-10)
    if leading_error > limit and strict:
        raise ConvergenceError(f"leading eigenvalue differs from one by {leading_error}")
    gap = float(1.0 - abs(subleading))
    if gap < -limit and strict:
        raise ConvergenceError(f"subleading eigenvalue modulus exceeds one by {-gap}")
    metadata = ExperimentMetadata(
        parameters={
            "definition": "1 - abs(lambda_2)",
            "lambda_1_real": float(np.real(leading)),
            "lambda_1_imag": float(np.imag(leading)),
            "lambda_2_real": float(np.real(subleading)),
            "lambda_2_imag": float(np.imag(subleading)),
        },
        precision="float64",
        warnings=spectrum.metadata.warnings,
        convergence=spectrum.metadata.convergence,
    )
    return AnalysisResult("spectral_gap", np.asarray([gap], dtype=np.float64), metadata=metadata)


def _operator_dimension(operator: Any) -> int:
    shape = getattr(operator, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] != shape[1]:
        raise ValidationError(f"operator must be square; got shape {shape}")
    dimension = int(shape[0])
    if dimension < 1:
        raise ValidationError("operator dimension must be positive")
    return dimension


def _dense_array(operator: Any) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    raw = operator.toarray() if issparse(operator) else np.asarray(operator)
    dense = np.asarray(raw, dtype=np.complex128)
    if not bool(np.all(np.isfinite(dense))):
        raise ValidationError("operator must contain only finite values")
    return dense


def _normalize_columns(
    vectors: np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    result = vectors.copy()
    for column in range(result.shape[1]):
        norm = np.linalg.norm(result[:, column])
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
            raise NumericalError("eigensolver returned a zero or non-finite eigenvector")
        result[:, column] /= norm
        pivot = int(np.argmax(np.abs(result[:, column])))
        phase = np.angle(result[pivot, column])
        result[:, column] *= np.exp(-1j * phase)
    return result


def _eigenvalue_order(
    values: np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.intp]]:
    moduli = np.abs(values)
    order = np.argsort(-moduli, kind="stable")
    stationary = int(np.argmin(np.abs(values - 1.0)))
    maximum = float(np.max(moduli))
    tie_tolerance = 1e-12 * max(1.0, maximum)
    if maximum - float(moduli[stationary]) <= tie_tolerance:
        order = np.concatenate(
            (np.asarray([stationary], dtype=np.intp), order[order != stationary])
        )
    return order.astype(np.intp, copy=False)


def _eigenpair_residuals(
    operator: Any,
    values: np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
    vectors: np.ndarray[tuple[int, ...], np.dtype[np.complex128]],
    *,
    side: Literal["right", "left"],
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    solved = operator.T.conjugate() if side == "left" else operator
    norm_estimate = (
        float(sparse_norm(solved)) if issparse(solved) else float(np.linalg.norm(solved))
    )
    residuals = np.empty(values.size, dtype=np.float64)
    for index, value in enumerate(values):
        target_value = np.conjugate(value) if side == "left" else value
        vector = vectors[:, index]
        numerator = np.linalg.norm(_matvec(solved, vector) - target_value * vector)
        denominator = max(
            np.finfo(np.float64).tiny,
            (norm_estimate + abs(target_value)) * np.linalg.norm(vector),
        )
        residuals[index] = numerator / denominator
    return residuals


def _matvec(operator: Any, vector: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(operator @ vector)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a positive integer; got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValidationError(f"{name} must be positive; got {result}")
    return result


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite positive number; got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValidationError(f"{name} must be a finite positive number; got {value!r}")
    return result


__all__ = ["leading_eigenpairs", "spectral_gap", "stationary_density"]

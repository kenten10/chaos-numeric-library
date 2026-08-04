"""Circular eigenphase preparation, unfolding, and nearest-level statistics."""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from chaos_numerics.core import (
    AnalysisResult,
    Diagnostic,
    EigenstateResult,
    ExperimentMetadata,
    NumericalError,
    NumericalWarning,
    ValidationError,
)
from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.types import ArrayLike, FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class PreparedEigenphases:
    """Owned raw, wrapped/sorted, and circular-spacing eigenphase data."""

    raw_phases: FloatArray
    phases: FloatArray
    spacings: FloatArray
    symmetry_sector: str | None
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def __post_init__(self) -> None:
        raw = _readonly_1d(self.raw_phases, name="raw_phases")
        phases = _readonly_1d(self.phases, name="phases")
        spacings = _readonly_1d(self.spacings, name="spacings")
        if phases.size != raw.size:
            raise ValidationError("raw_phases and phases must contain the same number of values")
        expected_spacings = raw.size if raw.size >= 2 else 0
        if spacings.size != expected_spacings:
            raise ValidationError(
                f"spacings must contain {expected_spacings} circular gaps; got {spacings.size}"
            )
        if np.any(np.diff(phases) < 0.0) or np.any(phases < 0.0) or np.any(phases >= 2.0 * np.pi):
            raise ValidationError("phases must be sorted in the half-open interval [0, 2*pi)")
        _sector(self.symmetry_sector)
        if not isinstance(self.metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        object.__setattr__(self, "raw_phases", raw)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "spacings", spacings)

    @property
    def count(self) -> int:
        return int(self.phases.size)


@dataclass(frozen=True, slots=True, eq=False)
class UnfoldedSpectrum:
    """Eigenphases and their unfolded levels/spacings with source data retained."""

    raw_phases: FloatArray
    phases: FloatArray
    values: FloatArray
    spacings: FloatArray
    symmetry_sector: str | None
    method: str
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def __post_init__(self) -> None:
        raw = _readonly_1d(self.raw_phases, name="raw_phases")
        phases = _readonly_1d(self.phases, name="phases")
        values = _readonly_1d(self.values, name="unfolded values")
        spacings = _readonly_1d(self.spacings, name="unfolded spacings")
        if raw.size != phases.size or phases.size != values.size:
            raise ValidationError(
                "raw, processed, and unfolded phase arrays must have equal length"
            )
        expected_spacings = values.size if values.size >= 2 else 0
        if spacings.size != expected_spacings:
            raise ValidationError(
                f"unfolded spacings must contain {expected_spacings} gaps; got {spacings.size}"
            )
        if not self.method or self.method.strip() != self.method:
            raise ValidationError("method must be a non-empty trimmed string")
        _sector(self.symmetry_sector)
        if not isinstance(self.metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        object.__setattr__(self, "raw_phases", raw)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "spacings", spacings)

    @property
    def count(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True, slots=True, eq=False)
class SpacingDistributionResult:
    """Histogram values and bin edges for normalized nearest-level spacings."""

    values: FloatArray
    bin_edges: FloatArray
    sample_count: int
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def __post_init__(self) -> None:
        values = _readonly_1d(self.values, name="histogram values")
        edges = _readonly_1d(self.bin_edges, name="bin_edges")
        if edges.size != values.size + 1 or np.any(np.diff(edges) <= 0.0):
            raise ValidationError(
                "bin_edges must be strictly increasing with len(values) + 1 entries"
            )
        if np.any(values < 0.0):
            raise ValidationError("histogram values must be non-negative")
        count = _nonnegative_int(self.sample_count, name="sample_count")
        if not isinstance(self.metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "sample_count", count)


def prepare_eigenphases(
    eigenphases: ArrayLike | EigenstateResult,
    *,
    symmetry_sector: str | None = None,
    degeneracy_tolerance: float = 1e-10,
) -> PreparedEigenphases:
    """Wrap phases to ``[0, 2*pi)``, sort, and retain the endpoint gap."""
    raw_input = (
        eigenphases.eigenphases if isinstance(eigenphases, EigenstateResult) else eigenphases
    )
    raw = as_float_array(raw_input, name="eigenphases", ndim=1, copy=True)
    sector = _sector(symmetry_sector)
    tolerance = _nonnegative_float(degeneracy_tolerance, name="degeneracy_tolerance")
    phases = np.sort(np.mod(raw, 2.0 * np.pi), kind="stable")
    spacings = _circular_spacings(phases, period=2.0 * np.pi)
    degenerate_pairs = int(np.count_nonzero(spacings <= tolerance))
    diagnostics: list[Diagnostic] = []
    if sector is None:
        message = "symmetry sector is unknown; mixed sectors can invalidate level statistics"
        diagnostics.append(Diagnostic("symmetry-sector-unknown", message, category="spectral"))
        python_warnings.warn(message, NumericalWarning, stacklevel=2)
    if degenerate_pairs:
        message = f"detected {degenerate_pairs} circular eigenphase gaps at or below {tolerance}"
        diagnostics.append(Diagnostic("eigenphase-degeneracy", message, category="spectral"))
        python_warnings.warn(message, NumericalWarning, stacklevel=2)
    if raw.size < 2:
        diagnostics.append(
            Diagnostic(
                "small-spectrum",
                "at least two eigenphases are required for circular spacings",
                category="spectral",
            )
        )
    metadata = ExperimentMetadata(
        parameters={
            "phase_interval": "[0, 2*pi)",
            "ordering": "ascending",
            "endpoint_spacing": True,
            "symmetry_sector": sector,
            "degeneracy_tolerance": tolerance,
            "degenerate_pairs": degenerate_pairs,
        },
        precision="float64",
        warnings=tuple(diagnostics),
    )
    return PreparedEigenphases(raw, phases, spacings, sector, metadata)


def unfold(
    spectrum: PreparedEigenphases,
    *,
    method: Literal["mean", "polynomial"] = "mean",
    degree: int = 5,
) -> UnfoldedSpectrum:
    """Unfold a prepared circular spectrum with an explicit smooth-count method."""
    if not isinstance(spectrum, PreparedEigenphases):
        raise ValidationError("spectrum must be a PreparedEigenphases instance")
    if method not in {"mean", "polynomial"}:
        raise ValidationError(f"unfolding method must be 'mean' or 'polynomial'; got {method!r}")
    requested_degree = _positive_int(degree, name="degree")
    count = spectrum.count
    diagnostics = list(spectrum.metadata.warnings)
    fit_residual = 0.0
    effective_degree: int | None = None

    if count == 0:
        values = np.empty(0, dtype=np.float64)
        spacings = np.empty(0, dtype=np.float64)
    elif count == 1:
        values = np.zeros(1, dtype=np.float64)
        spacings = np.empty(0, dtype=np.float64)
    elif method == "mean":
        mean_spacing = 2.0 * np.pi / count
        values = (spectrum.phases - spectrum.phases[0]) / mean_spacing
        spacings = spectrum.spacings / mean_spacing
    else:
        effective_degree = min(requested_degree, count - 1)
        ranks = np.arange(count, dtype=np.float64)
        coefficients = np.polynomial.polynomial.polyfit(
            spectrum.phases,
            ranks,
            deg=effective_degree,
        )
        fitted = np.polynomial.polynomial.polyval(spectrum.phases, coefficients)
        fit_residual = float(np.sqrt(np.mean((fitted - ranks) ** 2)))
        span = float(fitted[-1] - fitted[0])
        if not np.isfinite(span) or span <= 0.0 or np.any(np.diff(fitted) <= 0.0):
            raise NumericalError("polynomial unfolding is not strictly increasing on the spectrum")
        values = (fitted - fitted[0]) * (count - 1) / span
        spacings = _circular_spacings(values, period=float(count))

    metadata = ExperimentMetadata(
        parameters={
            "method": method,
            "requested_degree": requested_degree,
            "effective_degree": effective_degree,
            "fit_residual": fit_residual,
            "mean_unfolded_spacing": float(np.mean(spacings)) if spacings.size else None,
            "symmetry_sector": spectrum.symmetry_sector,
            "source_degenerate_pairs": spectrum.metadata.parameters["degenerate_pairs"],
        },
        precision="float64",
        warnings=tuple(diagnostics),
    )
    return UnfoldedSpectrum(
        spectrum.raw_phases,
        spectrum.phases,
        values,
        spacings,
        spectrum.symmetry_sector,
        method,
        metadata,
    )


def adjacent_gap_ratios(
    spectrum: PreparedEigenphases | UnfoldedSpectrum,
    *,
    degeneracy: Literal["drop", "zero", "raise"] = "drop",
    tolerance: float = 1e-12,
) -> AnalysisResult:
    """Return circular adjacent-gap ratios ``min(s_i,s_j)/max(s_i,s_j)``."""
    spacings, sector = _spectrum_spacings(spectrum)
    if degeneracy not in {"drop", "zero", "raise"}:
        raise ValidationError(f"invalid degeneracy policy {degeneracy!r}")
    threshold = _nonnegative_float(tolerance, name="tolerance")
    diagnostics = list(spectrum.metadata.warnings)
    if spacings.size < 2:
        ratios = np.empty(0, dtype=np.float64)
        diagnostics.append(
            Diagnostic(
                "small-spectrum",
                "at least two spacings are required for adjacent-gap ratios",
                category="spectral",
            )
        )
        dropped = 0
    else:
        left = spacings
        right = np.roll(spacings, -1)
        degenerate = (left <= threshold) | (right <= threshold)
        if degeneracy == "raise" and bool(np.any(degenerate)):
            raise ValidationError("zero or degenerate gaps are present")
        if degeneracy == "drop":
            ratios = np.minimum(left[~degenerate], right[~degenerate]) / np.maximum(
                left[~degenerate], right[~degenerate]
            )
            dropped = int(np.count_nonzero(degenerate))
        else:
            denominator = np.maximum(left, right)
            ratios = np.divide(
                np.minimum(left, right),
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > threshold,
            )
            dropped = 0
    metadata = ExperimentMetadata(
        parameters={
            "circular": True,
            "degeneracy_policy": degeneracy,
            "tolerance": threshold,
            "dropped_ratios": dropped,
            "mean_ratio": float(np.mean(ratios)) if ratios.size else None,
            "symmetry_sector": sector,
        },
        precision="float64",
        warnings=tuple(diagnostics),
    )
    return AnalysisResult("adjacent_gap_ratios", ratios, metadata=metadata)


def spacing_distribution(
    spectrum: PreparedEigenphases | UnfoldedSpectrum,
    *,
    bins: int | ArrayLike = 20,
    value_range: tuple[float, float] | None = None,
    density: bool = True,
) -> SpacingDistributionResult:
    """Histogram normalized circular spacings with exact counts and bin edges."""
    spacings, sector = _spectrum_spacings(spectrum)
    normalized = _normalized_spacings(spacings)
    histogram_bins = _histogram_bins(bins)
    limits = _value_range(value_range)
    if normalized.size:
        counts, edges = np.histogram(
            normalized,
            bins=histogram_bins,
            range=limits,
            density=False,
        )
        included = int(np.sum(counts))
        if density and included:
            values = counts / (included * np.diff(edges))
        else:
            values = counts.astype(np.float64)
    else:
        edges = (
            np.linspace(limits[0], limits[1], histogram_bins + 1)
            if isinstance(histogram_bins, int)
            else histogram_bins
        )
        values = np.zeros(edges.size - 1, dtype=np.float64)
        included = 0
    metadata = ExperimentMetadata(
        parameters={
            "density": density,
            "normalization": "unit_mean_spacing",
            "included_count": included,
            "excluded_count": int(normalized.size - included),
            "symmetry_sector": sector,
        },
        precision="float64",
        warnings=spectrum.metadata.warnings,
    )
    return SpacingDistributionResult(values, edges, int(normalized.size), metadata)


def _spectrum_spacings(
    spectrum: PreparedEigenphases | UnfoldedSpectrum,
) -> tuple[FloatArray, str | None]:
    if not isinstance(spectrum, (PreparedEigenphases, UnfoldedSpectrum)):
        raise ValidationError("spectrum must be prepared or unfolded eigenphase data")
    return spectrum.spacings, spectrum.symmetry_sector


def _normalized_spacings(spacings: FloatArray) -> FloatArray:
    if spacings.size == 0:
        return spacings.copy()
    mean = float(np.mean(spacings))
    if mean <= np.finfo(np.float64).tiny:
        return np.zeros_like(spacings)
    return np.asarray(spacings / mean, dtype=np.float64)


def _circular_spacings(values: FloatArray, *, period: float) -> FloatArray:
    if values.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(np.concatenate((values, values[:1] + period)))


def _readonly_1d(value: ArrayLike, *, name: str) -> FloatArray:
    array = as_float_array(value, name=name, ndim=1, copy=True)
    array.setflags(write=False)
    return array


def _sector(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or not value or value.strip() != value):
        raise ValidationError("symmetry_sector must be a non-empty trimmed string or None")
    return value


def _histogram_bins(value: int | ArrayLike) -> int | FloatArray:
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return _positive_int(value, name="bins")
    edges = as_float_array(value, name="bins", ndim=1, copy=True)
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValidationError("bins must contain at least two strictly increasing edges")
    return edges


def _value_range(value: tuple[float, float] | None) -> tuple[float, float]:
    if value is None:
        return (0.0, 4.0)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValidationError("value_range must be a (lower, upper) tuple")
    lower = _finite_float(value[0], name="value_range lower")
    upper = _finite_float(value[1], name="value_range upper")
    if lower >= upper:
        raise ValidationError("value_range lower must be less than upper")
    return (lower, upper)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValidationError(f"{name} must be positive; got 0")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a non-negative integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValidationError(f"{name} must be non-negative; got {result}")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ValidationError(f"{name} must be non-negative; got {result}")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    return result


__all__ = [
    "PreparedEigenphases",
    "SpacingDistributionResult",
    "UnfoldedSpectrum",
    "adjacent_gap_ratios",
    "prepare_eigenphases",
    "spacing_distribution",
    "unfold",
]

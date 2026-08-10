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
from chaos_numerics.core._payload import (
    ArrayPayload,
    add_convergence_history,
    metadata_payload,
)
from chaos_numerics.core._validation import as_float_array, wrap_into_half_open
from chaos_numerics.core.types import ArrayLike, FloatArray
from chaos_numerics.spectral._ensembles import (
    CanonicalEnsemble,
    Ensemble,
    canonical_ensemble,
)

#: Large-``N`` mean adjacent gap ratio ``<min(s_i,s_j)/max(s_i,s_j)>``. Poisson is
#: the closed form ``2*ln(2)-1``; the beta=1 and beta=2 entries are the numerical
#: constants quoted to four decimals by Oganesyan & Huse, Phys. Rev. B 75, 155111
#: (2007), Table I, and reproduced by Atas, Bogomolny, Giraud & Roux, Phys. Rev.
#: Lett. 110, 084101 (2013), Table I.
#:
#: The beta=4 entry was measured here rather than quoted, because no model in this
#: library produces GSE spectra to check a quoted number against. Sampling the
#: Dumitriu-Edelman beta-Hermite tridiagonal ensemble at ``n=4096`` and taking the
#: gap ratios of the central 40% of each spectrum, the same procedure recovers
#: 0.53069 for beta=1 and 0.59960 for beta=2 -- both agreeing with the published
#: constants to the last quoted digit -- and gives ``0.6744`` for beta=4. See
#: :func:`mean_gap_ratio_reference` for the sample sizes and error bars.
_MEAN_GAP_RATIO_LARGE_N: dict[CanonicalEnsemble, float] = {
    "poisson": float(2.0 * np.log(2.0) - 1.0),
    "goe": 0.5307,
    "gue": 0.5996,
    "cue": 0.5996,
    "gse": 0.6744,
}

#: Mean gap ratio of the 3x3 surmise of Atas et al. (2013), Eq. (5) and Table I.
#: The surmise is exact for Poisson, so that entry is unchanged. The beta=4 entry
#: is ``int_0^1 r*P_4(r) dr = 0.6761683`` of the folded surmise that
#: ``rmt_reference("gap_ratio_distribution", "gse", ...)`` returns, computed by
#: quadrature here and rounded to the four decimals the other entries carry.
_MEAN_GAP_RATIO_SURMISE: dict[CanonicalEnsemble, float] = {
    "poisson": float(2.0 * np.log(2.0) - 1.0),
    "goe": 0.5359,
    "gue": 0.6027,
    "cue": 0.6027,
    "gse": 0.6762,
}


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

    def __reduce__(self) -> tuple[type[PreparedEigenphases], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only.

        NumPy drops ``writeable=False`` when an array is pickled, and
        ``copy.deepcopy`` uses the same protocol, so without this a round trip
        silently returned a mutable result.
        """
        return (
            self.__class__,
            (
                self.raw_phases,
                self.phases,
                self.spacings,
                self.symmetry_sector,
                self.metadata,
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PreparedEigenphases):
            return NotImplemented
        return (
            np.array_equal(self.raw_phases, other.raw_phases)
            and np.array_equal(self.phases, other.phases)
            and np.array_equal(self.spacings, other.spacings)
            and self.symmetry_sector == other.symmetry_sector
            and self.metadata == other.metadata
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable copies of every stored array."""
        payload: ArrayPayload = {
            "raw_phases": self.raw_phases.copy(),
            "phases": self.phases.copy(),
            "spacings": self.spacings.copy(),
        }
        add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return the JSON descriptor, with array shapes but no array contents."""
        arrays: ArrayPayload = {
            "raw_phases": self.raw_phases,
            "phases": self.phases,
            "spacings": self.spacings,
        }
        add_convergence_history(arrays, self.metadata, copy=False)
        return metadata_payload(
            "prepared_eigenphases",
            self.metadata,
            arrays,
            symmetry_sector=self.symmetry_sector,
        )


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

    def __reduce__(self) -> tuple[type[UnfoldedSpectrum], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only."""
        return (
            self.__class__,
            (
                self.raw_phases,
                self.phases,
                self.values,
                self.spacings,
                self.symmetry_sector,
                self.method,
                self.metadata,
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnfoldedSpectrum):
            return NotImplemented
        return (
            np.array_equal(self.raw_phases, other.raw_phases)
            and np.array_equal(self.phases, other.phases)
            and np.array_equal(self.values, other.values)
            and np.array_equal(self.spacings, other.spacings)
            and self.symmetry_sector == other.symmetry_sector
            and self.method == other.method
            and self.metadata == other.metadata
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable copies of every stored array."""
        payload: ArrayPayload = {
            "raw_phases": self.raw_phases.copy(),
            "phases": self.phases.copy(),
            "values": self.values.copy(),
            "spacings": self.spacings.copy(),
        }
        add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return the JSON descriptor, with array shapes but no array contents."""
        arrays: ArrayPayload = {
            "raw_phases": self.raw_phases,
            "phases": self.phases,
            "values": self.values,
            "spacings": self.spacings,
        }
        add_convergence_history(arrays, self.metadata, copy=False)
        return metadata_payload(
            "unfolded_spectrum",
            self.metadata,
            arrays,
            symmetry_sector=self.symmetry_sector,
            method=self.method,
        )


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

    def __reduce__(self) -> tuple[type[SpacingDistributionResult], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only."""
        return (
            self.__class__,
            (self.values, self.bin_edges, self.sample_count, self.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpacingDistributionResult):
            return NotImplemented
        return (
            np.array_equal(self.values, other.values)
            and np.array_equal(self.bin_edges, other.bin_edges)
            and self.sample_count == other.sample_count
            and self.metadata == other.metadata
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable copies of every stored array."""
        payload: ArrayPayload = {
            "values": self.values.copy(),
            "bin_edges": self.bin_edges.copy(),
        }
        add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return the JSON descriptor, with array shapes but no array contents."""
        arrays: ArrayPayload = {
            "values": self.values,
            "bin_edges": self.bin_edges,
        }
        add_convergence_history(arrays, self.metadata, copy=False)
        return metadata_payload(
            "spacing_distribution",
            self.metadata,
            arrays,
            sample_count=self.sample_count,
        )


def prepare_eigenphases(
    eigenphases: ArrayLike | EigenstateResult,
    *,
    symmetry_sector: str | None = None,
    degeneracy_tolerance: float = 1e-10,
) -> PreparedEigenphases:
    """Wrap phases to ``[0, 2*pi)``, sort, and retain the endpoint gap.

    Wrapping goes through :func:`wrap_into_half_open` because ``np.mod`` rounds
    inputs a rounding error below zero (``np.angle`` legitimately returns
    ``-1e-18`` for eigenvalues just under the positive real axis) up to exactly
    ``2*pi``, which is outside the documented half-open interval.
    """
    raw_input = (
        eigenphases.eigenphases if isinstance(eigenphases, EigenstateResult) else eigenphases
    )
    raw = as_float_array(raw_input, name="eigenphases", ndim=1, copy=True)
    sector = _sector(symmetry_sector)
    tolerance = _nonnegative_float(degeneracy_tolerance, name="degeneracy_tolerance")
    phases = np.sort(wrap_into_half_open(raw, lower=0.0, upper=2.0 * np.pi), kind="stable")
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
    """Unfold a prepared circular spectrum with an explicit smooth-count method.

    Two properties of the returned spacings are construction identities rather
    than diagnostics, and must not be used as accuracy checks:

    - ``mean(spacings)`` is ``1`` to rounding for both methods. The circular gaps
      sum to the period by construction (``sum = period``, ``period / count``
      levels), so the mean carries no information about unfolding quality. Use
      ``fit_residual`` (``polynomial``) or a level-statistic comparison instead.
    - For ``method="polynomial"`` the final, branch-cut-crossing spacing is ``1``
      *by construction*, up to about one unit in the last place. The fitted
      counting function is rescaled onto ``[0, count-1]``, so the wrap-around gap
      is forced to ``count - (count - 1)``. The rescaling is a floating-point
      division, so the result is exactly ``1.0`` for most inputs but can miss by
      a few times ``2**-52``: over 400 random 128-level spectra, 336 landed on
      exactly ``1.0`` and 32 were off by around ``1e-16`` (the remaining 32 hit
      the strict-monotonicity guard below). Do not assert bit equality on it --
      the last bit of ``polyfit`` depends on the BLAS in use.

      One of ``count`` spacings is therefore synthetic;
      ``metadata.parameters["circular_gap_is_synthetic"]`` records this. A
      non-periodic polynomial cannot be extrapolated across the branch cut
      reliably, so the gap is not estimated from the fit. Use
      ``method="mean"`` when the wrap-around gap itself matters.

    ``method="polynomial"`` raises :class:`NumericalError` when the fitted
    counting function is not strictly increasing on the spectrum. A degree-5 fit
    to a Poisson-like spectrum does this routinely -- about 8% of 128-level
    uniform-random spectra -- because a least-squares polynomial is free to turn
    over inside a large gap, and a non-monotone counting function would produce
    negative unfolded spacings.
    """
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
        if spacings.size:
            diagnostics.append(
                Diagnostic(
                    "synthetic-circular-gap",
                    "polynomial unfolding rescales onto [0, count-1], so the final "
                    "branch-cut spacing is 1 by construction (to about one ulp), "
                    "not measured",
                    category="spectral",
                )
            )

    metadata = ExperimentMetadata(
        parameters={
            "method": method,
            "requested_degree": requested_degree,
            "effective_degree": effective_degree,
            "fit_residual": fit_residual,
            "mean_unfolded_spacing": float(np.mean(spacings)) if spacings.size else None,
            "mean_unfolded_spacing_is_identity": True,
            "circular_gap_is_synthetic": method == "polynomial" and spacings.size > 0,
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


def mean_gap_ratio_reference(ensemble: Ensemble, *, surmise: bool = False) -> float:
    """Return the reference mean of :func:`adjacent_gap_ratios` for an ensemble.

    Two different numbers are in circulation for every Wigner-Dyson ensemble and
    they must not be mixed up:

    - ``surmise=False`` (the default) returns the **large-``N``** mean, obtained
      from long random spectra: Poisson ``2*ln(2)-1 = 0.386294...``, GOE/COE
      ``0.5307``, GUE/CUE ``0.5996``, GSE/CSE ``0.6744``. This is the value to
      compare a measured ``<r>`` against, because :func:`adjacent_gap_ratios` is
      normally applied to a spectrum of many levels, and the estimator converges
      to the large-``N`` value as the spectrum grows.
    - ``surmise=True`` returns the mean of the **3x3 surmise** of Atas et al.,
      the closed-form ratio distribution of a 3x3 matrix: Poisson unchanged, GOE
      ``0.5359``, GUE ``0.6027``, GSE ``0.6762``. Use it only when comparing
      against the surmise *distribution* itself, or against genuinely tiny
      spectra; it overshoots the large-``N`` mean by about 1% (``+0.0052`` for
      GOE, ``+0.0031`` for GUE, ``+0.0018`` for GSE), which is larger than the
      sampling error of a few hundred spectra and will look like a systematic bias
      if used as the large-``N`` reference.

    The Poisson entry is computed as ``2*np.log(2) - 1`` in both cases, because
    the ratio distribution of independent exponential gaps is exact and the
    surmise reproduces it. The beta=1 and beta=2 entries are literature numerical
    constants carried to the four decimals that the sources quote, so treat them
    as accurate to ``+/-1e-4`` and do not assert on them more tightly than that.

    **The beta=4 entries were measured here, not quoted.** No model in this library
    generates GSE statistics, so there was nothing internal to check a remembered
    constant against, and both numbers were produced from scratch:

    - The surmise entry is ``int_0^1 r * P_4(r) dr = 0.6761683`` by quadrature of
      the folded 3x3 surmise that
      :func:`~chaos_numerics.spectral.rmt_reference` returns for ``"gse"``, the
      same self-consistency identity that gives 0.5358984 and 0.6026578 for the
      published 0.5359 and 0.6027.
    - The large-``N`` entry comes from the Dumitriu-Edelman beta-Hermite
      tridiagonal ensemble, whose eigenvalue density is exactly the beta=4 one, at
      ``n=4096`` with the gap ratios of the central 40% of each spectrum. The
      identical procedure returns ``0.53069 +/- 0.00026`` for beta=1 and
      ``0.59960 +/- 0.00023`` for beta=2 on 9.8e5 ratios each, reproducing the two
      published constants to the last digit they are quoted to, which is what
      licenses the beta=4 number from the same pipeline: ``0.674408 +/- 0.000064``
      over 9.8e6 ratios, stable against ``n`` (0.67416 at ``n=1024``, 0.67424 at
      ``n=8192``) and against the window (0.67368 on the central 20%). An
      independent 40-member Haar circular-symplectic sample at ``2N=128``, run
      through this library's own ``prepare_eigenphases`` pipeline, gives
      ``0.6708 +/- 0.0057``. The table stores ``0.6744``.

    References:
        V. Oganesyan and D. A. Huse, Phys. Rev. B 75, 155111 (2007) - large-``N``
        values. Y. Y. Atas, E. Bogomolny, O. Giraud and G. Roux, Phys. Rev. Lett.
        110, 084101 (2013) - the 3x3 surmise and both sets of values in Table I.
        I. Dumitriu and A. Edelman, J. Math. Phys. 43, 5830 (2002) - the
        tridiagonal beta-ensemble used for the beta=4 measurement.

    ``"coe"`` returns the ``"goe"`` value, ``"cue"`` the ``"gue"`` value and
    ``"cse"`` the ``"gse"`` value: the ratio distribution only depends on the
    Dyson index.

    A GSE spectrum is Kramers degenerate and these values describe the *distinct*
    levels. Passing :func:`adjacent_gap_ratios` a spectrum with both members of
    every doublet gives ``<r>`` of order ``1e-14``, not 0.6744, because half the
    gaps vanish; drop one eigenvalue per doublet first.
    """
    canonical = canonical_ensemble(ensemble)
    if not isinstance(surmise, bool):
        raise ValidationError(f"surmise must be a bool; got {surmise!r}")
    table = _MEAN_GAP_RATIO_SURMISE if surmise else _MEAN_GAP_RATIO_LARGE_N
    return table[canonical]


def spacing_distribution(
    spectrum: PreparedEigenphases | UnfoldedSpectrum,
    *,
    bins: int | ArrayLike = 20,
    value_range: tuple[float, float] | None = None,
    density: bool = True,
) -> SpacingDistributionResult:
    """Histogram normalized circular spacings with exact counts and bin edges.

    With ``density=True`` the counts are divided by the *total* number of
    spacings, not by the number that landed inside ``value_range``. The returned
    values are therefore a probability density of the full sample: the integral
    equals one only when ``value_range`` covers every spacing, and is less than
    one whenever ``metadata.parameters["excluded_count"]`` is non-zero.
    Renormalizing on the in-range count instead would rescale a truncated
    histogram up to unit mass and make it disagree with unclipped reference
    densities such as the Wigner surmise.
    """
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
        if density:
            values = counts / (normalized.size * np.diff(edges))
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
    excluded = int(normalized.size - included)
    diagnostics = list(spectrum.metadata.warnings)
    if excluded:
        diagnostics.append(
            Diagnostic(
                "spacings-outside-range",
                f"{excluded} of {normalized.size} spacings fall outside "
                f"[{edges[0]}, {edges[-1]}]; a density histogram integrates to "
                f"{included / normalized.size} rather than one",
                category="spectral",
            )
        )
    metadata = ExperimentMetadata(
        parameters={
            "density": density,
            "normalization": "unit_mean_spacing",
            "density_normalization": "total_sample_count",
            "included_count": included,
            "excluded_count": excluded,
            "symmetry_sector": sector,
        },
        precision="float64",
        warnings=tuple(diagnostics),
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
    "mean_gap_ratio_reference",
    "prepare_eigenphases",
    "spacing_distribution",
    "unfold",
]

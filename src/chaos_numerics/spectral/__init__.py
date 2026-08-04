"""Model-neutral spectral preparation, statistics, and references."""

from chaos_numerics.spectral.levels import (
    PreparedEigenphases,
    SpacingDistributionResult,
    UnfoldedSpectrum,
    adjacent_gap_ratios,
    prepare_eigenphases,
    spacing_distribution,
    unfold,
)
from chaos_numerics.spectral.long_range import (
    SpectralCurve,
    number_variance,
    rmt_reference,
    spectral_form_factor,
)

__all__ = [
    "PreparedEigenphases",
    "SpacingDistributionResult",
    "SpectralCurve",
    "UnfoldedSpectrum",
    "adjacent_gap_ratios",
    "number_variance",
    "prepare_eigenphases",
    "rmt_reference",
    "spacing_distribution",
    "spectral_form_factor",
    "unfold",
]

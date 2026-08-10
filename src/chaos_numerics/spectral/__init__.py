"""Model-neutral spectral preparation, statistics, and references.

Three ``Literal`` type aliases are part of the public surface because they appear
in the signatures of the exported functions, and a library that ships ``py.typed``
has to let callers annotate their own wrappers with the same types:

``Ensemble``
    Random-matrix ensemble label. Accepted by :func:`rmt_reference` and
    :func:`mean_gap_ratio_reference`. One of ``"poisson"``, ``"goe"``, ``"gue"``,
    ``"cue"``, ``"coe"``, ``"gse"``, ``"cse"``; ``"coe"`` is an alias of ``"goe"``
    and ``"cse"`` of ``"gse"``, because bulk statistics only depend on the Dyson
    index. No model in this library produces GSE (beta=4) statistics -- that needs
    an antiunitary symmetry squaring to ``-1``, so half-integer spin, and every map
    here is spinless -- so the beta=4 references exist for spectra the caller
    brings, and they describe the *distinct* levels of a Kramers-degenerate
    spectrum, one per doublet.

``Statistic``
    Which reference curve :func:`rmt_reference` should evaluate. One of
    ``"spectral_form_factor"``, ``"number_variance"``, ``"spacing_distribution"``,
    ``"spectral_rigidity"``, ``"gap_ratio_distribution"``.

``Window``
    Spectral window applied by :func:`spectral_form_factor` before the Fourier
    transform. Either ``"none"`` (flat) or ``"hann"``.

The canonical-ensemble label that the reference implementation resolves aliases
onto is deliberately *not* exported: it is an implementation detail of the curve
evaluation, and pinning it would have prevented adding ``"gse"`` to it later,
which is exactly what happened.
"""

from chaos_numerics.spectral._ensembles import Ensemble
from chaos_numerics.spectral.levels import (
    PreparedEigenphases,
    SpacingDistributionResult,
    UnfoldedSpectrum,
    adjacent_gap_ratios,
    mean_gap_ratio_reference,
    prepare_eigenphases,
    spacing_distribution,
    unfold,
)
from chaos_numerics.spectral.long_range import (
    SpectralCurve,
    Statistic,
    Window,
    number_variance,
    rmt_reference,
    spectral_form_factor,
    spectral_rigidity,
)

__all__ = [
    "Ensemble",
    "PreparedEigenphases",
    "SpacingDistributionResult",
    "SpectralCurve",
    "Statistic",
    "UnfoldedSpectrum",
    "Window",
    "adjacent_gap_ratios",
    "mean_gap_ratio_reference",
    "number_variance",
    "prepare_eigenphases",
    "rmt_reference",
    "spacing_distribution",
    "spectral_form_factor",
    "spectral_rigidity",
    "unfold",
]

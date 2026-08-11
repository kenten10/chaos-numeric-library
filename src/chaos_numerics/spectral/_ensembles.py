"""Random-matrix ensemble labels shared by the level and long-range statistics.

The ``Ensemble`` literal lives here rather than in :mod:`..spectral.long_range`
because :mod:`..spectral.levels` also needs it (``mean_gap_ratio_reference``) and
``long_range`` already imports ``levels``. ``Ensemble`` is re-exported from
:mod:`chaos_numerics.spectral`; ``CanonicalEnsemble`` and the alias table are
implementation details of the reference curves and stay private.
"""

from __future__ import annotations

from typing import Literal, cast, get_args

from chaos_numerics.core import ValidationError

Ensemble = Literal["poisson", "goe", "gue", "cue", "coe", "gse", "cse"]
CanonicalEnsemble = Literal["poisson", "goe", "gue", "cue", "gse"]

#: Bulk statistics only depend on the Dyson index, so the circular ensembles
#: coincide with their Gaussian counterparts: COE == GOE (beta=1),
#: CUE == GUE (beta=2) and CSE == GSE (beta=4). ``"coe"`` and ``"cse"`` are
#: accepted as aliases so that a Floquet spectrum can name its own ensemble:
#: ``"coe"`` for a time-reversal-symmetric map, ``"cse"`` for one that also
#: carries a half-integer spin and is therefore Kramers degenerate. ``"cue"``
#: is kept distinct because supplying a dimension selects a finite-``N``
#: circular kernel that has no Gaussian counterpart here; no such kernel is
#: implemented for beta=4, so ``"cse"`` collapses onto ``"gse"`` outright.
ENSEMBLE_ALIASES: dict[str, CanonicalEnsemble] = {"coe": "goe", "cse": "gse"}


def quoted(options: tuple[object, ...]) -> str:
    """Render literal options for an error message that cannot drift."""
    return ", ".join(repr(option) for option in options)


def canonical_ensemble(ensemble: object) -> CanonicalEnsemble:
    """Validate an ensemble label and resolve ``"coe"``/``"cse"`` onto their class."""
    if not isinstance(ensemble, str) or ensemble not in set(get_args(Ensemble)):
        raise ValidationError(
            f"unsupported RMT ensemble {ensemble!r}; expected one of {quoted(get_args(Ensemble))}"
        )
    return ENSEMBLE_ALIASES.get(ensemble, cast("CanonicalEnsemble", ensemble))

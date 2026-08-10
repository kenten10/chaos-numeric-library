"""Discrete classical maps and classical-dynamics analyses."""

from chaos_numerics.classical.lyapunov import largest_lyapunov_exponent, lyapunov_spectrum
from chaos_numerics.classical.maps import BakerMap, CatMap, LogisticMap, StandardMap
from chaos_numerics.classical.periodic import PeriodicOrbitResult, find_periodic_orbits
from chaos_numerics.classical.sections import PoincareSection, poincare_section
from chaos_numerics.classical.trajectories import iterate
from chaos_numerics.classical.transport import (
    Method,
    Observable,
    autocorrelation,
    local_diffusion_exponent,
    mean_square_displacement,
)

__all__ = [
    "BakerMap",
    "CatMap",
    "LogisticMap",
    "Method",
    "Observable",
    "PeriodicOrbitResult",
    "PoincareSection",
    "StandardMap",
    "autocorrelation",
    "find_periodic_orbits",
    "iterate",
    "largest_lyapunov_exponent",
    "local_diffusion_exponent",
    "lyapunov_spectrum",
    "mean_square_displacement",
    "poincare_section",
]

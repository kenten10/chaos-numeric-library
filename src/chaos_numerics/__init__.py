"""Numerical tools for reproducible experiments in classical and quantum chaos.

The top level re-exports exactly two things: the four shared result containers and
every built-in model. Analysis functions stay in their domain subpackage, so that
`from chaos_numerics.spectral import unfold` says which layer the call belongs to.
"""

from chaos_numerics._version import __version__ as __version__
from chaos_numerics.classical import BakerMap, CatMap, LogisticMap, StandardMap
from chaos_numerics.core import AnalysisResult, EigenstateResult, Spectrum, Trajectory
from chaos_numerics.experiment import Experiment, run_experiment, run_sweep
from chaos_numerics.quantum import (
    CylinderKickedRotor,
    KickedRotor,
    QuantumBakerMap,
    QuantumCatMap,
)

__all__ = [
    "AnalysisResult",
    "BakerMap",
    "CatMap",
    "CylinderKickedRotor",
    "EigenstateResult",
    "Experiment",
    "KickedRotor",
    "LogisticMap",
    "QuantumBakerMap",
    "QuantumCatMap",
    "Spectrum",
    "StandardMap",
    "Trajectory",
    "__version__",
    "run_experiment",
    "run_sweep",
]

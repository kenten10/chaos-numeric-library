"""Numerical tools for reproducible experiments in classical and quantum chaos."""

from chaos_numerics._version import __version__ as __version__
from chaos_numerics.classical import BakerMap, CatMap, StandardMap
from chaos_numerics.core import AnalysisResult, EigenstateResult, Spectrum, Trajectory
from chaos_numerics.experiment import Experiment, run_experiment, run_sweep
from chaos_numerics.quantum import KickedRotor, QuantumBakerMap, QuantumCatMap

__all__ = [
    "AnalysisResult",
    "BakerMap",
    "CatMap",
    "EigenstateResult",
    "Experiment",
    "KickedRotor",
    "QuantumBakerMap",
    "QuantumCatMap",
    "Spectrum",
    "StandardMap",
    "Trajectory",
    "__version__",
    "run_experiment",
    "run_sweep",
]

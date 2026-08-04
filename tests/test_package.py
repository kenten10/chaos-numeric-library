"""Smoke tests for the installable package boundary."""

from importlib.metadata import version
from importlib.resources import files

import chaos_numerics


def test_package_import_and_version() -> None:
    """The installed distribution and imported package expose one version."""
    assert chaos_numerics.__version__ == version("chaos-numerics")
    assert chaos_numerics.__all__ == [
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


def test_typing_marker_is_packaged() -> None:
    """The PEP 561 marker is present for downstream type checkers."""
    assert files("chaos_numerics").joinpath("py.typed").is_file()

"""Smoke tests for the installable package boundary."""

import re
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import pytest

import chaos_numerics

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_citation_metadata_declares_the_same_version() -> None:
    """``CITATION.cff`` carries the only copy of the version outside the package.

    ``pyproject.toml`` reads the version from ``_version.py`` through Hatchling, so
    the build and the import agree by construction. A citation file cannot be wired
    into the build the same way, which is what this test is for.
    """
    citation = REPOSITORY_ROOT / "CITATION.cff"
    if not citation.is_file():  # pragma: no cover - only when tests run from a wheel
        pytest.skip("CITATION.cff is not part of the installed distribution")
    match = re.search(
        r"^version:\s*(?P<version>\S+)\s*$", citation.read_text("utf-8"), re.MULTILINE
    )
    assert match is not None, "CITATION.cff must declare a version"
    assert match.group("version") == chaos_numerics.__version__


def test_package_import_and_version() -> None:
    """The installed distribution and imported package expose one version."""
    assert chaos_numerics.__version__ == version("chaos-numerics")
    assert chaos_numerics.__all__ == [
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


def test_typing_marker_is_packaged() -> None:
    """The PEP 561 marker is present for downstream type checkers."""
    assert files("chaos_numerics").joinpath("py.typed").is_file()

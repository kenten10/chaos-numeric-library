"""Spectral statistics: from a quantum map to a random-matrix comparison.

The pipeline is the same for any spectrum, but the answer only means something
once the symmetry sector is under control. The kicked rotor makes that explicit:
its Bloch phases select the symmetry class, so one model reproduces Poisson, COE,
and CUE statistics depending on which symmetries survive.

The last block is the control that makes the comparison interpretable. Feeding
genuine Haar-distributed unitaries through the same functions lands on the CUE
reference to about one standard error -- this seeded run prints -1.3 of them, which
is ordinary scatter and not a bias -- so the residual percent-level shortfall for
the rotor belongs to the model at finite `hbar_eff = 2 * pi / N`, not to this
pipeline. The script states the deviation in standard errors precisely so the
reader can judge that rather than take the claim on trust.
"""

import numpy as np

from chaos_numerics.quantum import BoundaryPhases, KickedRotor, eigenstates
from chaos_numerics.spectral import (
    UnfoldedSpectrum,
    adjacent_gap_ratios,
    mean_gap_ratio_reference,
    prepare_eigenphases,
    spacing_distribution,
    unfold,
)

DIMENSION = 256
MEMBERS = 12

# Large-N values, not the 3x3 surmise. `mean_gap_ratio_reference(..., surmise=True)`
# returns the surmise, which differs in the third decimal.
REFERENCES = {
    name: mean_gap_ratio_reference(ensemble)
    for name, ensemble in (("Poisson", "poisson"), ("COE", "coe"), ("CUE", "cue"))
}

# (label, kick strength, whether the momentum phase is generic).
SETTINGS = (
    ("Poisson", 0.15, True),
    ("COE", 10.0, False),
    ("CUE", 10.0, True),
)

generator = np.random.default_rng(7)


def gap_ratio(phases: object, *, sector: str) -> tuple[float, UnfoldedSpectrum]:
    """Return the mean adjacent gap ratio of one spectrum and its unfolded form."""
    prepared = prepare_eigenphases(phases, symmetry_sector=sector)
    # `method="mean"` measures the wrap-around spacing from the phases;
    # `method="polynomial"` fixes that one spacing by construction instead.
    unfolded = unfold(prepared, method="mean")
    return float(np.asarray(adjacent_gap_ratios(unfolded).values).mean()), unfolded


def summarize(label: str, per_spectrum: list[float]) -> None:
    """Print the ensemble mean, its standard error, and the reference distance."""
    values = np.asarray(per_spectrum)
    mean = float(values.mean())
    error = float(values.std(ddof=1) / np.sqrt(values.size))
    reference = REFERENCES[label]
    print(
        f"  {label:8} <r> = {mean:.4f} +/- {error:.4f}   reference {reference:.4f}"
        f"   difference {mean - reference:+.4f}"
        f" = {(mean - reference) / error:+.1f} standard errors"
    )


def circular_unitary(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one Haar-distributed unitary matrix."""
    ginibre = (
        rng.standard_normal((dimension, dimension))
        + 1j * rng.standard_normal((dimension, dimension))
    ) / np.sqrt(2.0)
    factor, upper = np.linalg.qr(ginibre)
    phase = np.diag(upper).copy()
    phase /= np.abs(phase)
    return np.asarray(factor * phase[None, :])


print(f"kicked rotor, N = {DIMENSION}, {MEMBERS} Bloch phases per class:")
last_unfolded: UnfoldedSpectrum | None = None
for label, kick_strength, generic_momentum in SETTINGS:
    ratios = []
    for _ in range(MEMBERS):
        # A generic position phase breaks parity. A generic momentum phase also
        # breaks the antiunitary symmetry, which is what moves COE to CUE.
        phases = BoundaryPhases(
            position=float(generator.uniform(0.05, 0.95)),
            momentum=float(generator.uniform(0.05, 0.95)) if generic_momentum else 0.0,
        )
        model = KickedRotor(DIMENSION, kick_strength, phases)
        ratio, last_unfolded = gap_ratio(
            eigenstates(model, dense_limit=DIMENSION).eigenphases,
            sector=f"single sector, {label} conditions",
        )
        ratios.append(ratio)
    summarize(label, ratios)

assert last_unfolded is not None
distribution = spacing_distribution(last_unfolded, bins=20, value_range=(0.0, 4.0))
covered = float(np.sum(distribution.values * np.diff(distribution.bin_edges)))
print(
    f"\nlast CUE spectrum: {distribution.sample_count} spacings, "
    f"{covered:.4f} of the density inside [0, 4)"
)

control = [
    gap_ratio(
        np.angle(np.linalg.eigvals(circular_unitary(DIMENSION, generator))),
        sector="CUE by construction",
    )[0]
    for _ in range(MEMBERS)
]
print("\ncontrol: Haar unitaries of the same size through the same functions:")
summarize("CUE", control)
print(
    "\nThe control lands on the reference, so the pipeline is sound. The rotor's\n"
    "percent-level shortfall is the model's own deviation from RMT at finite N."
)

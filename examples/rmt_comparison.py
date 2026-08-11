"""Compare long-range spectral curves against RMT for three spectra.

Two of them are the same map at two dimensions, which is the point. The Saraceno
baker map is uniformly hyperbolic classically, and once its parity sectors are
separated a generic even dimension follows COE. At `N = 2**k` it does not: the
classical map is the binary shift, so a power-of-two dimension resonates with it
and an arithmetic structure survives desymmetrization. Reading either dimension
alone gives the wrong general statement about the map.

The kicked rotor with generic Bloch phases, which breaks parity and the antiunitary
symmetry, follows CUE closely. For a genuine counterexample to RMT universality in
a uniformly hyperbolic map, see the arithmetic degeneracies of the quantum cat map
in `notebooks/known_results.ipynb`, section 9.

A single spectrum does not self-average, so the form factor is averaged over an
ensemble of Bloch phases for the rotor. The `bootstrap` argument estimates the
resampling error of one spectrum and is not a substitute for that average.
"""

import numpy as np

from chaos_numerics import QuantumBakerMap
from chaos_numerics.quantum import BoundaryPhases, KickedRotor, desymmetrize, eigenstates
from chaos_numerics.spectral import (
    adjacent_gap_ratios,
    mean_gap_ratio_reference,
    number_variance,
    prepare_eigenphases,
    rmt_reference,
    spacing_distribution,
    spectral_form_factor,
    unfold,
)

SEED = 7
TIMES = np.linspace(0.0, 2.0, 101)
LENGTHS = np.linspace(0.0, 8.0, 41)

REFERENCE_GAP_RATIOS = {
    name: mean_gap_ratio_reference(ensemble)
    for name, ensemble in (("Poisson", "poisson"), ("COE", "coe"), ("CUE", "cue"))
}


def parity_even_baker(dimension: int) -> tuple[object, float]:
    """Return the unfolded parity-even baker spectrum and its mean gap ratio."""
    model = QuantumBakerMap(dimension=dimension)
    eigensystem = eigenstates(model)
    # Mixing the two parity blocks would dilute level repulsion and make every
    # statistic below look more Poisson-like than the map really is.
    resolved = desymmetrize(eigensystem, model.symmetry_operators["parity"], sector="even")
    prepared = prepare_eigenphases(resolved, symmetry_sector="parity-even")
    unfolded = unfold(prepared, method="mean")
    ratios = np.asarray(adjacent_gap_ratios(unfolded).values)
    return unfolded, float(ratios.mean())


def cue_rotor_ensemble(dimension: int, members: int) -> tuple[list[object], float]:
    """Return unfolded rotor spectra over random Bloch phases and their mean ratio."""
    generator = np.random.default_rng(SEED)
    ensemble = []
    ratios = []
    for _ in range(members):
        phases = BoundaryPhases(
            position=float(generator.uniform(0.05, 0.95)),
            momentum=float(generator.uniform(0.05, 0.95)),
        )
        model = KickedRotor(dimension, 10.0, phases)
        prepared = prepare_eigenphases(
            eigenstates(model, dense_limit=dimension).eigenphases,
            symmetry_sector="parity and antiunitary symmetry both broken",
        )
        unfolded = unfold(prepared, method="mean")
        ensemble.append(unfolded)
        ratios.append(np.asarray(adjacent_gap_ratios(unfolded).values))
    return ensemble, float(np.concatenate(ratios).mean())


generic_spectrum, generic_ratio = parity_even_baker(126)
power_spectrum, power_ratio = parity_even_baker(128)
baker_spectrum, baker_ratio = generic_spectrum, generic_ratio
rotor_ensemble, rotor_ratio = cue_rotor_ensemble(256, members=12)

baker_form_factor = spectral_form_factor(
    baker_spectrum, TIMES, window="hann", bootstrap=100, seed=SEED
)
baker_variance = number_variance(baker_spectrum, LENGTHS, samples=4096, seed=SEED)

rotor_form_factor = np.mean(
    [np.asarray(spectral_form_factor(item, TIMES).values) for item in rotor_ensemble],
    axis=0,
)
rotor_variance = np.mean(
    [
        np.asarray(number_variance(item, LENGTHS, samples=4096, seed=SEED).values)
        for item in rotor_ensemble
    ],
    axis=0,
)

poisson_variance = rmt_reference("number_variance", "poisson", LENGTHS)
goe_variance = rmt_reference("number_variance", "goe", LENGTHS)
cue_variance = rmt_reference("number_variance", "cue", LENGTHS)
cue_form_factor = rmt_reference("spectral_form_factor", "cue", TIMES)

print("reference mean adjacent gap ratios:")
for name, value in REFERENCE_GAP_RATIOS.items():
    print(f"  {name:8} {value:.4f}")
print(f"parity-even baker, N=126 (generic) <r> = {generic_ratio:.4f}  -> COE")
print(f"parity-even baker, N=128 (2**k)    <r> = {power_ratio:.4f}  -> not COE")
print(f"kicked rotor, generic Bloch phases <r> = {rotor_ratio:.4f}  -> CUE")
print("Same map, same procedure: only the dimension changed.")

index = int(np.argmin(np.abs(LENGTHS - 5.0)))
print(f"\nnumber variance at L = {LENGTHS[index]:.1f}:")
print(f"  baker (parity-even, N=126) {baker_variance.values[index]:.3f}")
print(f"  rotor ensemble      {rotor_variance[index]:.3f}")
print(f"  Poisson reference   {poisson_variance.values[index]:.3f}")
print(f"  GOE/COE reference   {goe_variance.values[index]:.3f}")
print(f"  GUE/CUE reference   {cue_variance.values[index]:.3f}")

# Histogram the whole rotor ensemble against the surmise. Each Bloch phase is a
# separate spectrum, so the counts are pooled rather than the spectra concatenated.
EDGES = np.linspace(0.0, 4.0, 21)
WIDTH = float(EDGES[1] - EDGES[0])
pooled = np.zeros(EDGES.size - 1)
total = 0
for item in rotor_ensemble:
    histogram = spacing_distribution(item, bins=EDGES, density=False)
    pooled += np.asarray(histogram.values)
    total += histogram.sample_count
density = pooled / (total * WIDTH)
centers = 0.5 * (EDGES[:-1] + EDGES[1:])
surmise = np.asarray(rmt_reference("spacing_distribution", "cue", centers).values)
# Expected counting noise per bin, for reading the deviation below.
noise = np.sqrt(np.maximum(pooled, 1.0)) / (total * WIDTH)
print(
    f"\nspacing distribution of the rotor ensemble vs the Wigner surmise:"
    f"\n  {total} spacings pooled over {len(rotor_ensemble)} spectra"
    f"\n  maximum deviation {np.abs(density - surmise).max():.3f}"
    f", largest counting noise {noise.max():.3f}"
)
print(
    "  the surmise peaks at s = sqrt(pi)/2 = "
    f"{np.sqrt(np.pi) / 2.0:.4f} with height (8/pi)*exp(-1) = "
    f"{8.0 / np.pi * np.exp(-1.0):.4f}"
)

ramp = (TIMES > 0.05) & (TIMES < 0.5)
slope = np.polyfit(TIMES[ramp], rotor_form_factor[ramp], 1)[0]
print(f"\nrotor form-factor ramp slope on 0.05 < tau < 0.5: {slope:.3f} (CUE: 1)")
print(f"rotor plateau height for tau > 1.5: {rotor_form_factor[TIMES > 1.5].mean():.3f} (CUE: 1)")
print(
    "shapes:",
    baker_form_factor.values.shape,
    rotor_form_factor.shape,
    cue_form_factor.values.shape,
)

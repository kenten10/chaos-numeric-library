"""Prepare, unfold, and summarize one symmetry-resolved quantum spectrum.

The baker map is desymmetrized before any statistic is computed: mixing the two
parity sectors suppresses level repulsion and makes the spectrum look more
Poisson-like than it is.

Desymmetrization is necessary but not sufficient, and the dimension decides which.
The classical baker map is the binary shift, so `N = 2**k` resonates with it and an
arithmetic structure survives the parity split; at a generic even dimension both
sectors land on COE. That is a physical choice of `N`, not a resolution knob, so
this example runs both and prints the contrast.
"""

import numpy as np

from chaos_numerics import QuantumBakerMap
from chaos_numerics.quantum import desymmetrize, eigenstates
from chaos_numerics.spectral import (
    adjacent_gap_ratios,
    mean_gap_ratio_reference,
    prepare_eigenphases,
    spacing_distribution,
    unfold,
)

REFERENCES = {
    name: mean_gap_ratio_reference(ensemble)
    for name, ensemble in (("Poisson", "poisson"), ("COE", "coe"), ("CUE", "cue"))
}

for dimension in (126, 128):
    model = QuantumBakerMap(dimension=dimension)
    eigensystem = eigenstates(model)
    parity = model.symmetry_operators["parity"]
    power_of_two = dimension & (dimension - 1) == 0
    print(f"N = {dimension}" + ("  (a power of two)" if power_of_two else "  (generic even)"))

    for sector in ("even", "odd"):
        resolved = desymmetrize(eigensystem, parity, sector=sector)
        prepared = prepare_eigenphases(resolved, symmetry_sector=f"parity-{sector}")
        # `method="mean"` measures the circular gap from the phases. With
        # `method="polynomial"` that one gap is fixed by the rescaling instead.
        unfolded = unfold(prepared, method="mean")
        ratios = adjacent_gap_ratios(unfolded)
        distribution = spacing_distribution(unfolded, bins=30, value_range=(0.0, 4.0))
        widths = np.diff(distribution.bin_edges)
        integral = float(np.sum(distribution.values * widths))

        print(f"  parity-{sector}: levels={unfolded.count} of {dimension}")
        print(
            f"    maximum |<psi|S|psi>| - 1 = "
            f"{resolved.metadata.parameters['sector_expectation_deviation']:.2e}"
        )
        print(f"    mean gap ratio        = {ratios.metadata.parameters['mean_ratio']:.6f}")
        print(
            f"    histogram integral    = {integral:.12f} over {distribution.sample_count} spacings"
        )

print("\nreference mean adjacent gap ratios (large N):")
for name, value in REFERENCES.items():
    print(f"  {name:8} {value:.4f}")
print("At a generic even N the desymmetrized baker map follows COE; N = 2**k does not.")

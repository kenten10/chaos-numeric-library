"""Construct matching classical and quantum cat/baker map conventions."""

from chaos_numerics import BakerMap, CatMap, QuantumBakerMap, QuantumCatMap
from chaos_numerics.quantum import eigenstates, unitarity_defect

classical_cat = CatMap(matrix=((2, 1), (1, 1)))
quantum_cat = QuantumCatMap(dimension=64, matrix=classical_cat.matrix)

classical_baker = BakerMap(cut=0.5)
quantum_baker = QuantumBakerMap(dimension=64)

for classical, quantum in (
    (classical_cat, quantum_cat),
    (classical_baker, quantum_baker),
):
    spectrum = eigenstates(quantum)
    print(
        type(classical).__name__,
        quantum.parameters["quantization"],
        f"unitarity={unitarity_defect(quantum):.3e}",
        f"parity={spectrum.metadata.parameters['symmetry_defects']['parity']:.3e}",
    )

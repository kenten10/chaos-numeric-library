"""Finite-dimensional quantum maps and phase-space state analyses.

Three type aliases are part of the public surface because they appear in the
signatures of the exported callables, and a library that ships ``py.typed`` has
to let callers annotate their own wrappers with the same types:

``CatMatrix``
    Row-major ``2 x 2`` integer matrix, ``((a, b), (c, d))``, taken by the
    ``matrix`` parameter of :class:`QuantumCatMap`. The determinant must be
    exactly ``1`` and ``b`` must be ``+/-1``.

``EvolutionMethod``
    Which algorithm :func:`evolve` should run. One of ``"auto"``, ``"dense"``,
    ``"matrix_free"``, ``"fft"``.

``SymmetrySector``
    Which eigenvalue block :func:`desymmetrize` should keep. Either ``"even"``
    (``<psi|S|psi> = +1``) or ``"odd"`` (``-1``).

The defining modules carry the full description of each one.
"""

from chaos_numerics.quantum.diagnostics import loschmidt_echo, otoc, weyl_translations
from chaos_numerics.quantum.evolution import EvolutionMethod, QuantumEvolution, evolve
from chaos_numerics.quantum.kicked_rotor import CylinderKickedRotor, KickedRotor
from chaos_numerics.quantum.maps import CatMatrix, QuantumBakerMap, QuantumCatMap
from chaos_numerics.quantum.phase_space import (
    HusimiResult,
    WignerResult,
    coherent_state,
    husimi_distribution,
    inverse_participation_ratio,
    participation_ratio,
    shannon_entropy,
    wigner_distribution,
)
from chaos_numerics.quantum.states import (
    BoundaryPhases,
    QuantumBasis,
    basis_state,
    normalize_state,
    quantum_state,
)
from chaos_numerics.quantum.unitary import (
    DEFAULT_DENSE_LIMIT,
    DenseUnitary,
    SymmetrySector,
    desymmetrize,
    eigenphases,
    eigenstates,
    unitarity_defect,
)

__all__ = [
    "DEFAULT_DENSE_LIMIT",
    "BoundaryPhases",
    "CatMatrix",
    "CylinderKickedRotor",
    "DenseUnitary",
    "EvolutionMethod",
    "HusimiResult",
    "KickedRotor",
    "QuantumBakerMap",
    "QuantumBasis",
    "QuantumCatMap",
    "QuantumEvolution",
    "SymmetrySector",
    "WignerResult",
    "basis_state",
    "coherent_state",
    "desymmetrize",
    "eigenphases",
    "eigenstates",
    "evolve",
    "husimi_distribution",
    "inverse_participation_ratio",
    "loschmidt_echo",
    "normalize_state",
    "otoc",
    "participation_ratio",
    "quantum_state",
    "shannon_entropy",
    "unitarity_defect",
    "weyl_translations",
    "wigner_distribution",
]

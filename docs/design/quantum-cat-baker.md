# Quantum cat and baker map conventions

Status: implemented baseline for KEN-123  
Last reviewed: 2026-08-04

## Quantum cat map

`QuantumCatMap` quantizes an integer matrix
`A = ((a, b), (c, d))` with exact determinant one. The supported v0.1
convention requires even Hilbert-space dimension, periodic boundary phases
`(alpha, beta) = (0, 0)`, and `b = +1` or `b = -1`. These constraints make the
position-space generating function single-valued on the selected finite torus;
other metaplectic sectors are intentionally deferred.

In the position order `j = 0, ..., N - 1`, the kernel is

```text
U[k,j] = exp(-i sign(b) pi/4) / sqrt(N)
         * exp(2 pi i / N * (a j^2 - 2 j k + d k^2) / (2 b)).
```

The default matrix `((2, 1), (1, 1))` matches the classical `CatMap` default.
Its parity operator sends `j` to `-j mod N`.

## Quantum baker map

`QuantumBakerMap` implements the parity-preserving Saraceno quantization of the
symmetric classical baker map (`cut=0.5`). It requires even `N` and anti-periodic
boundary phases `(alpha, beta) = (1/2, 1/2)`. With the twisted Fourier matrix

```text
G_N[k,j] = exp(-2 pi i (k+1/2)(j+1/2)/N) / sqrt(N),
```

the Floquet operator is

```text
B_N = G_N* diag(G_(N/2), G_(N/2)).
```

Parity sends `j` to `N - 1 - j`.

### Choosing `N` for spectral statistics

Avoid `N = 2^k`. The classical map is the binary shift and a power-of-two
dimension resonates with it, leaving an arithmetic structure in the spectrum that
survives desymmetrization. Measured mean adjacent gap ratio per parity sector
against the COE reference 0.5307, standard error of the sector mean in brackets:

| `N`  | even sector     | odd sector      |
| ---- | --------------- | --------------- |
| 256  | 0.3806 (0.0244) | 0.4038 (0.0238) |
| 512  | 0.4078 (0.0177) | 0.4309 (0.0175) |
| 1024 | 0.4389 (0.0122) | 0.4559 (0.0122) |
| 700  | 0.5211 (0.0134) | 0.5324 (0.0134) |
| 802  | 0.5386 (0.0129) | 0.5568 (0.0132) |
| 900  | 0.5093 (0.0120) | 0.5310 (0.0120) |

The raw two-sector spectrum sits at 0.42 at every one of these dimensions. The
power-of-two sectors stay 6 to 8 standard errors below COE and hardly improve on
that; the generic even sectors agree with COE to within 2. Dimensions carrying a
large power of two are intermediate (`768 = 3 * 2^8`, even sector 0.5170). This is
a property of the map, not of the implementation: the unitarity defect stays at or
below `2.0e-13` and the parity expectation values are `+/-1` to `2.3e-15` at every
`N` in the table. Pick 700, 802, or 900 for an RMT comparison; `N = 2^k` remains
the right choice for studying the symbolic dynamics itself.

Odd dimensions, incompatible boundary phases,
non-unit-determinant cat matrices, and unsupported cat generating functions raise
`ValidationError` before a Floquet matrix is allocated.

Both models implement the common `QuantumMap` protocol, so `evolve`,
`eigenphases`, `eigenstates`, and `unitarity_defect` work unchanged. Dense
eigensystem metadata records the minimum circular eigenphase gap, detected
degenerate pairs, and normalized commutator defects for declared symmetries.

## Phase-space pictures

Both models satisfy the even-`N` requirement of `wigner_distribution`, and both
carry a nonzero twist that has to be passed explicitly, because the phase-space
functions default to `(0, 0)`:

| model             | required `boundary_phases`      | call                                                    |
| ----------------- | ------------------------------- | ------------------------------------------------------- |
| `QuantumCatMap`   | `BoundaryPhases(0, 0)`          | `wigner_distribution(psi)`                               |
| `QuantumBakerMap` | `BoundaryPhases(0.5, 0.5)`      | `wigner_distribution(psi, boundary_phases=BoundaryPhases(0.5, 0.5))` |

Passing the wrong twist does not raise: it silently changes the values (through
`beta`) and the grid labels (through `alpha`). `docs/design/quantum-phase-space.md`
records the convention and the exact marginal identities that a mismatched twist
breaks.

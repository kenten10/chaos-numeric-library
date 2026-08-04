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

Parity sends `j` to `N - 1 - j`. Odd dimensions, incompatible boundary phases,
non-unit-determinant cat matrices, and unsupported cat generating functions raise
`ValidationError` before a Floquet matrix is allocated.

Both models implement the common `QuantumMap` protocol, so `evolve`,
`eigenphases`, `eigenstates`, and `unitarity_defect` work unchanged. Dense
eigensystem metadata records the minimum circular eigenphase gap, detected
degenerate pairs, and normalized commutator defects for declared symmetries.

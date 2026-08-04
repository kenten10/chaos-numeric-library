# Kicked-rotor quantization convention

Status: implemented baseline for KEN-122  
Last reviewed: 2026-08-04

`KickedRotor` represents states in an `N`-point position basis, with the trailing
array axis ordered by `j = 0, ..., N - 1`. For boundary phases `(alpha, beta)`
measured in turns modulo one,

```text
q_j = (j + alpha) / N
hbar_eff = 2 pi / N
<p_k|q_j> = exp(-2 pi i (k + beta)(j + alpha) / N) / sqrt(N).
```

The DFT row index is NumPy FFT order. The kinetic phase uses the corresponding
signed modes from `numpy.fft.fftfreq(N) * N + beta`; this defines the Nyquist-mode
choice for even `N` as well as the odd-`N` ordering. A scalar `boundary_phase`
sets `alpha = beta`; `BoundaryPhases` specifies them independently.

One Floquet step is free-after-kick:

```text
D_kick[j] = exp(-i K cos(2 pi q_j) / hbar_eff)
D_free[k] = exp(-i hbar_eff n_k^2 / 2)
U = F* D_free F D_kick.
```

The FFT path factors the twisted `F` into diagonal phases and
`numpy.fft.fft(..., norm="ortho")`; the inverse uses `ifft` with the same
normalization. It costs `O(N log N)` time per state and `O(N)` stored phases,
without allocating an `N x N` operator. `to_dense()` independently evaluates
the DFT entries and matrix product as an `O(N^2)`-memory small-system reference.

Tests cover odd and even dimensions, zero and nonzero independent boundary
phases, ten seeded states per configuration, dense unitarity, the matrix-free
adjoint, and norm drift over 1,000 FFT steps.

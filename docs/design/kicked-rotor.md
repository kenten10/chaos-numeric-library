# Kicked-rotor quantization convention

Status: implemented baseline for KEN-122  
Last reviewed: 2026-08-09

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
choice for even `N` as well as the odd-`N` ordering. A scalar `boundary_phases`
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

## Naming: the literature's symbols and the API's names

The parameter names are spelled out rather than abbreviated, so a formula copied
from a paper has to be translated once. There are no single-letter aliases, by
design: an alias that a type checker cannot see is worse than a `TypeError`, and
`KickedRotor(64, K=5.0)` is a plain `TypeError` on purpose.

| literature | this API                                   | notes                                    |
| ---------- | ------------------------------------------ | ---------------------------------------- |
| `K`        | `kick_strength` (2nd positional argument)  | `KickedRotor(64, K=5.0)` is a `TypeError` |
| `N`        | `dimension` (1st positional argument)      | Hilbert-space size                       |
| `hbar_eff` | `effective_hbar` (read-only attribute)     | fixed to `2*pi/N` on the torus           |
| `hbar`     | `effective_hbar` (keyword argument)        | free parameter on the cylinder           |
| `alpha`    | `boundary_phases.position`                 | turns modulo one                         |
| `beta`     | `boundary_phases.momentum`                 | turns modulo one                         |
| `n_k`      | `momentum_numbers[k]`                      | `fftfreq(N)*N + beta`                     |
| `theta`    | `2*pi*q`                                   | the angle, not the turn coordinate       |
| `P`        | `hbar*n`                                   | canonically conjugate to `theta`         |

## Basis changes

`to_momentum_basis` and `to_position_basis` publish the twisted DFT `F` above for
scalar and batched `(..., N)` states, and `momentum_numbers` labels its output
axis with `fftfreq(N)*N + beta` — the `CylinderKickedRotor` ordering plus the
momentum twist. Round trips are unitary to `2.5e-16`, norms agree to `4.5e-16`,
and the transform matches the DFT entries above to `1.2e-14`, over `N` in
`{7, 8, 63, 64, 65, 128}` and three twists.

**`numpy.fft.fft` is not a substitute at nonzero twist.** Three separate errors,
measured at `N = 64` over 200 random unit states:

| mistake                    | effect on amplitudes             | effect on `abs(.)^2`        |
| -------------------------- | -------------------------------- | --------------------------- |
| omitting `norm="ortho"`    | scaled by `sqrt(N)`              | every bin `N = 64` too large |
| omitting `alpha`           | `norm` of the error 0.74–0.99    | unchanged, `4.2e-17`        |
| omitting `beta`            | —                                | total variation 0.087–0.147 at `beta=0.13`, 0.285–0.510 at `beta=0.5` |

So `alpha` moves only phases and a momentum histogram cannot detect its absence,
while `beta` moves probability between bins. At `(alpha, beta) = (0, 0)` the bare
FFT agrees bit for bit, which is why the mistake survives a test written at the
default twist.

The two rotor classes store opposite bases — `KickedRotor` position amplitudes,
`CylinderKickedRotor` momentum amplitudes — so the method names say which basis
comes *out*: on the cylinder `to_position_basis` consumes the stored state and
returns angle amplitudes on `theta_j = 2*pi*j/N`, and `to_momentum_basis` is its
inverse.

## Reflection symmetry

`symmetry_operators` publishes an exact `q -> -q` reflection whenever `2*alpha`
and `2*beta` are both integers, that is for all four twists with
`alpha, beta in {0, 1/2}` — not only for the two matched ones. On the grid
`q_j=(j+alpha)/N` the reflected point is `-q_j = q_k - w` with
`k = (-j - 2*alpha) mod N` and integer winding `w`, and the momentum twist makes
the wave function quasi-periodic, so

```text
(S psi)_j = exp(-2 pi i beta w_j) psi_{k_j}.
```

The prefactor is `+/-1` exactly when `2*beta` is an integer, which is also what
makes `S` an involution. `alpha` therefore sets the index map and `beta` only sets
the sign on the wrapped entries:

| `alpha` | `beta` | index map        | wrap factor            |
| ------- | ------ | ---------------- | ---------------------- |
| `0`     | `0`    | `j -> -j mod N`  | `+1`                   |
| `0`     | `1/2`  | `j -> -j mod N`  | `-1` for `j != 0`      |
| `1/2`   | `0`    | `j -> N - 1 - j` | `+1`                   |
| `1/2`   | `1/2`  | `j -> N - 1 - j` | `+1` (global sign set) |

The overall sign is a convention and is normalized to `+1` at `j = 0`; without
that the `(1/2, 1/2)` operator comes out as `-P`, which carries the same sectors
with the `even`/`odd` labels exchanged.

Measured `||[U,S]||_F/sqrt(N)` at `N = 128`, `K = 10` is `6.73e-14`, `6.75e-14`,
`6.51e-14`, and `6.48e-14` for those four rows. `(0, 1/2)` is the case that used
to be missed: the bare permutation `j -> -j` has defect `0.2491` there, so a
search restricted to permutations finds nothing and the sector split silently goes
missing, leaving a two-sector spectrum (mean adjacent gap ratio 0.42) to be
compared against COE 0.5307. For every other twist the grid is genuinely not
invariant — the smallest defect over all `N` reflections `j -> c - j` at `N = 128`
is `1.156` for `(0.25, 0)` and `1.218` for `(0.25, 0.13)` — and the mapping is
empty.

## `CylinderKickedRotor` localization lengths

Two lengths are both called `ell` and are not the same. The saturated momentum
profile decays as `exp(-|n|/ell_dist)` and a single Floquet eigenfunction decays
as `exp(-|n - n_0|/ell_eig)` around its own centre. **The textbook `ell ~ D/2`,
with `D` defined by `<(Delta n)^2> = D t`, predicts `ell_eig`, not `ell_dist`.**
Fitting the saturated profile and calling the answer `D/2` overestimates `D` by
about a factor of two.

Measured at `hbar = 1` with the Bessel-corrected quasilinear rate
`D = (K^2/2 hbar^2)(1 - 2 J_2(K/hbar))`, time-averaging over the second half of
the run:

| `K` | `D`    | `ell_eig` | `ell_dist` | `<n^2>_sat` | `D^2`  |
| --- | ------ | --------- | ---------- | ----------- | ------ |
| 5   | 11.336 | 4.48      | 8.59       | 117.9       | 128.5  |
| 8   | 39.231 | 24.33     | 30.38      | 1741.6      | 1539.1 |

So `ell_eig` is 0.79 and 1.24 times `D/2`, `ell_dist` is 0.76 and 0.77 times `D`,
and `ell_dist/ell_eig` is 1.9 and 1.2. The moment follows the distribution:
`<n^2>_sat` is 0.92 and 1.13 times `D^2`, not `D^2/2`. `ell_eig` is the log-linear
slope of the geometric mean over the spectrum of the peak-centred `|phi_n|^2`;
`ell_dist` uses the same self-consistent fit band `ell < |n| < 6*ell` on the
time-averaged profile.

`CylinderKickedRotor` also provides `apply_fft` as an exact alias of `apply`, so
`evolve(..., method="fft")` works on it. There is only one algorithm here and no
dense reference matrix, unlike `KickedRotor` where `apply`, `apply_fft`, and
`apply_dense` are three distinct routes.

## Test coverage

Tests cover odd and even dimensions, zero and nonzero independent boundary
phases, ten seeded states per configuration, dense unitarity, the matrix-free
adjoint, norm drift over 1,000 FFT steps, the published basis change (unitarity,
round trip, agreement with the DFT entries, diagonality of the free rotation in
the momentum basis, the measured size of the bare-FFT mistake, and the
momentum/position IPR split at `K = 0.2`), the reflection operator and its
commutator defect at all four self-reflecting twists, the brute-force absence of a
reflection at generic twists, the two cylinder localization lengths and their
ratio, and the `method="fft"` path on both rotors.

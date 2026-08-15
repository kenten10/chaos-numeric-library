# Torus coherent states and quantum phase-space data

Status: implemented baseline for KEN-126  
Last reviewed: 2026-08-09

## Coherent-state convention

States use the finite position basis `q_j=(j+alpha)/N`. For center `(q0,p0)`,
the unnormalized periodized Gaussian is

```text
psi_j = sum_m exp(-pi*N*(q_j-q0+m)^2)
              * exp(2*pi*i*N*p0*(q_j-q0+m))
              * exp(-2*pi*i*beta*m),       m=-images,...,+images.
```

`alpha` and `beta` come from `BoundaryPhases`; centers are wrapped to `[0,1)`
modulo one, silently and by design, because the torus has no outside:
`position=1.2` names the same point as `position=0.2`. A coordinate outside the
unit square is not an error, and only non-finite values are rejected.
The returned `complex128` state is normalized numerically. The default four
images on each side make omitted Gaussian tails negligible at supported binary64
precision, including centers near the torus cut.

## Husimi convention

`husimi_distribution` evaluates

```text
H(q,p) = N * abs(<q,p|psi>)^2
```

on a configurable `(n_q,n_p)` half-open midpoint grid. Axes and `grid_points`
use classical `(q,p)` order, allowing direct overlay with trajectory coordinates.
The raw midpoint integral is retained. With the default `normalize=True`, the
density is divided by that quadrature integral, making the returned integral one;
`normalize=False` exposes grid-convergence error. Coherent states are generated
in chunks, so auxiliary storage is `O(chunk_size*N*images)` rather than the full
grid tensor.

`HusimiResult` follows the same container contract as `PoincareSection` and the
`core.results` types: stored arrays are read-only, `__eq__` compares values, and
`__reduce__` rebuilds through `__init__` so that a `pickle` or `copy.deepcopy`
round trip preserves both properties. Without the explicit `__reduce__` the
`slots=True` dataclass reduction restores fields one by one, skips
`__post_init__`, and hands back writable arrays.

`HusimiResult.integral` is the midpoint quadrature of the stored density. Under
the default `normalize=True` it is 1 by construction and is therefore not a check
on anything; `raw_integral` is the pre-normalization value and the quantity that
says whether the grid and the `images` count resolved the state.

`metadata_payload()` describes `convergence_history` whenever `array_payload()`
ships it. The two halves of the persistence split must list the same arrays: a
descriptor that omits one makes the storage layer fail with a `KeyError` from a
missing digest instead of a `ValidationError`, and no library path attaches a
history to these containers today, which is how the asymmetry survived.

## Husimi cost

Essentially all of the cost is building the `(n_q*n_p, N)` grid of coherent
states, which depends only on `dimension`, `boundary_phases`, `grid_shape`, and
`images` — never on the state. Two independent ways to amortize it:

1. **Batch.** A `(..., N)` stack builds the grid once and then spends one
   `complex128` matrix product per state. Measured at `N = 1024`, best of three:
   0.55 s for one state and 0.57 s for twenty on a `(64, 64)` grid (19.3x per
   state), 2.25 s and 2.24 s on `(128, 128)` (20.1x).
2. **Module-level LRU cache** of that grid, keyed on exactly those four
   parameters. A repeat call at `N = 1024` / `(64, 64)` costs 0.002 s instead of
   0.55 s. At most four grids and at most 128 MiB (`_CACHE_BUDGET_BYTES`) are
   retained, entries are read-only so one caller cannot corrupt another's, and a
   grid larger than the budget — `N = 1024` on `(128, 128)` needs 256 MiB — is
   never cached at all. `chunk_size` is deliberately not part of the key: the
   grid is assembled row by row from elementwise expressions reduced over the
   image axis alone, so the block width cannot move a single bit, and the tests
   assert exact array equality between the cached path, the same path with a
   different `chunk_size`, and the uncached path with the budget set to zero.

## Wigner convention

The natural Weyl lattice of an `N`-state torus is the *half-integer* one, because
the midpoint of two position grid points generally is not a grid point:

```text
q_m = (m/2 + alpha)/N,  p_n = (n/2 + beta)/N,   m, n = 0, ..., 2N - 1
A[m,n] = sum_j psi_j conj(psi_[m-j]) exp(-2 pi i (n/2 + beta)(2 j - m)/N) / (2 N)
```

with the conjugated index extended quasi-periodically,
`psi_[k+N] = exp(2 pi i beta) psi_k`. Only `beta` enters the values; `alpha`
enters the grid labels alone, because the exponent depends on `q_j - q_j'`.

`wigner_distribution` returns the `2 x 2` block sums of `A`,

```text
W[j,k] = sum over a, b in {0,1} of A[2j+a, 2k+b]
```

on the ordinary lattice `q_j = (j+alpha)/N`, `p_k = (k+beta)/N` — the same `q`
grid the states live on, and `p_k = momentum_numbers[k]/N` modulo one in the DFT
storage order of `to_momentum_basis`. **The block sum is why `N` must be even and
an odd `N` is refused.**

The block sum is not cosmetic. A quasi-periodic state has periodic images one
period apart, so `A` carries a full-strength *ghost* of every feature half a
period away, obeying exactly `A[m+N,n] = (-1)^n A[m,n]` (measured `3.6e-16`).
Those ghosts oscillate at the Nyquist frequency of the `p` axis, so summing
adjacent `n` cancels them — and the pairing `{2k, 2k+1}` commutes with the `+N`
shift only for even `N`. Reported directly, `A` for a single coherent state at
`N = 32` has **four equal maxima** at `(0.25|0.75, 0.25|0.75)` and negative
weight 1.50, indistinguishable from a cat state's 1.47; after the block sum the
maximum is unique and on the requested centre and the negative weights separate
as 0.13 against 0.61.

Exact identities, worst case over `N` in `{2, 4, 6, 8, 16, 32}` and five twists
including `(0.25, 0.13)`:

| identity                                                | measured |
| ------------------------------------------------------- | -------- |
| agreement with an independent `O(N^3)` transcription     | `2.5e-16` |
| discarded imaginary part of `A`                          | `3.9e-16` |
| `sum_k W[j,k] = abs(psi_j)^2`                            | `2.2e-16` |
| `sum_j W[j,k] = abs(to_momentum_basis(psi)[k])^2`        | `5.6e-16` |
| `sum_{j,k} W = 1`                                        | `3.3e-16` |
| `N * sum_{j,k} W_psi W_phi = abs(<psi|phi>)^2`           | `2.2e-16` |

The two marginals are the decisive checks: almost any sign, factor, index, or
missing-twist error breaks one of them. The proportionality constant in the
quadratic form is `1/N`, so `N * sum W^2 = 1` for a pure state.

`W < 0` where a state interferes with itself, which is the reason to have it at
all — the Husimi density is a Gaussian smoothing of the same information and is
non-negative by construction. For `coherent(0.25, 0.5) + coherent(0.75, 0.5)`,
`min W` is `-3.98e-2`, `-2.51e-2`, `-1.40e-2` at `N = 16, 32, 64` against peaks
of `+5.97e-2`, `+3.05e-2`, `+1.54e-2`, while the Husimi minimum on the same grid
is `2.2e-5`, `7.1e-12`, `3.3e-25`. A *position* basis state has `W >= 0` exactly
(`negative_weight == 0.0` bit-exactly); a momentum basis state has it only to
rounding, `1.1e-16` at `N = 8` and `1.7e-16` at `N = 32`, because it reaches the
position basis through the twisted transform. A coherent state's negative weight
shrinks with
`hbar_eff`: 0.19, 0.13, 0.092 at `N = 16, 32, 64`. Some negativity at finite `N`
is therefore expected even for the most classical available state, and
`negative_weight` is comparable only between states of equal `N`.

`WignerResult` follows the same container contract as `HusimiResult`, including
the `convergence_history` descriptor.

## Localization measures

For probabilities `p_j=|psi_j|^2`, the basis localization measures are

```text
IPR = sum_j p_j^2
participation ratio = 1/IPR
Shannon entropy = -sum_j p_j log(p_j)    (natural logarithm).
```

All three operate along the trailing Hilbert-space axis and preserve leading
batch axes. A basis state has `(IPR, PR, entropy)=(1,1,0)`; a uniform state in
dimension `N` has `(1/N,N,log(N))`.

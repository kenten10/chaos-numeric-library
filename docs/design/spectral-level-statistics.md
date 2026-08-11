# Eigenphase preparation and local level statistics

Status: implemented baseline for KEN-124  
Last reviewed: 2026-08-09

`prepare_eigenphases` preserves the input array, wraps a separate copy to the
half-open interval `[0, 2*pi)`, sorts it, and calculates all circular gaps. The
last gap crosses the branch cut from the largest phase to the smallest phase plus
`2*pi`. Gaps at or below `degeneracy_tolerance` are marked as degenerate.

Wrapping goes through `core._validation.wrap_into_half_open`, not `np.mod`.
`np.mod(-1e-18, 2*pi)` rounds up to exactly `2*pi`, which is outside the
documented interval, and `np.angle` legitimately returns tiny negative phases for
eigenvalues just below the positive real axis (an operator close to the identity,
a symmetry-protected eigenvalue at `+1`). The plain modulo therefore made
preparation reject valid unitaries with a `ValidationError`.

Level statistics require a single symmetry sector. Omitting `symmetry_sector`
emits `NumericalWarning` and stores a stable `symmetry-sector-unknown` diagnostic;
mixed independent spectra can give misleading random-matrix comparisons.

`unfold` supports two explicit methods:

- `mean` divides by the global circular mean spacing `2*pi/N`;
- `polynomial` fits the empirical counting ranks as a polynomial of phase,
  requires the fitted map to be strictly increasing, and rescales the circular
  result onto `[0, N-1]`.

`UnfoldedSpectrum` retains raw phases, wrapped/sorted phases, unfolded levels,
unfolded circular spacings, method settings, and fit residual. Empty and
single-level inputs return empty spacing/statistic arrays with diagnostics.

### Two construction identities that are not diagnostics

`mean(spacings) == 1` holds exactly for both methods and every input: circular
gaps sum to the period by construction and there are `N` of them, so the mean is
`period/N` divided by itself. It says nothing about unfolding quality.
`metadata.parameters["mean_unfolded_spacing_is_identity"]` records this. Use
`fit_residual` or a level-statistic comparison to judge an unfolding.

For `method="polynomial"` the final, branch-cut-crossing spacing is `1` by
construction as well, because rescaling the fitted counting function onto
`[0, N-1]` forces the wrap-around gap to `N - (N-1)`. One of the `N` returned
spacings is therefore synthetic rather than measured;
`metadata.parameters["circular_gap_is_synthetic"]` is `True` and a
`synthetic-circular-gap` diagnostic is attached. The gap is not estimated from
the fit because a non-periodic polynomial cannot be extrapolated across the
branch cut reliably — the endpoints are exactly where a least-squares polynomial
is least trustworthy. Use `method="mean"`, which derives every gap including the
wrap-around one from the phases themselves, when that gap matters.

"By construction" is not the same as bit-exact, and the earlier wording ("exactly
`1`") led to a test asserting `== 1.0`. The rescaling is a floating-point
division, so the result is *usually* the bit pattern of `1.0` but need not be:
over 400 random 128-level spectra at degree 5, 336 gave exactly `1.0`, 32 were
off by around `1e-16`, and the other 32 tripped the monotonicity guard below.
Which of the three happens depends on the last bits of `polyfit`, and therefore
on the BLAS the wheel was built against, so the test now allows a few ulp.

### The strictly-increasing guard is a normal outcome, not an edge case

`method="polynomial"` raises `NumericalError` when the fitted counting function
is not strictly increasing on the spectrum, because a non-monotone fit yields
negative unfolded spacings and every downstream statistic then becomes
meaningless. This is not rare: a degree-5 fit turns over inside a large gap for
roughly 8% of 128-level uniform-random spectra. It is covered by a dedicated
`pytest.raises` test rather than being left to chance.

### Local statistics

`adjacent_gap_ratios` includes adjacent pairs across the circular branch cut.
For zero/degenerate gaps, `drop` (default) excludes affected ratios, `zero`
defines them as zero, and `raise` rejects the input.

### Two different reference values for `<r>`

`mean_gap_ratio_reference(ensemble, *, surmise=False)` publishes the expected mean
of `adjacent_gap_ratios` so that callers stop hand-copying it. Two numbers circulate
for every Wigner-Dyson ensemble and the API deliberately makes the choice explicit
rather than picking one silently:

| ensemble | large `N` (default) | 3x3 surmise (`surmise=True`) |
| --- | --- | --- |
| Poisson | `2*ln(2)-1 = 0.386294...` | same, the surmise is exact here |
| GOE/COE | `0.5307` | `0.5359` |
| GUE/CUE | `0.5996` | `0.6027` |
| GSE/CSE | `0.6744` | `0.6762` |

The default is the large-`N` value, because `adjacent_gap_ratios` is normally
applied to a spectrum of many levels and its mean converges there. The surmise
overshoots by `+0.0052` (GOE), `+0.0031` (GUE) and `+0.0018` (GSE) — around 1%,
which is several times the sampling error of a few hundred spectra, so substituting
one for the other shows up as a systematic bias rather than as noise. That is
exactly what the function exists to prevent: before it existed, the tutorial, two
examples and the notebook each carried their own hand-written literal, and the CUE
test asserted against `0.60266`, the *surmise* value, under a comment about the
bulk value.

Only the Poisson entry is computed (`2*np.log(2) - 1`); the beta=1 and beta=2
entries are literature constants carried to the four decimals their sources quote,
so they are accurate to about `+/-1e-4` and must not be asserted on more tightly.
Sources: Oganesyan & Huse, Phys. Rev. B 75, 155111 (2007) for the large-`N`
values; Atas, Bogomolny, Giraud & Roux, Phys. Rev. Lett. 110, 084101 (2013) for
the 3x3 surmise and for both sets in its Table I.

#### The beta=4 row was measured, not quoted

No model in this library generates GSE statistics, so there was nothing internal
to check a remembered constant against and both numbers were produced from
scratch.

The **surmise** entry is the same self-consistency integral that validates the
other rows: `int_0^1 r P_4(r) dr = 0.6761683` of the folded Atas density that
`rmt_reference("gap_ratio_distribution", "gse", ...)` returns, rounded to
`0.6762`.

The **large-`N`** entry comes from the Dumitriu–Edelman beta-Hermite tridiagonal
ensemble (J. Math. Phys. 43, 5830 (2002)), whose eigenvalue density is exactly the
beta=4 one and which costs `eigvalsh_tridiagonal` rather than a dense symplectic
diagonalization. Gap ratios were taken from the central 40% of each `n=4096`
spectrum. What licenses the number is that the *identical* pipeline reproduces the
two published constants to the last digit they are quoted to:

| beta | this pipeline | published |
| --- | --- | --- |
| 1 | `0.53069 +/- 0.00026` | `0.5307` |
| 2 | `0.59960 +/- 0.00023` | `0.5996` |
| 4 | `0.674408 +/- 0.000064` | — |

The beta=4 figure is stable against the matrix size (`0.67416` at `n=1024`,
`0.67424` at `n=8192`) and against the window (`0.67368` on the central 20%), and
an independent 60-member Haar circular-symplectic sample at `2N=128`, run through
this library's own `prepare_eigenphases` pipeline, gives `0.6708 +/- 0.0057`. A
third construction agrees as well: 400 quaternion-real self-dual Gaussian matrices
`H = [[A, B], [-conj(B), conj(A)]]` with `A` Hermitian and `B` antisymmetric,
built at `2n = 400` (exactly Hermitian and exactly self-dual against
`H = Z H^T Z^-1`, Kramers pairs split by at most `3.2e-14`), give
`0.673465 +/- 0.001012` over `3.9e4` ratios from the central half of each
spectrum, which is `0.9` standard errors from the stored value and `2.7` from the
surmise. The table stores `0.6744`.

`"coe"`, `"cue"` and `"cse"` resolve onto `"goe"`, `"gue"` and `"gse"`: the ratio
distribution depends only on the Dyson index. The `Ensemble` literal and the alias
table are shared with `rmt_reference` through the private `spectral._ensembles`
module, because `long_range` already imports `levels` and the label is now needed
on both sides of that edge.

The measured agreement is pinned by a test rather than asserted on faith: 200 Haar
unitaries of dimension 64 give a run mean of `0.6005 +/- 0.0024` over 25 seeds
against `0.5996`, and larger samples converge on the same constant (40 members at
`N=256` give `0.5996 +/- 0.0031`, 20 at `N=512` give `0.5997 +/- 0.0023`).

#### A symplectic spectrum must have its Kramers doublets removed first

Every beta=4 number above describes the *distinct* levels, one per Kramers
doublet, and the suite generates its own circular symplectic ensemble to check it
(`U = V^D V` with `V` Haar on `U(2n)` and `X^D = J X^T J^-1`; a self-dual unitary
has doubly degenerate eigenvalues, reproduced here to 1e-14).

Leaving the doublets in does not bias `<r>`, it destroys it. Half the circular gaps
are then exactly zero, so every adjacent-gap ratio has a vanishing numerator: the
measured mean is `1.3e-14` instead of `0.6744` under `degeneracy="zero"`, and under
the default `degeneracy="drop"` policy *every* ratio is discarded and the result is
an empty array. `prepare_eigenphases` does flag it — the degeneracy diagnostic
fires even at `degeneracy_tolerance=0.0` — and the test asserts all three
behaviours so that the failure mode stays loud.

### The gap-ratio *distribution*, not only its mean

`mean_gap_ratio_reference` publishes `<r>` and nothing else, which is not enough to
draw the figure that modern localization and thermalization papers actually show:
the histogram of `r` against a reference curve. `rmt_reference` therefore gained
`statistic="gap_ratio_distribution"`, evaluating the 3x3 surmise of Atas,
Bogomolny, Giraud and Roux, Phys. Rev. Lett. 110, 084101 (2013),

```text
P_beta(r) = (r + r^2)^beta / (Z_beta * (1 + r + r^2)^(1 + 3*beta/2))
```

with `beta = 1` for GOE/COE, `beta = 2` for GUE/CUE and `beta = 4` for GSE/CSE
(`Z_4 = 4*pi/(729*sqrt(3))`). Like the spacing reference this is a **surmise**:
`reference_type` reads `"atas_surmise_3x3"`, and it is not a tool for resolving
small departures from Wigner-Dyson statistics.

#### The folded convention is the default, because it is the one that is measured

`adjacent_gap_ratios` returns `min(s_i,s_j)/max(s_i,s_j)`, which lives in `[0, 1]`,
and `mean_gap_ratio_reference` publishes the mean of exactly that. So the default
(`folded=True`) is the density of `r_tilde = min(r, 1/r)` on `[0, 1]`, and it is
zero above `r = 1` — returned as zero rather than rejected so that a histogram grid
may run slightly past the support.

Both densities satisfy `P(1/r)/r^2 = P(r)`, so the two branches of the fold
coincide and folding is exactly a factor of two below `r = 1`. `folded=False`
returns the unfolded density on `(0, inf)` for callers who compute
`r = s_{n+1}/s_n` themselves. Shipping only the unfolded curve was rejected: it
differs from the measurable one by a silent factor of two on the interval where
every histogram lives, and the unfolded first moment diverges for Poisson, so the
self-consistency check below would not exist. `folded` is rejected for every other
statistic rather than ignored, the same way `dimension` is.

#### One integral fixes both `Z_beta` and the shape

`Z_beta` is not taken on trust. The implementation carries the closed forms
`Z_1 = 8/27`, `Z_2 = 4*pi/(81*sqrt(3))` and `Z_4 = 4*pi/(729*sqrt(3))`, and a test
recovers them by adaptive quadrature of the *unnormalized* density over `(0, inf)`,
agreeing to better than 1e-12 relative (3.3e-16 for beta=4). The decisive check, though, is the first moment: because the folded
curve must both integrate to one and reproduce
`mean_gap_ratio_reference(ensemble, surmise=True)`, the normalization and the
exponent structure are pinned simultaneously. Measured on a two-million-point grid,
with mass `1.000000000000` for every ensemble:

| ensemble | `int_0^1 r P(r) dr` | `mean_gap_ratio_reference(surmise=True)` | difference |
| --- | --- | --- | --- |
| Poisson | `0.3862943611` | `2*ln(2)-1 = 0.3862943611` | 4e-14 |
| GOE/COE | `0.5358983849` | `0.5359` | 1.6e-06 |
| GUE/CUE | `0.6026577908` | `0.6027` | 4.2e-05 |
| GSE/CSE | `0.6761683102` | `0.6762` | 3.2e-05 |

The beta=1 and beta=2 rows are literature constants quoted to four decimals, so the
agreement is at the limit of what the published numbers can say; the tolerance in
the test is `1e-4` for that reason and not because the curve is uncertain. The
beta=4 row runs the other way: the table entry was *produced* by this integral, so
for GSE the check is that the stored value is still the rounded curve.

#### Poisson is *not* the `beta -> 0` limit of the surmise

This is stated the other way round often enough to be worth a paragraph. Setting
`beta = 0` in the formula above leaves `1/(1 + r + r^2)`, whose normalization on
`(0, inf)` is `Z_0 = 2*pi/(3*sqrt(3)) = 1.2091996` and whose folded mean is
`0.408545` — 5.8% above the exact Poisson `2*ln(2) - 1 = 0.386294`, and 200 times
outside the `+/-1e-4` that `mean_gap_ratio_reference` claims for its entries. The
ratio density of independent exponential gaps is exactly `P(r) = 1/(1+r)^2`
(`2/(1+r)^2` folded), which is what `ensemble="poisson"` returns;
`reference_type` reads `"exact_poisson_ratio_density"` so that the two grades of
reference are distinguishable programmatically. A test pins the 0.022 gap between
the two so that nobody "simplifies" the implementation by taking the limit.

#### The measured comparison is noise-bound, and says so

80 circular-ensemble members of dimension 128 give 10240 ratios, so a width-0.05
density bin near the peak carries about 0.05 of counting noise. The maximum
deviation over 20 bins runs 0.047 to 0.115 across four seeds, i.e. 0.9 to 2.6 times
that per-bin sigma — consistent with the ~2.2 expected maximum of 20 standard
normals. The comparison is therefore dominated by counting noise and cannot resolve
the surmise-versus-exact discrepancy at all; the library does not ship the exact
ratio density. What makes the test discriminating is that the same histogram misses
the *other* Dyson index by at least 0.42 and the Poisson curve by at least 1.71, so
a swapped `beta`, a missing factor of two from the fold, or a wrong `Z_beta` cannot
pass.

`spacing_distribution` normalizes spacings to unit empirical mean before
returning histogram values and bin edges. With `density=True` the counts are
divided by the **total** number of spacings, never by the number that landed
inside `value_range`. The values are therefore a probability density of the full
sample: the integral is one only when `value_range` covers every spacing, and
equals `included_count / sample_count` otherwise. Both counts are in
`metadata.parameters`, and a non-zero `excluded_count` attaches a
`spacings-outside-range` diagnostic.

Renormalizing on the in-range count instead — the earlier behaviour — made every
histogram integrate to one over whatever range was requested. A `(0, 1)` window
on a 400-level CUE spectrum was lifted by roughly 1.85x, so overlaying it on an
unclipped Wigner surmise showed a large disagreement that was purely an artefact
of the normalization.

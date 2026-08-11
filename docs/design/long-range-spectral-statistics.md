# Long-range spectral statistics and RMT references

Status: implemented baseline for KEN-125  
Last reviewed: 2026-08-09

All functions consume `PreparedEigenphases` or `UnfoldedSpectrum` from the local
level-statistics API. Levels have mean spacing one and a circular period `N`.
`SpectralCurve` returns the x-axis, values, estimator uncertainty, estimator
variance, and metadata as plot-ready read-only arrays; plotting remains outside
the numerical core.

## Spectral form factor

For Heisenberg-normalized time `tau=t/t_H`,

```text
K(tau) = |sum_n w_n exp(2 pi i tau x_n)|^2 / sum_n w_n^2.
```

The `none` and `hann` windows use the same `sum(w^2)` normalization, giving a
unit plateau for uncorrelated phases. With `connected=True`, the Fourier amplitude
of the continuous uniform density over the same finite spectral span and window
is subtracted before squaring; this removes the disconnected peak without adding
a discrete-grid recurrence. Seeded
bootstrap resampling returns pointwise estimator variance and its square-root
uncertainty.

The normalization, the window shape, and the connected subtraction are pinned
together by one exact value. For `N=64` levels at the integers and `tau=1` every
`sinc` in the continuous background lands on a non-zero integer, so the
subtraction vanishes and `K(1) = (sum w)^2 / sum(w^2)`. With `np.hanning(64)`,
`sum(w) = 31.5` and `sum(w^2) = 23.625`, so `K_hann(1) = 42` and
`K_none(1) = 64`, both exact. Normalizing by `sum(w)` would give `31.5`.

That value has a blind spot, and it needs a second test rather than a caveat:
`sinc(N*tau) = 0` at every `tau` where `N*tau` is a non-zero integer, so *any*
prefactor on the background term reproduces it. Subtracting `sinc(N tau)` instead
of `N sinc(N tau)` is invisible on the integer grid and turns the exact `K(0) = 0`
of an equally spaced spectrum into `(N-1)^2/N = 62.0`. The suite therefore also
evaluates the connected estimator at `tau=0` and at the first half-integer point
`tau = 1/(2N)`, where the background is largest away from the origin.

### `bootstrap` is a resampling diagnostic, not an error bar on `K(tau)`

`K(tau)` is a coherent sum over the whole spectrum and **does not self-average**:
its relative fluctuation from one spectrum to the next stays of order one at any
`N`, and only averaging `K` over an ensemble of spectra reduces it. Resampling
levels with replacement destroys the level-level correlations that fix the value
of `K`, so the bootstrap spread is not an estimate of that fluctuation.

Calibrated against the true scatter of 150 Haar-CUE spectra at `N=128`
(`connected=True`, `window="none"`, 200 resamples), the bootstrap standard
deviation comes out

| `tau` | true realization std | bootstrap std | ratio |
| --- | --- | --- | --- |
| 0.3 | 0.326 | 1.222 | **3.7x** |
| 1.0 | 1.110 | 1.658 | 1.5x |
| 1.5 | 1.053 | 1.658 | 1.6x |

The feature is kept rather than removed, because it is a legitimate resampling
diagnostic of one spectrum and it errs *high* — it cannot manufacture a spurious
detection of non-universality. But it is wide enough to hide a real departure
from RMT, so `metadata.parameters["error_semantics"]` states in machine-readable
form that these arrays are not realization scatter, and the docstring repeats the
numbers above. There is no known cheap resampling scheme that recovers the
ensemble scatter of a coherent sum from a single spectrum; the honest answer is
to average over an ensemble and take the scatter of the members, which is what
`examples/rmt_comparison.py` does.

`bootstrap=2` is used in a test to pin `ddof=1`: with two replicates the variance
must be `(x1-x2)^2/2`, and `ddof=0` halves every bootstrap variance in the
library without changing anything else about the result.

## Number variance

The implementation counts levels in half-open windows `[origin, origin+L)` and
returns

```text
Sigma^2(L) = mean_origin((n(origin,L) - L)^2).
```

Origins sit on the **half-shifted** uniform circular grid
`origin_k = (k + 0.5) * N / samples`, which is what `metadata` reports under
`window_origin`. A grid starting at zero coincides with the level positions
whenever `samples` is a multiple of the level count — the common case, since both
the default `samples` and typical Hilbert-space dimensions are powers of two —
and `searchsorted(..., "left")` then breaks those ties by excluding the level
sitting exactly on the left edge, biasing every count low by one. On an equally
spaced spectrum of 32 levels with `samples=128` this reported
`Sigma^2 = 0.0625` at integer `L`, where the exact answer is `0`. The half shift
keeps every origin strictly between two levels, and the estimator then reproduces
the closed form for an equally spaced spectrum,

```text
Sigma^2(L) = frac(L) * (1 - frac(L)),
```

exactly, for any `samples` whose origin grid resolves `frac(L)`.

The default reports the finite circular-spectrum result. The explicit
`finite_size_correction=True` option multiplies by `1/(1-L/N)` and therefore
requires `L<N`.

### The uncertainty is an independent-window count, not an origin count

This section previously said that the error "is governed by the origin count,
not by the number of levels", so that `sqrt((L + 2*L^2)/samples)` and "only
`samples` buys accuracy at large `L`". **That is wrong**, and it was actively
dangerous advice: it is only true for windows that do not overlap.

Origins are dense on *one* spectrum. A window of length `L` shares almost all of
its levels with the window one grid step away, so the `samples` windows are
`samples` correlated readings of the same data, not `samples` measurements.
Dividing the spread by `samples` therefore did two bad things at once:

- it understated the realization-to-realization scatter of `Sigma^2(L)` by a
  factor of 2 to 4 (measured against 90 Haar-CUE spectra at `N=256`,
  `samples=2048`: reported/true = 0.47 at `L=2`, 0.38 at `L=5`, 0.27 at
  `L=10`), and
- on a **fixed** spectrum the error bar shrank without limit as `samples` grew,
  while the value itself stopped moving. At `N=256`, `L=10` the estimate settled
  on 0.544 while the reported uncertainty fell from 0.035 at `samples=512` to
  0.0021 at `samples=131072`, putting the RMT reference 0.5793 an apparent 16
  sigma away.

At most `floor(N/L)` windows of length `L` fit side by side on `N` levels, so
the implementation now reports a **batch-means standard error over
`effective_samples = min(samples, floor(N/L))` contiguous arcs** of the origin
circle: the origins inside an arc are averaged first, and the scatter is taken
across arcs. `metadata.parameters` carries `effective_samples` (one entry per
requested length) and an `error_semantics` string.

Averaging inside an arc is what makes this work rather than the cruder
`sqrt(var_of_one_window / (N/L))`: it removes the fast noise and leaves the
long-wavelength fluctuation that actually differs between realizations. Measured
against the same CUE ensembles, the calibration ratio reported/true is

| `L` | `N=64` | `N=128` | `N=256` |
| --- | --- | --- | --- |
| 2 | 0.87 | 0.85 | 0.82 |
| 5 | 0.76 | 0.71 | 0.87 |
| 10 | 0.70 | 0.70 | 0.77 |

against 0.25-0.50 for the old estimator on the same spectra (`N=256`: 0.47, 0.38,
0.27). The remaining ~20% shortfall is expected and is on the conservative side
of nothing: neighbouring arcs of a spectrally rigid sequence are anticorrelated,
which the batch-means estimator ignores.

`bootstrap` resamples the **arcs**, not the origins. An origin bootstrap
reproduces the `sqrt(var/samples)` understatement exactly, because resampling
overlapping windows asserts precisely the independence that does not hold; the
block bootstrap agrees with the analytic batch-means error to within a few
percent (0.84/0.70/0.66 against 0.85/0.71/0.70 at `N=128`, 400 resamples).

When `L > N/2` only one window fits, `effective_samples` is 1, and no scatter
can be extracted from a single spectrum at all. The function then emits a
`NumericalWarning` with the stable code `number-variance-unresolved-error` and
reports the spread of that single window, which overstates rather than
understates the error.

## Spectral rigidity `Delta_3(L)`

`spectral_rigidity` returns the Dyson-Mehta statistic

```text
Delta_3(L) = < min_{A,B} (1/L) * int_{E0}^{E0+L} (N(E) - A*E - B)^2 dE >_{E0},
```

the mean-square residual of the best straight-line fit to the staircase counting
function inside a window of length `L`, averaged over window origins. It is the
companion of the number variance that the level-statistics literature has plotted
since Bohigas-Giannoni-Schmit, and every convention is deliberately identical to
`number_variance`: `PreparedEigenphases` or `UnfoldedSpectrum` in, circular
windows of period `N`, origins on the half-shifted grid `(k+0.5)*N/samples`,
half-open counting interval, batch-means error over
`effective_samples = min(samples, floor(N/L))` arcs, and the same
`error_semantics` string convention in `metadata.parameters`. Sharing the
conventions is the point: the two curves are read off the same plot and any
difference between them would be attributed to the physics.

### The window integral is closed-form, and it is checked against a naive one

`N(E)` is a staircase, so the least-squares residual inside a window is a finite
expression rather than a quadrature. Projecting the staircase onto `span{1, x}`
and writing `s_k = x_k/L` for the `n` in-window levels measured from the origin,

```text
Delta_3 = n^2 - sum_k (2k-1) s_k - (n - sum_k s_k)^2 - 3 * (sum_k s_k (1-s_k))^2.
```

The implementation evaluates the three sums as differences of prefix sums over the
doubled level array, so one length costs one `searchsorted` and a handful of
vectorized operations rather than a loop over `samples` windows.

The closed form is validated against a naive independent implementation that
integrates each staircase segment exactly and solves the 2x2 normal equations
directly — no orthogonal basis, no prefix sums, no shared algebra:

| what is compared | agreement |
| --- | --- |
| per-window formula vs naive exact integration | 1.5e-15 absolute |
| the vectorized function vs the same naive average | 5.0e-13 absolute (worst of `L=0.7..11`, `N=24`) |
| 2e6-point Riemann sum of the fitted residual | 2.5e-7, the Riemann sum's own error |

The gap between the first two rows is cancellation inside the prefix sums, which
is why the number is reported rather than rounded to "exact".

### The equally spaced spectrum has a closed form, and it is not `1/12`

For levels at the integers the deviation of the staircase from its best straight
line is the sawtooth `-frac(E)`, whose variance is `1/12`. That is *not* the
answer, because for integer `L` the fitted slope is not zero: it is set by
`c(phi) = 1/12 + (phi^2-phi)/2` with `phi` the fractional part of the origin, and
removing the line subtracts `12*c(phi)^2/L^2`. Averaging `c^2` over uniform `phi`
gives `1/720`, so

```text
Delta_3(L) = 1/12 - 1/(60*L^2)   (integer L, unit lattice),
```

which is `1/15` at `L=1`, `19/240` at `L=2`, and `1/12` only in the limit. Writing
the test against a flat `1/12` would have failed at exactly the small `L` where
the statistic is most used.

The half-shifted origin grid is a *midpoint rule* for the average over `phi`, so
the measurement converges on that closed form as `(samples/N)^-2`. At `N=64` and
`L=1` the error is 2.03e-5 at `samples=4096`, 7.95e-8 at `samples=65536` and
4.97e-9 at `samples=262144` — a factor of 256 per factor of 16 in `samples`, which
the test asserts so that the agreement cannot be a coincidence.

### The error bar was recalibrated, not inherited

`number_variance` and `spectral_rigidity` share `_batch_means_variance`, so the
two cannot drift into reporting different things under the same name, but sharing
the code does not transfer the calibration. Measured against the true scatter of
80 Haar-CUE spectra at `N=128` with `samples=1024`, reported/true is

| `L` | ratio |
| --- | --- |
| 2 | 1.00 |
| 5 | 0.85 |
| 10 | 0.79 |

and over four independent ensembles of 80 the ratios span 0.74 to 1.00 — inside
the 0.5-2 band this library requires, erring low for the same reason as the number
variance (neighbouring arcs of a rigid spectrum are anticorrelated). `bootstrap`
resamples arcs, not origins. When `L > N/2` a single window is all that fits, the
function emits a `NumericalWarning` with the stable code
`spectral-rigidity-unresolved-error` and reports the spread of that one window.

### The `Delta_3` reference comes from the Mehta relation, not from an asymptotic form

`rmt_reference("spectral_rigidity", ...)` does not evaluate a published
large-`L` formula. It integrates the exact Mehta relation

```text
Delta_3(L) = (2/L^4) * int_0^L (L^3 - 2 L^2 r + r^3) * Sigma^2(r) dr
```

against the *same* two-level cluster function `Y_2` that the number-variance
reference uses, so the two curves cannot disagree about the ensemble. Substituting
`Sigma^2(r) = r - 2 int_0^r (r-s) Y_2(s) ds` and doing the `r` integral
analytically collapses the nested integral to a single quadrature:

```text
Delta_3(L) = L/15 - (4/L^4) * int_0^L Y_2(s) * G(s,L) ds
G(s,L)     = int_s^L (L^3 - 2 L^2 r + r^3)(r-s) dr
           = L^5/30 - L^4 s/4 + L^3 s^2/2 - L^2 s^3/3 + s^5/20
```

with `G(L,L) = 0`, `G(0,L) = L^5/30`, and `int_0^L G ds = 0` identically — the last
identity being why `Delta_3 -> L/15` as `L -> 0` for every ensemble. The quadrature
is a 24-node Gauss-Legendre rule on each unit-length panel, one panel per
oscillation period of `Y_2` (per *two* periods for beta=4, whose kernel oscillates
twice as fast); doubling the order or quartering the panels moves the result by
1e-15 at `L=5` and 3e-13 at `L=3000` for every index. Panels are evaluated in
chunks so that a large `L` costs time rather than memory.

The **number variance uses the same panel rule**, through a shared
`_cluster_integral(ensemble, length, dimension, weight)` that differs between the
two statistics only in the polynomial weight (`L - s` for `Sigma^2`, `G(s,L)` for
`Delta_3`). It previously called `scipy.integrate.quad` on `[0, L]`, which is
adaptive and silently loses an oscillatory integrand once the interval holds a few
hundred periods: at `L=1024` it returned `Sigma^2_GUE = 0.9078` against the correct
`1.0483`, and `Sigma^2_GSE = -65.4` against `0.6218`, in both cases after an
`IntegrationWarning` that a caller filtering warnings would never see. beta=4 is
what forced the issue — its `Y_2` decays only like `1/r` where beta=1,2 decay like
`1/r^2` — but the bug was already there for beta=2. The panel rule agrees with
`quad` to 4e-16 wherever `quad` was converging, including on the finite-`N` CUE
kernel, so no reference value below `L=64` moves.

Four independent checks, all measured:

1. **Poisson is exact.** `Y_2 = 0`, so the relation reduces to
   `(2/L^4)(L^5/30) = L/15` with no quadrature left. The implementation reproduces
   `L/15` bit for bit over `L` from 0 to 100, which pins the polynomial weight, the
   `2/L^4` prefactor, and the reduction above all at once. This is the single best
   check available and the test asserts it at `rtol=1e-6` *and* by exact equality.
2. **The reduction equals the nested integral.** Performing the double integral
   instead, calling the public `"number_variance"` reference for every
   `Sigma^2(r)`, agrees to 1.8e-15 relative at worst for GOE, GUE and GSE over
   `L = 0.5..20`. Running it for beta=4 as well is what ties the symplectic
   rigidity to the symplectic number variance: both are the same `Y_2` under
   different weights, so a doubling convention applied in one place and not the
   other would show up here even though every check internal to one statistic
   would still pass.
3. **The large-`L` slopes are the analytic logarithms.** Regressing on `ln L` over
   `L` in `[512, 8192]` gives

   | beta | measured slope | analytic `1/(beta*pi^2)` | ratio |
   | --- | --- | --- | --- |
   | 1 | 0.10126081 | 0.10132118 | 0.99940 |
   | 2 | 0.05066059 | 0.05066059 | 1.0000000 |
   | 4 | 0.02533787 | 0.02533030 | 1.00030 |

   The beta=1 and beta=4 offsets are `1/L` corrections and not wrong slopes: adding
   a `c/L` term to the same fit gives ratios 0.9999988 and 0.9999997. The series is
   `1/(beta*pi^2)` — one over beta, **not** one over two beta, which for beta=4
   would give `1/(8*pi^2) = 0.01266515`, a clean factor of two away from what the
   curve does. The slope *ratios* are 2 and 4 exactly and are what the suite
   asserts most tightly, because they carry no `1/L` ambiguity.
4. **The constants were read off, not remembered.** Pinning the slope to the
   analytic value leaves

   ```text
   Delta_3_GOE(L) -> (1/pi^2)     * ln L - 0.006950
   Delta_3_GUE(L) -> (1/(2 pi^2)) * ln L + 0.0590243
   Delta_3_GSE(L) -> (1/(4 pi^2)) * ln L + 0.0783197
   ```

   The GUE constant is stable to eight digits from `L=512` upwards; the GOE and GSE
   ones still drift by about `1.2e-5` and `1.5e-6` per octave and are quoted as
   `1/L` Richardson extrapolations of `L=4096` (`-0.0069267`, `0.0783167`) and
   `L=8192` (`-0.0069391`, `0.0783182`). These are the values this implementation
   produces, and the docstring says so rather than citing a table.

   Below `L=2` the three curves are not ordered by beta at all: every ensemble
   starts from the common `Delta_3 -> L/15`, and at `L=0.5` beta=4 is the largest
   of the three (`0.03319691` against `0.03282211` and `0.03242198`).

The often-quoted "GOE is twice GUE" is only a statement about the slopes. The
ratio of the curves reaches 2 logarithmically slowly — 1.342 at `L=10`, 1.576 at
`L=100`, 1.695 at `L=1000`, 1.758 at `L=8192` — so the suite asserts that the ratio
is monotone, below 2, and past 1.75 by `L=8192`, and puts the sharp requirement on
the slope ratio instead.

`dimension` is **rejected** for this statistic rather than selecting a finite-`N`
CUE kernel as it does for the number variance. The Mehta relation is a bulk
identity for a translation-invariant spectrum; feeding it a finite circular kernel
would return a number that looks like a finite-size correction and is not one.
`reference_type` reads `"mehta_integral_of_number_variance"` and `finite_size`
reads `"not applicable"`.

## RMT references

`rmt_reference` provides analytic connected form factors for Poisson, GOE, GUE,
CUE and GSE. It provides Poisson number variance directly and evaluates the
GOE/GUE/GSE bulk cluster-function integrals numerically. When CUE dimension is
supplied, number variance uses the finite circular kernel
`[sin(pi*s)/(N*sin(pi*s/N))]^2`; otherwise it uses the GUE bulk limit. Exact
analytic references return zero uncertainty and variance arrays.

### Circular and Gaussian ensembles coincide in the bulk

Bulk spectral correlations depend only on the Dyson index `beta`, so the circular
ensembles reproduce their Gaussian counterparts level for level:
`COE == GOE` (`beta=1`), `CUE == GUE` (`beta=2`) and `CSE == GSE` (`beta=4`).
Quantum maps are unitary and their natural reference is therefore a circular
ensemble, so `"coe"` is accepted as an alias of `"goe"`; a time-reversal-symmetric
map (the cat and baker maps, the default kicked rotor) no longer has to know that
it should ask for `"goe"`. `metadata.parameters` keeps both names: `ensemble` is
the canonical one actually evaluated and `requested_ensemble` is what the caller
passed, with `ensemble_alias_applied` recording whether they differ. `"cue"`
remains distinct from `"gue"` only in that supplying `dimension` selects the
finite-`N` circular kernel above.

`"cse"` is accepted as an alias of `"gse"` on the same reasoning that produced
`"coe"`, and it collapses *outright*: there is no finite-`N` symplectic kernel
here, so unlike `"cue"`/`"gue"` the two labels can never differ in any argument.
The alternative — shipping `"gse"` alone — was rejected because it would leave the
alias table asymmetric for no reason a caller could infer: a Floquet operator with
half-integer spin and time reversal is in the circular symplectic ensemble, and it
should be able to name itself, exactly as a time-reversal-symmetric one can name
`"coe"`.

### beta=4 exists as a reference even though no model here reaches it

**No model in this library produces GSE statistics.** beta=4 requires an
antiunitary symmetry squaring to `-1` — half-integer spin with time reversal — and
the cat map, the baker map and the kicked rotor are all spinless, so they land in
beta=1 or beta=2. Adding `"gse"` therefore adds a reference curve with no internal
comparison partner, which is unusual enough to record the reasoning:
`rmt_reference` compares against a spectrum the **caller** supplies, and what the
caller can bring is independent of what the library can simulate. The alternative
reading — that a reference curve should only exist if a bundled model produces it —
would make the function a property of the model catalogue rather than of random
matrix theory.

The cost of the decision is that the beta=4 tests have to generate their own
ensemble. They build Dyson's circular symplectic ensemble directly, as `U = V^D V`
with `V` Haar on `U(2n)` and `X^D = J X^T J^-1` the quaternion dual: `(XY)^D =
Y^D X^D` and `(X^D)^D = X` make `U` self-dual, and a self-dual unitary is Kramers
degenerate. Everything is seeded and sized for CI (`2N=128`, 40–60 members).

**Every beta=4 curve here describes the *distinct* levels, one per Kramers
doublet.** This is not a convention that can be papered over: the `sin(2 pi r)`
inside `Y_2^GSE` *is* the statement that the kernel is built on half as many levels
as the matrix has eigenvalues. Feeding a statistic the doubled spectrum does not
add noise, it replaces the measurement — half the circular gaps become zero, and
the measured `<r>` falls from `0.6744` to `1.3e-14` (or, under the default
`degeneracy="drop"` policy, every ratio is discarded and nothing is left). The
suite asserts both halves of that.

`dimension` is rejected for `"gse"`/`"cse"` on every statistic, not only on the
three that reject it for everyone. The one finite-`N` kernel implemented is the
circular unitary one; there is no symplectic counterpart to substitute, so a
dimension would be accepted and ignored, which is what the rest of this API
refuses to do.

#### `K_GSE(tau)` has a real pole at the Heisenberg time, and it is rejected

The three connected form factors are

```text
beta=1  2*tau - tau*ln(1+2*tau)      (tau<=1);  2 - tau*ln((2tau+1)/(2tau-1))
beta=2  min(tau, 1)
beta=4  tau/2 - (tau/4)*ln|1-tau|    (tau<=2);  1 above
```

Each was verified here against the Fourier transform of the `Y_2` this module
implements, `K(tau) = 1 - 2 int_0^inf Y_2(r) cos(2 pi tau r) dr` by oscillatory
quadrature, agreeing to eight digits; a test repeats that for beta=4 so the closed
form and the cluster function cannot drift apart. The beta=4 curve is the odd one
out three times over: it reaches the plateau at `tau=2` rather than `tau=1`, it
*overshoots* it on the way (`K(1.1) = 1.1832`), and it diverges logarithmically at
`tau=1`.

That divergence is physical — it is the Kramers doublets piling up at the
Heisenberg time of the doublet-resolved spectrum — so it is neither clipped nor
returned as `inf`. `SpectralCurve` refuses non-finite values, so an `inf` would
surface as a message about curve values rather than about the ensemble; the
reference raises a `ValidationError` naming `tau=1` instead. The cost is that a
grid like `np.linspace(0, 2, 5)` is refused, which is a real inconvenience and is
why the message says to step over `tau=1`. The curve is perfectly ordinary
arbitrarily close by (4.0 at `tau = 1 +/- 1e-6`).

Invalid `statistic` and `ensemble` values are rejected with a message that lists
the accepted values, enumerated from the `Literal` aliases so that the message
cannot drift from the implementation.

### `statistic="spacing_distribution"` returns a surmise, not the exact density

`rmt_reference("spacing_distribution", ensemble, s)` evaluates the Wigner surmise
with `s` in units of the mean spacing:

```text
P_Poisson(s) = exp(-s)
P_beta(s)    = a_beta * s^beta * exp(-b_beta * s^2)

P_GOE/COE(s) = (pi*s/2)        * exp(-pi*s^2/4)
P_GUE/CUE(s) = (32*s^2/pi^2)   * exp(-4*s^2/pi)
P_GSE/CSE(s) = (2^18*s^4/(3^6*pi^3)) * exp(-64*s^2/(9*pi))
```

This closes the most conspicuous gap in the reference surface: `P(s)` against
Wigner-Dyson is the standard quantum-chaos figure, and until now the notebook and
the tutorial each rewrote these three expressions by hand.

All four are normalized twice over — `integral P ds = 1` and `integral s P ds = 1`
— which is what makes them comparable to a `spacing_distribution` histogram, whose
spacings are rescaled to unit *empirical* mean. Unit mean forces
`sqrt(b) = Gamma((beta+2)/2)/Gamma((beta+1)/2)` and unit mass then forces
`a = 2*b^((beta+1)/2)/Gamma((beta+1)/2)`, so the rate and the prefactor are a
single choice: a prefactor edited without its rate breaks both integrals. The
implementation carries the closed forms in one `_WIGNER_SURMISE` table keyed by
canonical ensemble rather than as a branch per index, and a test integrates each
curve on a fine grid to `rtol=1e-6` rather than trusting the algebra. The modes
move right and up with beta — `0.5642`, `0.8862`, `0.9400` at heights `0.7602`,
`0.9368`, `1.2253` — which is what makes a measured histogram able to tell them
apart at all.

The name is load-bearing: these are the exact spacing densities of a **2x2**
matrix, carried over to the bulk by the usual abuse, not the large-`N`
Gaudin-Mehta distributions. They are what textbooks plot and are correct to a
couple of percent, but they are not a tool for measuring small departures from
Wigner-Dyson statistics. The library does not ship the exact densities, and
`metadata.parameters["reference_type"]` says `"wigner_surmise_2x2"` instead of the
`"analytic"` that the form factor and number variance report, so a caller can tell
the two grades of reference apart programmatically.

Two argument decisions:

- **`dimension` is rejected, not ignored.** The surmise contains no `N`, so there
  is no finite-size correction to apply; accepting the argument silently would
  imply one had been. `finite_size` reads `"not applicable"`.
- **`s < 0` is rejected**, by the same `_nonnegative_1d` check that guards `tau`
  and `L`. Continuing `P` as zero below the origin would be defensible, but a
  negative spacing is a caller error worth surfacing, and consistency across the
  three statistics is worth more than the convenience.

### Published type aliases

`Statistic`, `Ensemble`, and `Window` are exported from `chaos_numerics.spectral`.
They appear in the signatures of `rmt_reference` and `spectral_form_factor`, and a
package that ships `py.typed` has to let callers annotate their own wrappers with
the same types — `def plot(window: Window)` was previously unwritable without
reaching into a private module. `Ensemble` lives in `spectral._ensembles` so that
`levels.mean_gap_ratio_reference` can share it without an import cycle.

Exporting them constrains later changes, which is the point of recording it here:
`Window` and `Statistic` may only grow new members, and removing or renaming a
member of any of the three is a breaking change. `Statistic` has grown twice under
that rule, to `"spectral_rigidity"` and `"gap_ratio_distribution"`, and `Ensemble`
once, to `"gse"`/`"cse"`; the error-message test that enumerates the candidates
picks additions up automatically because the message is built from
`get_args(...)`. The new labels are appended rather than inserted so that the
message keeps its old prefix.

The canonical-ensemble label that alias resolution produces is deliberately **not**
exported; it is an artefact of how the curves are evaluated. That decision paid
for itself here: adding `"gse"` grew `CanonicalEnsemble` as well as `Ensemble`, and
a published canonical alias would have made that a breaking change.

### Every persisted array must appear in the descriptor

`array_payload()` and `metadata_payload()` are two views of the same set of arrays:
the first carries the numbers for the archive, the second describes their shape and
dtype for the JSON side, and the storage layer matches the two sets when it computes
per-array digests. `add_convergence_history` adds a `"convergence_history"` entry to
the payload whenever `metadata.convergence.history` is set, so it has to be applied
to the descriptor mapping as well.

All four spectral containers — `PreparedEigenphases`, `UnfoldedSpectrum`,
`SpacingDistributionResult`, `SpectralCurve` — applied it on the payload side only.
Saving any of them with a convergence history attached therefore failed with a bare
`KeyError` from the digest step instead of a diagnosable error, and the existing
round-trip coverage missed it because those tests build their examples from metadata
that has no history at all. The fix is one `add_convergence_history(..., copy=False)`
call per `metadata_payload`, matching `core.results`; the guard is a parametrized
test that constructs each container with a history on purpose and asserts
`set(array_payload()) <= set(metadata_payload()["arrays"])` plus shape/dtype
agreement for every key. A container that grows a new array and forgets the
descriptor now fails immediately.

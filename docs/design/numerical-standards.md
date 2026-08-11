# Numerical Accuracy, Validation, and Performance Standards

Status: accepted baseline for KEN-110  
Applies to: Chaos Numerics Library v0.1  
Last reviewed: 2026-08-04

## 1. Purpose

This document defines the numerical quality contract for v0.1. It specifies
canonical precision, comparison rules, invariant checks, reference
implementations, convergence reporting, stochastic reproducibility, and
performance-regression gates. It complements:

- `docs/product/mvp-scope.md`, which defines what v0.1 contains; and
- `docs/design/public-api.md`, which defines names, shapes, dtypes, and module
  ownership.

Thresholds here are test defaults, not claims that every physical problem has the
same conditioning. A function may expose a user tolerance, but its result must
record the effective tolerance and achieved error or residual.

## 2. Precision and comparison model

### 2.1 Canonical precision

| Data | Computation and output dtype | Policy |
| --- | --- | --- |
| Classical states and real observables | `numpy.float64` | Integer and lower-precision real inputs are promoted at the public boundary. |
| Quantum states/operators and general eigenvalues | `numpy.complex128` | Real inputs may be promoted; imaginary components are preserved. |
| Probabilities, densities, eigenphases, residuals | `numpy.float64` | Negative roundoff is not silently clipped unless the algorithm documents and records that normalization step. |
| Sparse Ulam data | `numpy.float64` | Indices use the SciPy-selected integer width; serialized indices use an explicit dtype. |
| Accumulations | At least the canonical dtype | Pairwise/stable summation is preferred where cancellation or long reductions matter. |

The binary64 machine epsilon is
`eps = numpy.finfo(numpy.float64).eps`, approximately `2.22e-16`. Thresholds
between `1e-14` and `1e-12` allow tens to thousands of rounding operations while
still detecting a wrong sign, normalization, transpose, FFT convention, or
coordinate order. Iterative sparse algorithms receive looser residual thresholds
because solver stopping criteria and conditioning dominate raw roundoff.

v0.1 provides no accuracy guarantee for `float32` or `complex64`. Unsupported
object, Boolean-as-numeric, unsafe complex-to-real, or incompatible shape inputs
raise `ValidationError` rather than being guessed or silently narrowed.

### 2.2 Scalar and array comparison

Unless a row in this document overrides it, compare a computed value `a` with a
reference `b` elementwise using:

```text
abs(a - b) <= atol + rtol * abs(b)
```

| Comparison class | `rtol` | `atol` | Use |
| --- | ---: | ---: | --- |
| Algebraic/small dense | `1e-13` | `1e-14` | Closed-form steps, scalar/batch agreement, exact transforms. |
| Multi-step deterministic | `1e-11` | `1e-13` | Short, non-chaotic accumulation where pointwise comparison is meaningful. |
| Dense vs matrix-free | `5e-12` | `5e-13` | Independently implemented operator actions. |
| Iterative analysis result | quantity-specific | quantity-specific | Use achieved residual/convergence history, not this default. |

Both absolute and relative terms are required: absolute tolerance controls values
near zero, while relative tolerance scales nonzero references. Tests must fail on
NaN or infinity unless non-finite output is the explicitly documented result.

For angles or periodic coordinates, compare the minimum wrapped displacement, not
the raw subtraction. For eigenvectors, compare a phase-invariant residual or
subspace, not elementwise vectors. For degenerate eigenvalues, compare the
invariant subspace using principal angles or projector distance.

### 2.3 Scale-normalized residuals

Norms below are Euclidean/Frobenius norms unless documented otherwise.

```text
eigenpair_residual(A, lambda, v) =
    ||A v - lambda v||_2 /
    max(tiny, (||A||_estimate + |lambda|) ||v||_2)

unitarity_defect(U) = ||U^* U - I||_F / sqrt(N)

symplectic_defect(J) =
    ||J^T Omega J - Omega||_F / max(1, ||Omega||_F)

probability_defect(P) = max_j |sum_i P[i, j] - 1|

projector_distance(V, W) = ||V V* - W W*||_2
```

`||A||_estimate` is the exact matrix norm for small dense tests and a documented,
reproducible norm estimate for matrix-free tests. Residuals, norm estimates, and
effective solver tolerances are stored in result metadata.

## 3. Periodic coordinates and boundary validation

The canonical interval is half-open, `[lower, upper)`, with normalization:

```text
wrapped = lower + mod(x - lower, upper - lower)
```

| Check | Test data | Acceptance |
| --- | --- | --- |
| Domain containment | Values inside, outside, and on both endpoints | `lower <= wrapped < upper`; no terminal upper value survives. |
| Period equivalence | `x` and `x + n * period`, `|n| <= 10^6` | Minimum wrapped difference `<= 64 * eps * max(1, period, abs(x))`. |
| Upper endpoint | `upper`, `nextafter(upper, +inf)`, `nextafter(upper, lower)` | `upper` maps to `lower`; adjacent floats follow the half-open rule without producing an out-of-range index. |
| Internal cut ownership | Exact partition/Baker cuts and adjacent floats | Exact cut belongs to the positive/right cell; adjacent floats land on their respective sides. |
| Index round trip | Every cell plus random cells | `ravel(unravel(i)) == i` exactly; C-order final coordinate changes fastest. |

The `10^6`-period test bound detects unstable modulo implementations while avoiding
an unrealistic promise for inputs so large that binary64 has lost all sub-period
information. Inputs whose magnitude makes the requested periodic resolution
unrepresentable produce `NumericalWarning`; they are not claimed accurate.

## 4. Classical maps and trajectories

### 4.1 Required checks

| Feature | Validation quantity | Acceptance | Rationale |
| --- | --- | ---: | --- |
| Standard/Cat/Baker one-step result | Independent closed-form reference | `rtol=1e-13`, `atol=1e-14` | Only a small fixed number of binary64 operations is involved. |
| Scalar/batch agreement | Batch output vs stacked scalar calls | `rtol=1e-13`, `atol=1e-14` | Detects axis and broadcasting mistakes. |
| Analytic Jacobian | Independent closed form | `rtol=1e-12`, `atol=1e-13` | Allows modest expression-order differences. |
| Jacobian cross-check | Centered finite difference with scale-aware step | `rtol=5e-6`, `atol=5e-8` | Finite differences balance truncation and cancellation and are not a precision reference. |
| Standard-map symplecticity | `symplectic_defect(J)` | `<= 1e-12` per sampled state | Thousands of eps is strict enough to expose a wrong Jacobian. |
| Cat-map area preservation | `abs(det(J) - 1)` | `<= 1e-14`; integer matrix determinant checked exactly | The built-in matrix is exact integer data. |
| Output periodicity | Section 3 rules | All generated wrapped states in bounds | Avoids delayed failures in partitions and plots. |
| Short trajectory | Independent repeated scalar reference, at most 32 steps | `rtol=1e-11`, `atol=1e-13` | Catches history indexing while limiting chaotic amplification. |

Long chaotic trajectories must not be compared point by point: exponential
sensitivity makes that test dependent on operation order and BLAS/compiler
details. Long-run validation instead uses invariants, statistical observables,
Lyapunov convergence, deterministic replay in the same environment, and known
analytic quantities.

### 4.2 Lyapunov analysis

| Check | Acceptance |
| --- | --- |
| Cat-map exponent/spectrum | Absolute error `<= 1e-10` against log-magnitudes of the analytic eigenvalues for a constant Jacobian test. |
| QR orthogonality | `||Q.T @ Q - I||_F / sqrt(d) <= 1e-12` after each recorded reorthogonalization in diagnostic tests. |
| Sum rule for symplectic 2D map | `abs(sum(exponents)) <= 1e-10` for the analytic Cat-map reference; empirical maps use the reported convergence uncertainty. |
| Finite-time history | Final history entry equals the returned estimate within `rtol=1e-14`, `atol=1e-15`. |
| Convergence | Difference between estimates from the final two equal-length windows `<= max(1e-8, 1e-3 * max(abs(estimate), 1e-8))`, unless the caller supplies a stricter rule. |

The default relative convergence target of `1e-3` is a diagnostic for a finite-time
chaotic estimate, not a claim of machine-precision physics. Failure returns the
estimate and history with `converged=False` and emits `ConvergenceWarning`; strict
mode raises `ConvergenceError`.

### 4.3 Correlation, transport, and periodic orbits

| Feature | Reference | Acceptance |
| --- | --- | --- |
| Autocorrelation | Constant, delta, alternating, and directly summed short series | `rtol=1e-12`, `atol=1e-14`; lag-zero normalization exactly documented. |
| Mean-square displacement | Constant velocity and seeded random walk | Deterministic case `rtol=1e-12`; stochastic ensemble mean within `5` estimated standard errors. |
| Local diffusion exponent, central value | Synthetic power laws over a fixed fit interval | Absolute slope error `<= 5e-3`; result records interval, sample count, and fit residual. |
| Local diffusion exponent, reported error | 200 realizations of two known exponents: a unit random walk (`alpha = 1`) and fractional Brownian motion with `H = 0.75` (`alpha = 1.5`) | Ratio of the true estimator spread to the reported uncertainty in `[0.5, 2]`, and at most `20%` of realizations further than two reported errors from the truth. Measured `0.95`-`1.19` and `3.0%`-`11.5%`. |
| Mean-square displacement, reported error | 200-400 realizations of unit random walks of 2000 steps, whose MSD is exactly `<dx**2> = t` | Ratio of the true estimator spread to the reported uncertainty in `[0.5, 2]`, and at most `20%` of realizations further than two reported errors from the truth. Measured `0.98`-`1.13` and `4.5%`-`9.5%` for the batch standard error over lags `1`-`500` at `batch = 16`. The retired time-origin standard error measured `1.05`-`20.87` and `4.8%`-`90.0%` over the same lags and fails at every lag from `5` upwards. |
| Periodic orbit | `||F^period(x) - x||_2` using wrapped displacement | `<= max(user_tol, 1e-10)` for a successful result. |
| Monodromy matrix | Product of independently evaluated Jacobians | `rtol=1e-10`, `atol=1e-12` for low periods covered by v0.1. |
| Stability multipliers | Eigenpair residual of monodromy matrix | `<= 1e-10`. |

Fits with fewer than three valid points, rank deficiency, a non-finite slope, or a
window outside the available data raise `ValidationError`. A valid fit that fails
its requested conditioning/convergence target returns diagnostics and emits
`NumericalWarning`, and so does a fit whose error cannot be calibrated.

The error criterion is a criterion because the obvious estimator fails it. A
least-squares standard error assumes independent residuals, and a
time-origin-averaged MSD is a smooth curve whose adjacent lags are strongly
correlated, so the residuals are tiny and the error bar comes out about `30` times
too small: measured on a unit random walk, the true spread of the exponent was
`0.142` against a reported `0.005`, and `93%` of realizations sat more than two
reported errors from the true value where `5%` was expected. Feeding a batch of
trajectories did not help, because the same correlated-residual fit was applied to
the ensemble-averaged curve. Block jackknife and Newey-West estimators were
measured as well and reached ratios of only `1.5`-`9.9`.

The contract is therefore: with an ensemble the reported uncertainty is the
standard error of the mean across independent trajectories -- over per-curve fits
for the exponent, over per-trajectory curves for the MSD and the autocorrelation
-- and **with a single curve or a single trajectory the uncertainty is `None`**
together with a `NumericalWarning`, because one curve carries no information about
the spread of its own value. Every result records what its error means in
`metadata.parameters["error_semantics"]`.

`mean_square_displacement` used to escape that contract by reporting the scatter
over the time origins of one trajectory, which is the same failure one level down:
the origins of a time-origin-averaged curve are not independent draws. Measured
against the exact `<dx**2> = t`, it understated the true spread by factors of
`1.0`, `2.6`, `8.8` and `20.9` at lags `1`, `10`, `100` and `500`, leaving `4.8%`,
`46%`, `82%` and `90%` of realizations more than two of its sigma from the truth.
The estimator is gone rather than deprecated, and a regression test recomputes it
on the same input to assert that it would still fail, so restoring it fails the
suite. The batch path it was replaced by was already calibrated and is unchanged.

Overflow in the squared or fourth-power accumulation raises `NumericalError`
naming the coordinate magnitude, the representable limit, and the remedy, in place
of leaking a bare `RuntimeWarning` followed by a generic non-finite
`ValidationError`. This matches the Lyapunov overflow policy in section 4.2.

## 5. Partitions and Ulam operators

### 5.1 Partition checks

- `locate`, flat/multi-index conversion, and `cell_bounds` are exhaustive for
  small partitions and property-tested for larger shapes.
- Every seeded sample satisfies its cell's half-open bounds. Samples must not be
  repaired by clipping into the target cell.
- Scalar and batched lookup results agree exactly as integer indices.
- Invalid bounds, zero/negative cell counts, dimension mismatches, and non-finite
  coordinates raise `ValidationError`.

### 5.2 Ulam matrix checks

The v0.1 convention is column-stochastic:
`P[target, source] = Pr(target | source)`.

`P` below is a `UlamMatrix`. Its CSR arrays are reached through `P.matrix`; the
container itself forwards only the operator surface (`shape`, `dtype`, `nnz`,
`matvec`, `toarray()`, `@`).

| Quantity | Closed-system acceptance | Notes |
| --- | ---: | --- |
| Entry non-negativity | `min(P.matrix.data) >= -1e-15` | Count-derived construction should normally be exactly nonnegative; tolerance catches sparse arithmetic roundoff. |
| Probability conservation | `probability_defect(P) <= 1e-12` | About `4.5e3 * eps`; well above summation noise but far below a missing transition. |
| Entry upper bound | `max(P.matrix.data) <= 1 + 1e-15` | A count cannot exceed samples per source cell. |
| Open-system mass accounting | Reported escape probability agrees with `1 - column_sum` within `1e-12` | Open behavior requires explicit `open_system=True`; a closed build treats escape as an error. Column sums no greater than one are enforced by the `UlamMatrix` constructor, so substochasticity is an invariant of the type rather than a per-consumer check. |
| Dense/CSR equality | `rtol=0`, `atol=0` when built from the same count table | Conversion/storage must not change values. |
| Independent dense vs CSR build | `rtol=1e-13`, `atol=1e-15` with identical predetermined samples | Detects orientation/indexing errors without conflating RNG order. |
| Stationary density normalization | `abs(sum(rho) - 1) <= 1e-12` | Density uses the documented column-vector convention. |
| Stationary density negativity | `min(real(rho)) >= -1e-12`; imaginary norm `<= 1e-12` | Larger violations indicate the wrong eigenvector or failed convergence. |
| Invariance | `||P @ rho - rho||_1 <= 1e-10` | Covers sparse eigensolver tolerance and normalization. |
| Leading eigenvalue | `abs(lambda_0 - 1) <= 1e-10` | Required for a closed stochastic matrix after successful convergence. |

Sampling error is not measured by column-sum error: normalization can be exact
while transition probabilities remain noisy. Convergence examples therefore vary
both partition resolution and `samples_per_cell`, retain the seed, and report the
change in observables/eigenvalues rather than claiming a universal Ulam rate.

## 6. Quantum states and operators

### 6.1 State and operator validation

| Feature | Validation quantity | Acceptance |
| --- | --- | ---: |
| Constructed basis state | `abs(||psi||_2 - 1)` | `<= 5e-15` |
| Coherent state | `abs(||psi||_2 - 1)` | `<= 1e-12` after periodization/normalization |
| Dense unitarity | `unitarity_defect(U)` for `N <= 256` | `<= 1e-12` |
| One-step norm preservation | `abs(||apply(psi)||_2 - ||psi||_2)` | `<= 2e-13` for normalized seeded states |
| 1,000-step norm drift | Maximum recorded absolute norm drift | `<= 5e-11`; otherwise emit `NumericalWarning` |
| Dense vs matrix-free action | `||U_dense @ psi - apply(psi)||_2` | `<= 5e-12` for at least ten seeded normalized states and supported boundary phases |
| Kicked-rotor dense vs FFT | Same as above | `<= 1e-12` for `N` values covering odd/even supported cases |
| Invalid quantization | Constructor validation | Raises `ValidationError` before allocating an `N x N` matrix. |

Dense unitarity uses a normalized Frobenius defect so the threshold does not grow
merely because more matrix entries are present. Matrix-free implementations are
not converted to dense above their documented small-reference limit; they are
tested through action, adjoint/action identities, norm preservation, and scaling.

### 6.2 Quantum eigensystems

| Check | Acceptance |
| --- | --- |
| Eigenvalue modulus for a unitary operator | `max(abs(abs(lambda) - 1)) <= 1e-12` for dense reference; `<= 1e-9` for iterative results. |
| Dense eigenpair residual | Maximum normalized residual `<= 1e-12`. |
| Iterative eigenpair residual | Every converged pair `<= max(requested_tol, 1e-9)`; default requested residual tolerance is `1e-10`. |
| Eigenvector orthogonality | Nondegenerate dense vectors: `||V*V-I||_F / sqrt(K) <= 1e-12`. |
| Dense vs iterative eigenvalues | Optimal circular matching error `<= 1e-9` for converged requested pairs. |
| Degenerate subspaces | Projector distance `<= 1e-8`; individual vector phases/bases are not compared. |

Eigenphases are wrapped to one documented interval and circularly sorted.
Degeneracy within the effective residual/phase tolerance is recorded and emits a
diagnostic `NumericalWarning` only when it makes a requested statistic ambiguous.

## 7. Spectral and phase-space statistics

| Feature | Validation | Acceptance |
| --- | --- | --- |
| Phase wrapping/sorting | Hand-constructed endpoint and duplicate cases | Wrapped interval and order exact apart from `atol=1e-14` at the boundary. |
| Unfolding | Synthetic polynomial counting function | Reconstruction residual matches the configured fit tolerance, and the interior spacings are checked independently. A unit mean unfolded spacing is **not** an acceptance criterion: for both methods it follows algebraically from the circular spacings summing to the period over `count` levels. `method="polynomial"` additionally fixes the single wrap-around spacing at exactly `1` by construction, which the result records as `circular_gap_is_synthetic`. |
| Adjacent-gap ratios | Hand-computed small arrays | `rtol=1e-13`, `atol=1e-14`; zero gaps follow the documented degeneracy policy. |
| Spacing distribution | Counts and normalization | Counts exact; with `density=True` the histogram is normalized by the total sample count, so the integral equals one within `1e-12` only when `value_range` covers every spacing, and is otherwise the fraction inside the range. |
| Spectral form factor | Direct complex-sum reference at small `N` | `rtol=1e-12`, `atol=1e-14` for the same window/normalization. |
| Number variance | Direct sliding-window reference, plus the closed form for an equally spaced spectrum | `rtol=1e-11`, `atol=1e-13` against the reference; `atol=1e-12` against `frac(L) * (1 - frac(L))`. Window origins sit on a half-shifted grid so that a `samples` value commensurate with the level count cannot land on a level and miscount by one. |
| Poisson/RMT ensembles | Seeded ensemble reference | Absolute error is both within `5` reported standard errors and `<= 0.02`. |
| Spacing-distribution reference | Analytic identities of the Wigner surmise | `int P(s) ds = 1` and `int s P(s) ds = 1` to `rtol=1e-6`; closed-form peak positions to `rtol=1e-12`. Comparing a measured histogram against the surmise is counting-noise limited: at 20 bins over `[0, 4)` the systematic surmise-versus-exact difference stays under `0.02`, below the noise of any tractable sample, so acceptance uses `0.075` and additionally requires the wrong symmetry class to be rejected. |
| Mean gap-ratio reference | Published large-`N` and 3x3 surmise values | Constants exact to their quoted digits; a seeded ensemble reproduces the large-`N` value within `0.012`, about five standard errors at 200 matrices of dimension 64. |
| Husimi distribution | Quadrature normalization | `abs(integral - 1) <= 1e-10`. |
| IPR, participation, entropy | Basis-localized and uniform states | IPR/participation `rtol=1e-12`, `atol=1e-14`; entropy absolute error `<= 1e-12`. |

The `5`-standard-error stochastic limit makes false failures rare while still
scaling with sample size. All ensemble tests use fixed seeds and record empirical
variance. Tests must not tune a seed after observing a failure. Symmetry sectors
must be supplied before level statistics; absent/unknown sector metadata emits
`NumericalWarning` because mixed sectors can invalidate RMT comparisons.

## 8. Eigenpair and solver convergence policy

Every iterative result records:

- requested and achieved tolerance;
- normalized residual per returned pair;
- iteration/matvec count when available;
- convergence flag and solver termination reason;
- eigenvalue ordering rule and any shift/target;
- normalization convention;
- warnings and partial-result status.

For a general dense solver, require maximum normalized residual `<= 1e-12`. For
sparse or matrix-free solvers, request `tol=1e-10` by default and accept each pair
only when its independently recomputed normalized residual is `<= 1e-9`. The
factor-of-ten gap accounts for solver-specific stopping metrics but prevents a
solver's success flag from replacing independent verification.

If some requested eigenpairs converge, default behavior returns only clearly
marked partial results with `converged=False` and emits `ConvergenceWarning`.
If none converge, required arrays are invalid, or downstream normalization is
impossible, raise `ConvergenceError`. `strict=True` converts any partial or
nonconverged result into `ConvergenceError`.

## 9. Dense reference and scalable implementation agreement

### 9.1 Independence requirement

A reference is useful only when it does not call the optimized kernel being
tested. Small dense references may use straightforward loops, explicit matrices,
direct sums, and dense linear algebra. Production paths use vectorization, CSR,
FFT, or `LinearOperator` as appropriate.

Both paths may share validation and parameter objects, but must not share the
indexing/phase kernel whose correctness is under test.

### 9.2 Required cross-checks

| Pair | Reference sizes | Inputs | Acceptance |
| --- | --- | --- | --- |
| Scalar vs batch map step | Built-in models; batch `1, 2, 17` | Boundary and seeded interior states | Algebraic default tolerance. |
| Repeated scalar vs vectorized trajectory | Up to `32` steps | Same states | Multi-step deterministic tolerance. |
| Dense vs CSR Ulam | Up to `16 x 16` 2D cells, `64` samples/cell | Predetermined sample table | Same count table bitwise; independent build `rtol=1e-13`, `atol=1e-15`. |
| Dense Floquet vs FFT/apply | `N = 8, 16, 31, 32, 64, 128, 256` where supported | Basis vectors plus ten seeded normalized states | Quantum action thresholds in section 6. |
| Dense vs matrix-free adjoint | Same small `N` | Seeded state pairs | `abs(<x, A y> - <A* x, y>) <= 5e-12`. |
| Dense vs sparse eigenpairs | Matrix size `32` to `256` | Separated and deliberately clustered spectra | Eigenvalue residual/matching and degenerate projector thresholds. |

Reference tests cover zero coupling, weak/strong representative coupling,
supported boundary phases, odd/even dimensions where valid, exact partition cuts,
and invalid parameter regimes. The reference-size ceiling prevents correctness
tests from normalizing accidental dense allocation at production sizes.

## 10. Reproducibility policy

### 10.1 Guarantees

A stochastic public entry point accepts a keyword-only seed and constructs a local
NumPy generator. It never uses or mutates NumPy's process-global RNG. A result
records the root seed, deterministic child-seed derivation identifier, algorithm
version, parameters, Python/NumPy/SciPy versions, platform, and Git commit when
available.

With identical inputs, seed, package/dependency versions, platform, worker count,
and execution mode, library-controlled sampling and sweep scheduling must reproduce
the same arrays bit for bit. Tests compare exact arrays for Ulam sample generation,
partitions, RMT samples, and derived sweep seeds.

Bitwise identity is not promised across BLAS/LAPACK implementations, architectures,
dependency versions, or parallel reductions. In those cases compare invariant
quantities, residuals, subspaces, and statistical summaries using this document's
tolerances.

It is also not promised **within one process for complex elementwise arithmetic on
the oldest supported NumPy**. Measured on NumPy 1.26.4 with CPython 3.11, the
complex128 phase product inside one kicked-rotor Floquet step returns different
results on repeated calls with the same input array: eight invocations spread over
`2e-17`, and two `evolve` runs with identical arguments differed by `3e-16`. The
same code on NumPy 2.5 is bit-stable, and `numpy.fft` alone is bit-stable on both.
Tests therefore compare complex phase and FFT pipelines at rounding level rather
than with `assert_array_equal`.

The same difference shows up in absolute tolerances that were set from one NumPy
version. The unitarity defect of the Weyl translation operators measures `0.38 eps`
on NumPy 2.5 and `0.72 eps` on NumPy 1.26 over the same dimension and boundary-phase
grid, so a `1e-16` bound passed on one and failed on the other while both are pure
rounding in a single complex exponential. Bounds on quantities whose only error is
rounding are therefore stated as multiples of `numpy.finfo(numpy.float64).eps`, not
as absolute constants; an absolute constant below a few `eps` pins a NumPy build
rather than the library.

#### What this means when writing a test

Four assertions in this suite passed on the development machine and failed in CI,
each because they pinned a property of one machine rather than of the library. The
patterns are worth naming, because a green local run is not evidence against any of
them:

- **A reduction whose order is an implementation choice is not bit-reproducible.**
  Changing `chunk_size` changes the shape of the array handed to the matrix
  product, so BLAS blocks it differently and the sums land differently. Compare
  such results at rounding level, and bound the difference against the **peak** of
  the array rather than per element: a Husimi tail cell at `1e-10` beside a peak at
  `30` moves by an absolute `7e-18`, which is a relative `1e-11`, so a per-element
  `rtol` rejects a correct result. Genuine memoization -- same inputs, cache on
  versus off -- *is* bit-exact and should be asserted with `assert_array_equal`.
- **`argmax` over a tie is decided by the implementation.** A coherent state
  centered at `0.25` on a 64-cell midpoint grid sits exactly between two cells
  that hold bit-identical values. Assert that the peak lies within one cell of the
  requested point, or first assert that there is no tie
  (`sum(values >= values.max() - eps) == 1`) and only then pin the cell.
- **A finite-time quantity from a single chaotic orbit is not reproducible at all.**
  Rounding differences are amplified by `exp(lambda t)`, so two machines following
  "the same" orbit have diverged completely within a few hundred steps. Use a
  phase-space mean over an ensemble and size the tolerance from the spread across
  orbit sets. Where an exact answer is wanted, pin a quantity that does not depend
  on the orbit, as `tests/test_lyapunov.py` does by using a fixed point whose
  Jacobian is constant.
- **Complex elementwise arithmetic is not reproducible call to call on the floor**,
  per the paragraphs above.

Seeded reproducibility is unaffected and is asserted at the floor: Lyapunov
exponents, Ulam matrices, stationary densities, spectral statistics, and sweep child
seeds all reproduce bit for bit across processes and thread counts on both the
newest and the oldest supported dependency set. What the floor does not support is a
bit-exact comparison between two complex-valued results computed from
equal-but-distinct arrays.

### 10.2 Failure and warning behavior

| Situation | Default behavior |
| --- | --- |
| Stochastic low-level call with `seed=None` | Allowed for interactive use; result records non-reproducible seed state. |
| Persisted `Experiment`/sweep with no seed | Emit `ReproducibilityWarning`; strict reproducibility mode raises `ValidationError`. |
| Requested deterministic replay but environment/algorithm version differs | Emit `ReproducibilityWarning`, record the differences, and require explicit opt-in to overwrite prior results. |
| Parallel execution would change RNG consumption order | Derive child seeds from stable parameter/job identity; never rely on completion order. |
| Failed sweep item | Record exception and seed, continue when configured, and make only incomplete/failed items eligible for resume. |

## 11. Warning and exception policy

Warnings are both emitted through Python's warning system and retained in result
metadata as stable codes plus human-readable messages. A repeated warning from a
batch is summarized with affected counts/indices rather than emitted per element.

| Condition | Outcome |
| --- | --- |
| Invalid shape, dtype, bounds, parameter, normalization request, or impossible quantization | `ValidationError`; no numerical work begins. |
| Non-finite internal value that invalidates the result | `NumericalError` if no meaningful result exists; otherwise partial result plus `NumericalWarning`. |
| Iterative solver fails all required outputs | `ConvergenceError`. |
| Iterative solver returns meaningful partial/unconverged output | Result with `converged=False` plus `ConvergenceWarning`; strict mode raises. |
| Norm, conservation, or residual exceeds the documented diagnostic threshold after a result exists | Result plus `NumericalWarning`, unless the function's contract requires the invariant, in which case raise `NumericalError`. |
| Insufficient sample size for a reliable asymptotic/statistical interpretation | Result with uncertainty plus `NumericalWarning`; structurally impossible sample sizes raise `ValidationError`. |
| Missing symmetry-sector information for RMT comparison | `NumericalWarning`; metadata marks sector as unknown. |
| Non-reproducible or environment-mismatched execution | `ReproducibilityWarning`; strict reproducibility mode raises `ValidationError`. |

Public hierarchy additions required by this policy are `NumericalError` and
`ReproducibilityWarning`, both rooted in `ChaosNumericsError` or
`ChaosNumericsWarning` as appropriate.

## 12. Performance benchmarks

Correctness tests run on every supported environment. Timing regression is
blocking only on a pinned, low-variance runner; other CI runners publish results
without failing a pull request. Peak resident memory and forbidden dense
allocations remain blocking wherever measurement is reliable.

### 12.1 Benchmark cases

| ID | Capability | Standard case | Smoke case | Report |
| --- | --- | --- | --- | --- |
| `classical_scalar_iterate` | Scalar orbit | Standard map, `100_000` steps | `20_000` steps | median time, MAD, IQR, peak allocation |
| `classical_batch_iterate` | Batch orbit | Standard map, batch `1_024`, `1_000` steps | batch `256`, `100` steps | median time, MAD, IQR, peak allocation |
| `ulam_build` | CSR Ulam construction | Cat map, `64 x 64` cells, `128` samples/cell, fixed seed | `16 x 16` cells, `64` samples/cell | median time, MAD, IQR, peak allocation |
| `quantum_fft_evolve` | Matrix-free evolution | Kicked rotor, `N=16_384`, `100` steps, fixed state | `N=4_096`, `20` steps | median time, MAD, IQR, peak allocation |
| `dense_eigensystem` | Dense reference eigensystem | Seeded unitary, `N=256`, full spectrum | `N=96` | median time, MAD, IQR, peak allocation |
| `sparse_leading_eigenpairs` | Sparse eigenanalysis | Fixed seeded stochastic sparse matrix, `N=4_096`, about `9` nonzeros/column, `count=8` | `N=1_024` | median time, MAD, IQR, peak allocation |

`benchmarks/benchmark_suite.py --mode smoke` runs the smoke variants on ordinary
pull requests; `--mode standard` runs the standard cases on the pinned benchmark
job and before release. Benchmark fixtures, seeds, thread counts, dependency
versions, and a coarse runner class are recorded with the result.

Peak memory is reported as the `tracemalloc` peak of traced Python allocations,
which is what the "must not allocate a dense `(N, N)` array" gates need and is
portable across the supported platforms. Native peak RSS remains deferred; see
section 14.

### 12.2 Timing protocol and regression gate

- Pin BLAS/OpenMP thread count for the baseline and candidate.
- Warm up imports, FFT planning/caches, and JIT-free code before measurement.
- Use at least `10` measured repetitions for sub-second cases and at least `5` for
  longer cases; report median, median absolute deviation (MAD), and interquartile
  range.
- Aggregate very short operations until each measured sample lasts at least
  `100 ms`; do not threshold timer noise.
- Compare only results from the same runner class and benchmark fixture/schema.

A timing regression fails the pinned job when both conditions hold:

```text
candidate_median > 1.30 * baseline_median
candidate_median - baseline_median > max(50 us, 6 * baseline_MAD)
```

The absolute floor was 5 ms in an earlier revision, which turned out to silence
the regressions it was meant to survive. Because short cases are aggregated to at
least 100 ms per sample, their medians are not timer-resolution limited, and a
5 ms floor sat far above them: at the recorded smoke medians it allowed
`quantum_fft_evolve` to degrade 4.8x and `classical_batch_iterate` 3.5x without
failing. With a 50 us floor the `6 * MAD` term is the binding one, as intended,
and every recorded case now fails at the documented 1.30x.

Only `--check-timing` makes that gate blocking, and it is reserved for the pinned
runner. The recorded baseline carries the runner class it was measured on, and the
timing comparison is skipped outright when the current runner class or the
benchmark mode differs.

That policy has a consequence worth stating plainly: **no timing comparison
happens in CI today.** The committed baseline was recorded on `Darwin-arm64` and
the shared runner is `Linux-x86_64`, so every case reports `NEW` and nothing is
compared. Timing regressions are caught only once a pinned runner records its own
baseline and runs with `--check-timing`. Until then the suite's blocking value
comes entirely from the allocation and scaling gates, which are deterministic and
block on every runner.

A change above `15%` that does not cross the hard gate is reported as a warning.
The dual `30%` and variability threshold avoids failing on ordinary runner noise
while detecting material regressions. A baseline is never updated automatically:
the change requires a reviewed benchmark artifact, an explanation, and at least
five stable candidate runs.

### 12.3 Scaling and memory gates

| Capability | Scaling check | Acceptance |
| --- | --- | --- |
| Scalar/batch iterate | Double steps with other inputs fixed | Time ratio `< 2.6`; peak memory grows linearly with stored output and must not include an additional full-size copy. |
| Ulam build | Double `samples_per_cell` | Time ratio `< 2.8`; no allocation with shape `(cell_count, cell_count)` in production path. |
| FFT evolution | Double `N` | Time ratio `< 2.8` at fixed steps; peak working memory ratio `< 2.6`; no `(N, N)` allocation. |
| Sparse eigensolver | Double sparse `N` at fixed density/count | Time ratio `< 3.5`, reported with matvec count; no dense conversion. |

These ratios are guardrails, not asymptotic proofs. They include room for cache and
solver-iteration changes but reject obvious quadratic dense behavior in paths that
must scale. Scaling gates run on the pinned runner with at least three size pairs.

### 12.4 Known scaling limits in v0.1

The gates above check the ratio that each capability promises. They do not claim
that every path is as fast as it could be, and two limits are worth stating so that
a user does not discover them as an apparent hang.

**Ulam construction is quadratic in the cell count.** `build_ulam` allocates and
scans an array of length `cell_count` once per source cell, so the cost is
`O(cell_count**2)` independently of `samples_per_cell`. Measured on an Apple
Silicon laptop with a Cat map:

| Partition | Cells | Samples/cell | Wall time | CSR size |
| --- | --- | --- | --- | --- |
| `32 x 32` | 1,024 | 256 | 0.09 s | 0.10 MB |
| `64 x 64` | 4,096 | 256 | 0.42 s | 0.43 MB |
| `128 x 128` | 16,384 | 256 | 1.66 s | 1.75 MB |
| `256 x 256` | 65,536 | 64 | 10.4 s | 6.45 MB |

The empirical fit is `t ~ cells * (70 us + 1.45 ns * cells)`, which puts
`512 x 512` near two minutes and `1024 x 1024` near half an hour. Treat
`256 x 256` as the practical ceiling until the accumulation is rewritten as a
single COO build, which is linear in `cell_count * samples_per_cell`.

A corollary is that **raising `samples_per_cell` is nearly free**: the fixed
per-cell cost dominates, so `64 x 64` with 256 samples per cell costs 1.25 times
what 64 samples per cell costs while giving four times the sampling statistics.
There is no reason to run with a small `samples_per_cell`.

**Correlation and displacement statistics are quadratic in the series length**
when they run their direct reference algorithm, because the default `max_lag` is
`count - 1`. `method="direct"` is that reference algorithm and is what the
acceptance criteria above are stated against; `method="fft"` computes the same
quantity through Wiener-Khinchin in `O(n log n)`, and `method="auto"` (the default)
selects it once `time * max_lag` exceeds 250,000. The two agree to `1e-13`
relative on bounded coordinates. The fourth-moment uncertainty estimator loses more
precision in the FFT path, `5e-5` relative on unwrapped coordinates against `2e-15`
on bounded ones, because it evaluates a fourth moment through a correlation rather
than a direct sum; use `method="direct"` when that uncertainty is the reported
quantity.

Peak memory uses process RSS or an equivalent native-allocation-aware measure;
`tracemalloc` alone is insufficient for NumPy/SciPy buffers. Tests additionally
instrument constructors or allocation shapes to reject forbidden dense matrices.

## 13. Feature validation matrix

| Feature family | Analytic/independent reference | Property/invariant | Cross-implementation | Stochastic/convergence |
| --- | --- | --- | --- | --- |
| Classical maps | Known one-step and Jacobian | Periodicity, symplecticity/area, batch shape | Scalar vs batch | Long-run statistics where relevant |
| Lyapunov | Cat-map analytic exponents | QR orthogonality, symplectic sum | Alternate iteration lengths | Window convergence and warnings |
| Correlation/transport | Synthetic time series/power laws | Symmetry, non-negativity where applicable | Direct short sums | Standard errors and fit diagnostics |
| Periodic orbits | Known low-period points | Wrapped residual, multiplier residual | Independent Jacobian product | Solver termination history |
| Partitions | Hand-enumerated grids | Bounds, index round trip, seed replay | Scalar vs batch lookup | Exact seeded samples |
| Ulam | Predetermined transition counts | Non-negativity, column sums | Dense vs CSR | Resolution/sample convergence |
| Quantum maps | Small dense matrices | Norm/unitarity, adjoint identity | Dense vs FFT/LinearOperator | Long-step norm drift |
| Eigensystems | Dense direct solve | Residuals, orthogonality/subspaces | Dense vs iterative | Convergence/partial-result policy |
| Spectral statistics | Hand sums and theory/reference | Normalization, circular ordering | Vectorized vs direct small case | Seeded ensembles with error bars |
| Husimi/localization | Localized/uniform states | State/grid normalization | Direct small-grid reference | Grid-resolution convergence |
| Experiment/storage | Known fixture round trip | Metadata/array separation | Fresh vs resumed run | Stable derived seeds and failures |

Each implementation issue selects the relevant rows and adds tests before it may
be marked complete. A single golden file is never the sole validation of a
numerical algorithm.

## 14. CI and release enforcement

### Pull requests

- deterministic unit/property/reference tests;
- small dense/scalable cross-checks;
- dtype, shape, boundary, warning, and exception tests;
- reproducibility tests with fixed seeds;
- benchmark smoke execution, with nonblocking timing report on shared runners;
- allocation checks preventing prohibited dense paths.

### Nightly or scheduled pinned runner

- standard benchmark cases and scaling pairs;
- larger stochastic/RMT confidence checks;
- partition/sample-size and grid-resolution convergence studies;
- repeated results sufficient to estimate benchmark variability.

### Release candidate

- all supported Python/NumPy/SciPy combinations pass correctness tests;
- pinned performance suite passes against the reviewed baseline;
- wheel/sdist installation reproduces documented examples;
- numerical metadata contains tolerances, residuals, convergence, environment,
  seed, and warnings where applicable;
- no known invariant failure is waived without a documented limitation and issue.

## 15. Open implementation decisions

The standards above are fixed at the behavioral level. These choices remain for
their owning implementation issues and must not weaken the acceptance criteria:

- the exact benchmark framework and native peak-RSS collector (KEN-111/KEN-129);
- the serialized structured-warning schema and result-field layout (KEN-113);
- the finite-difference helper and scale-aware step formula used only as a
  Jacobian cross-check (KEN-114);
- the sparse eigensolver selection and norm estimator (KEN-120/KEN-121);
- model-specific supported quantum dimensions and boundary phases (KEN-122/123);
- the specific unfolding families and RMT reference-generation method
  (KEN-124/125);
- the Husimi quadrature convention and grid refinement schedule (KEN-126);
- cross-platform reproducibility labels and persistence migrations (KEN-128).

Any implementation choice is recorded in result metadata and tested against the
scale-normalized criteria here.

## 16. Traceability

- KEN-108 defines v0.1 scope.
- KEN-109 defines API, shapes, dtypes, coordinates, and dependencies.
- KEN-110 and this document define accuracy, validation, reproducibility,
  convergence, and performance gates.
- KEN-111 through KEN-129 implement these gates and provide feature-specific
  evidence.

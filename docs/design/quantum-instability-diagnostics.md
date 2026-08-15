# Loschmidt echo and out-of-time-order correlator

Status: implemented baseline  
Last reviewed: 2026-08-11

Unitary evolution preserves every overlap, so the classical Lyapunov instability
has no direct quantum analogue. These two routines are the standard
finite-dimensional stand-ins, and they answer the same question from opposite
ends: the echo measures how fast two *slightly different* operators pull the
*same* state apart, the correlator how fast *one* operator spreads a local
observable until it stops commuting with a second one.

## Why both take whole models

```text
loschmidt_echo(reference, perturbed, initial_state, *, steps, ...)
otoc(model, *, steps, operator_a=None, operator_b=None, state=None, ...)
```

The echo takes two models rather than one model and a perturbation size. The
rejected signature was `loschmidt_echo(model, *, delta, steps)`, which would have
to decide *which* parameter `delta` perturbs. That is a per-model choice
(`kick_strength` for the rotors, the matrix for the cat map, the boundary phases
for anything), and it is exactly the thing a reader of the result needs to know.
Nothing about the pair is assumed beyond a shared `dimension`, which is
validated: the two may be different classes, and
`tests/test_quantum_foundations.py` pins a `DenseUnitary` identity against a
`DenseUnitary` Hadamard for that reason.

`otoc` takes one model because `A(t) = U^(-t) A U^t` needs one propagator. The
operator pair is where the convention lives, so it is a defaulted argument
rather than a hidden constant, and the default is published as
`weyl_translations`.

## Weyl translation convention

`weyl_translations(N, boundary_phases=...)` returns `(T_q, T_p)` in the position
basis, matching the `KickedRotor` ordering `q_j = (j + alpha) / N`:

```text
T_q = diag(exp(2*pi*i*q_j))
T_p = the cyclic shift j -> j+1, carrying exp(2*pi*i*beta) on the wrap
```

Both are unitary and they satisfy

```text
T_q T_p = exp(2*pi*i/N) * T_p T_q
```

Measured over `N` in `{2, 3, 8, 16, 64, 128, 257, 512, 1024}` and boundary phases
in `{(0,0), (1/2,1/2), (1/4,0.13)}`, the worst unitarity defect
`||U.H U - I||_F / sqrt(N)` is `0.38 eps` and the worst Weyl-relation residual is
`2.43 eps`. Both are pure rounding in `exp(2*pi*i*(j+alpha)/N)`, the only inexact
factor in either matrix, so the tests bound them in multiples of
`numpy.finfo(numpy.float64).eps` rather than as absolute constants. An earlier
`1e-16` bound passed on NumPy 2.5 (`0.38 eps`) and failed on the NumPy 1.26 floor
(`0.72 eps`); see the reproducibility section of `numerical-standards.md`.

## `C(0)` is a closed form, and it is what pins the convention

Because `[T_q, T_p] = (w - 1) T_p T_q` with `w = exp(2*pi*i/N)` and `T_p T_q` is
unitary, `[A,B].H [A,B]` is a multiple of the identity. So

```text
C(0) = |w - 1|^2 = 4 * sin(pi/N)^2
```

for the infinite-temperature trace **and** for every pure state, independently of
which state. Measured agreement is `2.2e-16` or better over `N` from 4 to 256 for
both expectation values. This is the load-bearing test of the whole operator
convention: a transposed shift, a missing boundary phase, or the conjugate
ordering all break it, while leaving the correlator bounded and growing.

The Heisenberg recursion is separately checked against a brute-force reference
that rebuilds each `A(t)` with an independent `matrix_power`, agreeing to
`1.3e-15`. Property tests alone would not catch a wrong-but-consistent recursion,
which still starts at the right place, stays bounded, and grows.

## `C` saturates near 2, and that is the operators talking

`T_q` and `T_p` are bounded unitaries, not the unbounded `q` and `p`, so
`||[A,B]|| <= 2 ||A|| ||B||` gives a hard bound of `4`. Measured at `N = 128`,
`K = 10`: maximum `2.0426` over `t <= 200`, minimum `1.9557` for `t >= 12`,
time-average `1.9946` over `100 <= t <= 200`. The correlator stops rather than
continuing, and the value it stops at is a property of the operator pair, not of
the dynamics.

`K = 0` is the free rotor. Free rotation only multiplies `T_q` by phases, so
`C(t)` is constant to rounding: measured worst variation `3.3e-16` over 200 steps
at `N = 128`, from the summation order in `||[A,B]||_F^2` rather than from any
bit-exact cancellation.

## The fitted rate is not `2*lambda`, and it is reported rather than fitted away

The ratio of the fitted exponential rate to `2*lambda` runs `8.00`, `2.02`,
`1.09`, `0.55` as `K` goes `1, 2, 5, 10`. It crosses `1` rather than converging to
it. Three effects, all of them physics:

* `C(t)` measures the *second moment* of the stability multiplier, so its growth
  rate is the order-2 generalized Lyapunov exponent, which is `>= 2*lambda`
  whenever the finite-time exponent fluctuates. That is the overshoot at low `K`.
* The Ehrenfest time `t_E ~ ln(1/hbar_eff)/lambda` is only `2.7` steps at
  `K = 10`, `N = 512`. The fit window is then longer than the exponential regime
  it is fitting, and saturation pulls the rate down. That is the undershoot at
  high `K`.
* The operators are bounded, per the previous section.

Widening the semiclassical window is not available: `hbar_eff` is tied to `N` by
the torus quantization and raising `N` costs `O(N**3)`. **The documented use is
the ratio between regimes, not the absolute rate.** No `lyapunov_from_otoc`
helper is provided, because it would have to pick a fit window and would then
report a number that this table shows is not the quantity its name claims.

## Echo values are not clipped

`M(0)` is `1` by construction: bit-exactly `1.0` for an initial state with
exactly representable amplitudes such as a basis state, and otherwise `1` to the
rounding of one normalization (worst measured error `2.2e-16` over 200 random
states at `N = 128`). Two identical models give `M(t) = 1` to `2.7e-14` over 50
steps, and `M(t)` may exceed `1` by up to `1.8e-15`.

The values are deliberately **not** clipped into `[0, 1]`. A clipped diagnostic
hides exactly the numerical drift it should be reporting: a run whose norm audit
is drifting shows up as `M > 1`, and clipping would turn that signal into a
plausible-looking `1.0`.

`|M|` is unchanged by exchanging the two models, because the overlap becomes its
own conjugate. That is a property of the definition rather than of the
implementation, and the suite pins it. It holds bit-exactly on NumPy 2.5, since
the two runs are the same code on swapped inputs, but the test compares at
rounding level: the oldest supported NumPy does not make complex elementwise
arithmetic reproducible even call to call, and the swap differs by `1.8e-15` there.
See the reproducibility section of `numerical-standards.md`.

## Regime separation is the diagnostic; ordering inside a regime is not

Measured at `N = 256`, both models differing only in `kick_strength` by `1e-2`,
starting from the coherent state at `(q, p) = (0.5, 0.0)`:

| `K` | `M(20)` | `M(40)` |
| --- | --- | --- |
| 0.2 | 0.976508 | 9.23e-01 |
| 0.5 | 0.997523 | 9.88e-01 |
| 1.0 | 0.998252 | 9.92e-01 |
| 2.0 | 0.993132 | 9.74e-01 |
| 5.0 | 0.020998 | 1.31e-03 |
| 10.0 | 0.073779 | 5.86e-03 |

`M(40)` falls by more than two orders of magnitude between the last KAM torus
(`K_c = 0.9716`) and the globally chaotic regime, on an unchanged perturbation:
the worst-decaying regular row is `166` times above the best-surviving chaotic
one. Two honest caveats, both of which the notebook and the docstring state:

* The ordering is not monotone in `K` on the regular side. `K = 0.2` decays
  *more* than `K = 0.5`, because the coherent state used here sits in a large
  island at `K = 0.5` and in a much flatter landscape at `K = 0.2`.
* Nor on the chaotic side. `K = 10` decays *less* than `K = 5`, because the echo
  responds to `delta K / hbar_eff` and the golden-rule regime is not monotone in
  it either.

## Cost and the dense limit

`loschmidt_echo` runs two `evolve` calls with `return_history=True`, so it
inherits the matrix-free and FFT routes and their `norm_tolerance` audit, and its
memory is the two histories: `2 * (steps + 1) * N` complex numbers, 16 MiB at
`N = 512`, `steps = 1000`.

`otoc` requires a unitary model, to the same `1e-12` defect tolerance
`eigenstates` uses, and raises `NumericalError` otherwise. The recursion inverts
the propagator by taking its adjoint, which is the inverse only for a unitary
operator. This guard was missing: a model whose dense form is `0.9 * I` (defect
0.19) returned the smooth decaying correlator `[2.0, 1.312, 0.861, 0.565]` with no
exception and no warning, while `eigenstates` and `evolve` both refused the same
object. The defect was recorded in the metadata, which only helps a caller who
thinks to read it.

`otoc` has only a dense path, because `A(t)` is a matrix and there is nothing for
a matrix-free algorithm to act on. Each step is four `O(N**3)` products and
`dense_limit` guards materializing the model exactly as it does elsewhere.
Measured: 0.04 s at `N = 256`, `steps = 24`; 0.11 s at `N = 512`, `steps = 8`.

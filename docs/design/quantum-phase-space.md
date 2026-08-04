# Torus coherent states and quantum phase-space data

Status: implemented baseline for KEN-126  
Last reviewed: 2026-08-05

## Coherent-state convention

States use the finite position basis `q_j=(j+alpha)/N`. For center `(q0,p0)`,
the unnormalized periodized Gaussian is

```text
psi_j = sum_m exp(-pi*N*(q_j-q0+m)^2)
              * exp(2*pi*i*N*p0*(q_j-q0+m))
              * exp(-2*pi*i*beta*m),       m=-images,...,+images.
```

`alpha` and `beta` come from `BoundaryPhases`; centers are wrapped to `[0,1)`.
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

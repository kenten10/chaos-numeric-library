# Long-range spectral statistics and RMT references

Status: implemented baseline for KEN-125  
Last reviewed: 2026-08-05

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

## Number variance

For uniformly spaced circular origins, the implementation counts levels in
half-open windows `[origin, origin+L)` and returns

```text
Sigma^2(L) = mean_origin((n(origin,L) - L)^2).
```

The default reports the finite circular-spectrum result. The explicit
`finite_size_correction=True` option multiplies by `1/(1-L/N)` and therefore
requires `L<N`. Uncertainty is the standard error across origins, or the seeded
bootstrap standard deviation when bootstrap resampling is requested.

## RMT references

`rmt_reference` provides analytic connected form factors for Poisson, GOE, GUE,
and CUE. It provides Poisson number variance directly and evaluates GOE/GUE bulk
cluster-function integrals numerically. When CUE dimension is supplied, number
variance uses the finite circular kernel
`[sin(pi*s)/(N*sin(pi*s/N))]^2`; otherwise it uses the GUE bulk limit. Exact
analytic references return zero uncertainty and variance arrays.

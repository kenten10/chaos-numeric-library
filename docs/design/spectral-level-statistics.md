# Eigenphase preparation and local level statistics

Status: implemented baseline for KEN-124  
Last reviewed: 2026-08-04

`prepare_eigenphases` preserves the input array, wraps a separate copy to the
half-open interval `[0, 2*pi)`, sorts it, and calculates all circular gaps. The
last gap crosses the branch cut from the largest phase to the smallest phase plus
`2*pi`. Gaps at or below `degeneracy_tolerance` are marked as degenerate.

Level statistics require a single symmetry sector. Omitting `symmetry_sector`
emits `NumericalWarning` and stores a stable `symmetry-sector-unknown` diagnostic;
mixed independent spectra can give misleading random-matrix comparisons.

`unfold` supports two explicit methods:

- `mean` divides by the global circular mean spacing `2*pi/N`;
- `polynomial` fits the empirical counting ranks as a polynomial of phase,
  requires the fitted map to be strictly increasing, and rescales the circular
  result to exact unit mean spacing.

`UnfoldedSpectrum` retains raw phases, wrapped/sorted phases, unfolded levels,
unfolded circular spacings, method settings, and fit residual. Empty and
single-level inputs return empty spacing/statistic arrays with diagnostics.

`adjacent_gap_ratios` includes adjacent pairs across the circular branch cut.
For zero/degenerate gaps, `drop` (default) excludes affected ratios, `zero`
defines them as zero, and `raise` rejects the input. `spacing_distribution`
normalizes spacings to unit empirical mean before returning histogram values and
bin edges; density histograms integrate to one over the included range.

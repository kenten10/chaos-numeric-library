from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad, trapezoid  # type: ignore[import-untyped]

from chaos_numerics.core import NumericalError, NumericalWarning, ValidationError
from chaos_numerics.spectral import (
    Ensemble,
    adjacent_gap_ratios,
    mean_gap_ratio_reference,
    prepare_eigenphases,
    rmt_reference,
    spacing_distribution,
    unfold,
)


def _haar_unitary(
    rng: np.random.Generator, dimension: int
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    """Draw one Haar-distributed unitary (CUE) via a phase-fixed QR decomposition."""
    gaussian = rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal(
        (dimension, dimension)
    )
    unitary, triangular = np.linalg.qr(gaussian)
    diagonal = np.diag(triangular)
    return np.asarray(unitary * (diagonal / np.abs(diagonal)).conj())


def test_prepare_wraps_sorts_keeps_endpoint_and_diagnoses_degeneracy() -> None:
    raw = np.asarray([2.0 * np.pi, -np.pi, 0.5, 3.0 * np.pi, 0.0])

    with pytest.warns(NumericalWarning, match="detected 2 circular"):
        result = prepare_eigenphases(raw, symmetry_sector="even")

    np.testing.assert_allclose(result.phases, [0.0, 0.0, 0.5, np.pi, np.pi], atol=1e-14)
    np.testing.assert_allclose(
        result.spacings,
        np.diff(np.asarray([0.0, 0.0, 0.5, np.pi, np.pi, 2.0 * np.pi])),
        atol=1e-14,
    )
    np.testing.assert_array_equal(result.raw_phases, raw)
    assert result.metadata.parameters["degenerate_pairs"] == 2
    assert [warning.code for warning in result.metadata.warnings] == ["eigenphase-degeneracy"]
    assert not result.raw_phases.flags.writeable


@pytest.mark.parametrize(
    "phase",
    [-1e-18, -1e-16, -5e-16, 2.0 * np.pi - 1e-17],
)
def test_prepare_accepts_phases_a_rounding_error_outside_the_interval(phase: float) -> None:
    """``np.mod`` returns exactly ``2*pi`` for tiny negative inputs.

    ``np.angle(complex(1, -1e-18))`` legitimately evaluates to ``-1e-18``, so any
    unitary with an eigenvalue just below the positive real axis (an operator
    near the identity, a symmetry-protected eigenvalue) used to make preparation
    raise ``ValidationError``.
    """
    prepared = prepare_eigenphases([0.1, phase], symmetry_sector="near-identity")

    assert bool(np.all(prepared.phases >= 0.0))
    assert bool(np.all(prepared.phases < 2.0 * np.pi))
    np.testing.assert_array_equal(prepared.phases, np.sort(prepared.phases))


def test_unknown_sector_warns_and_empty_small_inputs_are_safe() -> None:
    with pytest.warns(NumericalWarning, match="symmetry sector is unknown"):
        empty = prepare_eigenphases([], symmetry_sector=None)
    singleton = prepare_eigenphases([0.25], symmetry_sector="only")

    assert empty.count == 0
    assert empty.spacings.size == 0
    assert unfold(empty).values.size == 0
    assert adjacent_gap_ratios(empty).values.size == 0
    assert singleton.spacings.size == 0
    assert spacing_distribution(singleton, bins=4).sample_count == 0


def test_mean_and_polynomial_unfolding_retain_source_data() -> None:
    ranks = np.arange(64, dtype=np.float64)
    phases = (-10.0 + np.sqrt(100.0 + 4.0 * ranks)) / 2.0
    prepared = prepare_eigenphases(phases[::-1], symmetry_sector="unit-test")

    polynomial = unfold(prepared, method="polynomial", degree=2)
    mean = unfold(prepared, method="mean")

    np.testing.assert_allclose(polynomial.values, ranks, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(polynomial.spacings, np.ones(64), rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(polynomial.raw_phases, phases[::-1])
    np.testing.assert_array_equal(polynomial.phases, prepared.phases)
    fit_residual = polynomial.metadata.parameters["fit_residual"]
    assert isinstance(fit_residual, float)
    assert fit_residual <= 1e-10
    assert polynomial.metadata.parameters["mean_unfolded_spacing"] == pytest.approx(1.0)
    assert np.mean(mean.spacings) == pytest.approx(1.0, abs=1e-14)


def test_unfold_flags_its_construction_identities_instead_of_hiding_them() -> None:
    """Unit mean spacing and the polynomial wrap-around gap are not diagnostics.

    Circular spacings sum to the period by construction, so ``mean == 1`` holds
    for every method and every input. Polynomial unfolding additionally rescales
    the fitted counting function onto ``[0, count-1]``, which forces the final
    branch-cut spacing to exactly ``1``: one of ``count`` spacings is synthetic.
    Both facts must be advertised in metadata so that neither is mistaken for
    evidence that the unfolding worked.

    "By construction" is not "bit-exact". The rescaling onto ``[0, count-1]`` is a
    floating-point division, so the wrap-around spacing is *usually* the bit
    pattern of ``1.0`` but need not be: over 400 random 128-level spectra 336 gave
    exactly ``1.0``, 32 missed by around ``1e-16``, and 32 tripped the
    monotonicity guard. Which of those happens depends on the last bits of
    ``polyfit`` and therefore on the BLAS the wheel was built against, so this
    asserts to a few ulp rather than to equality.
    """
    rng = np.random.default_rng(11)
    prepared = prepare_eigenphases(rng.uniform(0.0, 2.0 * np.pi, 128), symmetry_sector="uniform")

    polynomial = unfold(prepared, method="polynomial", degree=5)
    mean = unfold(prepared, method="mean")

    assert abs(float(polynomial.spacings[-1]) - 1.0) <= 4.0 * float(np.finfo(float).eps)
    assert polynomial.metadata.parameters["circular_gap_is_synthetic"] is True
    assert "synthetic-circular-gap" in [code.code for code in polynomial.metadata.warnings]

    assert mean.metadata.parameters["circular_gap_is_synthetic"] is False
    assert "synthetic-circular-gap" not in [code.code for code in mean.metadata.warnings]
    assert float(mean.spacings[-1]) != 1.0

    for spectrum in (polynomial, mean):
        assert spectrum.metadata.parameters["mean_unfolded_spacing_is_identity"] is True
        assert float(np.sum(spectrum.spacings)) == pytest.approx(128.0, rel=1e-14)


def test_polynomial_unfolding_rejects_a_non_monotone_fit() -> None:
    """A least-squares polynomial can turn over inside a large gap.

    A non-monotone counting function produces negative unfolded spacings and makes
    every downstream statistic meaningless, so ``unfold`` refuses it. This is not a
    corner case that needs contriving: at degree 5 on 128 uniform-random levels it
    happens for about 8% of spectra (32 of the first 400 seeds), and seed 10 is the
    first of them.
    """
    rng = np.random.default_rng(10)
    prepared = prepare_eigenphases(rng.uniform(0.0, 2.0 * np.pi, 128), symmetry_sector="uniform")

    with pytest.raises(NumericalError, match="not strictly increasing"):
        unfold(prepared, method="polynomial", degree=5)

    # The same spectrum unfolds fine by the mean method, so the guard is about the
    # fit and not about the input being unusable.
    assert unfold(prepared, method="mean").count == 128


def test_exact_zero_gaps_are_degenerate_at_zero_tolerance() -> None:
    """``tolerance=0.0`` must still catch a gap of exactly zero.

    Both degeneracy checks compare ``spacing <= tolerance``. With the default
    ``1e-10`` a strictly-less-than comparison behaves identically, because a
    genuine double eigenvalue gives a gap of exactly ``0.0`` and ``0 < 1e-10``.
    Only ``tolerance=0.0`` separates the two, and that is the documented way to
    ask for "exact degeneracies only" -- ``<`` would then find nothing, silently
    drop the diagnostic, and let a repeated eigenphase through into a gap-ratio
    denominator.
    """
    with pytest.warns(NumericalWarning, match="detected 1"):
        prepared = prepare_eigenphases(
            [0.0, 0.0, 1.0], symmetry_sector="all", degeneracy_tolerance=0.0
        )

    assert prepared.metadata.parameters["degenerate_pairs"] == 1
    assert prepared.metadata.parameters["degeneracy_tolerance"] == 0.0
    assert [item.code for item in prepared.metadata.warnings] == ["eigenphase-degeneracy"]
    np.testing.assert_array_equal(prepared.spacings[:2], [0.0, 1.0])

    # Two of the three circular ratios touch the zero gap, so "drop" keeps one.
    dropped = adjacent_gap_ratios(prepared, degeneracy="drop", tolerance=0.0)
    assert dropped.values.size == 1
    assert dropped.metadata.parameters["dropped_ratios"] == 2
    zeroed = adjacent_gap_ratios(prepared, degeneracy="zero", tolerance=0.0)
    assert int(np.count_nonzero(np.asarray(zeroed.values) == 0.0)) == 2
    with pytest.raises(ValidationError, match="degenerate gaps"):
        adjacent_gap_ratios(prepared, degeneracy="raise", tolerance=0.0)


@pytest.mark.parametrize(
    "roundtrip",
    [lambda item: pickle.loads(pickle.dumps(item)), copy.deepcopy],
    ids=["pickle", "deepcopy"],
)
def test_level_containers_survive_pickle_and_deepcopy(
    roundtrip: Callable[[Any], Any],
) -> None:
    """Read-only arrays and value equality must survive both round trips.

    NumPy drops ``writeable=False`` when it pickles an array and ``deepcopy`` uses
    the same protocol, so a container without ``__reduce__`` returns a mutable copy
    of arrays that the public API documents as read-only views. ``eq=False`` on the
    dataclasses additionally made the restored object compare unequal to its own
    source, which breaks any caching or de-duplication a caller might do.
    """
    rng = np.random.default_rng(5)
    prepared = prepare_eigenphases(
        np.sort(rng.uniform(0.0, 2.0 * np.pi, 48)),
        symmetry_sector="cue",
        degeneracy_tolerance=0.0,
    )
    unfolded = unfold(prepared, method="mean")
    histogram = spacing_distribution(unfolded, bins=8, value_range=(0.0, 4.0))

    for original, arrays in (
        (prepared, ("raw_phases", "phases", "spacings")),
        (unfolded, ("raw_phases", "phases", "values", "spacings")),
        (histogram, ("values", "bin_edges")),
    ):
        restored = roundtrip(original)
        assert restored == original, type(original).__name__
        assert restored is not original
        for name in arrays:
            np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
            assert not getattr(restored, name).flags.writeable, f"{original!r}.{name}"
        assert restored.metadata == original.metadata
        assert original.__eq__(object()) is NotImplemented

    assert roundtrip(prepared) != roundtrip(unfolded)
    assert roundtrip(unfolded) != unfold(prepared, method="polynomial", degree=3)


def test_adjacent_ratios_match_direct_circular_reference_and_degeneracy_policies() -> None:
    prepared = prepare_eigenphases([0.0, 1.0, 3.0, 5.0], symmetry_sector="all")
    spacings = prepared.spacings
    expected = np.minimum(spacings, np.roll(spacings, -1)) / np.maximum(
        spacings, np.roll(spacings, -1)
    )

    np.testing.assert_allclose(adjacent_gap_ratios(prepared).values, expected, atol=1e-14)

    with pytest.warns(NumericalWarning, match="detected 1 circular"):
        degenerate = prepare_eigenphases([0.0, 0.0, 2.0], symmetry_sector="all")
    assert adjacent_gap_ratios(degenerate, degeneracy="drop").values.size == 1
    assert np.count_nonzero(adjacent_gap_ratios(degenerate, degeneracy="zero").values == 0.0) == 2
    with pytest.raises(ValidationError, match="degenerate gaps"):
        adjacent_gap_ratios(degenerate, degeneracy="raise")


def test_spacing_distribution_has_exact_counts_and_normalized_density() -> None:
    phases = np.cumsum(np.asarray([0.0, 0.5, 1.0, 1.5]))
    prepared = prepare_eigenphases(phases, symmetry_sector="all")
    counts = spacing_distribution(prepared, bins=np.asarray([0.0, 0.75, 1.25, 4.0]), density=False)
    density = spacing_distribution(prepared, bins=8, value_range=(0.0, 4.0), density=True)

    assert int(np.sum(counts.values)) == counts.sample_count == 4
    assert np.sum(density.values * np.diff(density.bin_edges)) == pytest.approx(1.0, abs=1e-12)
    assert density.metadata.parameters["normalization"] == "unit_mean_spacing"

    outside = spacing_distribution(prepared, bins=4, value_range=(10.0, 11.0), density=True)
    np.testing.assert_array_equal(outside.values, np.zeros(4))


def test_density_is_normalized_on_the_total_sample_not_the_in_range_count() -> None:
    """A clipped range must lose mass, not be rescaled back up to one.

    Normalizing on the in-range count made every histogram integrate to one over
    whatever ``value_range`` was requested, inflating a truncated distribution
    (roughly 1.85x for ``(0, 1)`` on a 400-level CUE spectrum) and making it
    disagree with an unclipped Wigner surmise. The correct density integrates to
    ``included_count / sample_count``.
    """
    rng = np.random.default_rng(3)
    gaussian = rng.standard_normal((400, 400)) + 1j * rng.standard_normal((400, 400))
    unitary, triangular = np.linalg.qr(gaussian)
    diagonal = np.diag(triangular)
    unitary *= (diagonal / np.abs(diagonal)).conj()
    prepared = prepare_eigenphases(np.angle(np.linalg.eigvals(unitary)), symmetry_sector="cue")

    full = spacing_distribution(prepared, bins=40, value_range=(0.0, 8.0), density=True)
    clipped = spacing_distribution(prepared, bins=20, value_range=(0.0, 1.0), density=True)

    assert full.metadata.parameters["excluded_count"] == 0
    assert np.sum(full.values * np.diff(full.bin_edges)) == pytest.approx(1.0, abs=1e-12)
    assert not full.metadata.warnings

    included = clipped.metadata.parameters["included_count"]
    excluded = clipped.metadata.parameters["excluded_count"]
    assert isinstance(included, int)
    assert isinstance(excluded, int)
    assert excluded > 0
    assert included + excluded == clipped.sample_count == 400
    assert np.sum(clipped.values * np.diff(clipped.bin_edges)) == pytest.approx(
        included / 400, abs=1e-12
    )
    assert "spacings-outside-range" in [code.code for code in clipped.metadata.warnings]
    assert clipped.metadata.parameters["density_normalization"] == "total_sample_count"

    counts = spacing_distribution(prepared, bins=20, value_range=(0.0, 1.0), density=False)
    np.testing.assert_allclose(
        clipped.values, counts.values / (400.0 * np.diff(counts.bin_edges)), rtol=1e-13
    )


def test_seeded_poisson_circular_spectrum_has_expected_gap_ratio() -> None:
    rng = np.random.default_rng(124)
    spacings = rng.exponential(size=100_000)
    phases = 2.0 * np.pi * np.cumsum(np.concatenate(([0.0], spacings[:-1]))) / np.sum(spacings)
    prepared = prepare_eigenphases(phases, symmetry_sector="poisson")
    ratios = adjacent_gap_ratios(prepared)
    expected = 2.0 * np.log(2.0) - 1.0

    assert expected == pytest.approx(0.3862943611198906, rel=1e-15)
    # Observed deviation is 1.4e-4 here and stays below 8.3e-4 over seeds 1..2026.
    assert abs(float(np.mean(ratios.values)) - expected) <= 0.005


def test_seeded_cue_samples_have_expected_gap_ratio() -> None:
    rng = np.random.default_rng(125)
    samples: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    for _ in range(200):
        gaussian = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        unitary, triangular = np.linalg.qr(gaussian)
        diagonal = np.diag(triangular)
        unitary *= (diagonal / np.abs(diagonal)).conj()
        phases = np.angle(np.linalg.eigvals(unitary))
        prepared = prepare_eigenphases(phases, symmetry_sector="cue")
        samples.append(adjacent_gap_ratios(prepared).values)
    mean_ratio = float(np.mean(np.concatenate(samples)))

    # 200 matrices give a per-run scatter of about 3.1e-3 around the N=32 mean;
    # deviations from the bulk GUE value stayed below 5.0e-3 over seeds 1..2026.
    assert abs(mean_ratio - 0.60266) <= 0.012


def test_mean_gap_ratio_reference_separates_large_n_values_from_the_surmise() -> None:
    """The large-``N`` mean and the 3x3-surmise mean are different numbers.

    Conflating them is the mistake this function exists to prevent: the surmise
    overshoots the large-``N`` value by ``+0.0052`` (GOE), ``+0.0031`` (GUE) and
    ``+0.0018`` (GSE), which is several times the sampling error of a few hundred
    spectra, so using the surmise as the large-``N`` reference reads as a
    systematic bias.
    """
    exact_poisson = 2.0 * np.log(2.0) - 1.0

    assert mean_gap_ratio_reference("poisson") == exact_poisson
    assert mean_gap_ratio_reference("poisson", surmise=True) == exact_poisson
    assert exact_poisson == pytest.approx(0.3862943611198906, rel=1e-15)

    assert mean_gap_ratio_reference("goe") == 0.5307
    assert mean_gap_ratio_reference("gue") == 0.5996
    assert mean_gap_ratio_reference("gse") == 0.6744
    assert mean_gap_ratio_reference("goe", surmise=True) == 0.5359
    assert mean_gap_ratio_reference("gue", surmise=True) == 0.6027
    assert mean_gap_ratio_reference("gse", surmise=True) == 0.6762

    wigner_dyson: tuple[Ensemble, ...] = ("goe", "gue", "gse")
    for ensemble in wigner_dyson:
        assert mean_gap_ratio_reference(ensemble, surmise=True) > mean_gap_ratio_reference(ensemble)
    assert mean_gap_ratio_reference("goe", surmise=True) - 0.5307 == pytest.approx(0.0052, abs=1e-9)
    assert mean_gap_ratio_reference("gue", surmise=True) - 0.5996 == pytest.approx(0.0031, abs=1e-9)
    assert mean_gap_ratio_reference("gse", surmise=True) - 0.6744 == pytest.approx(0.0018, abs=1e-9)

    # Both columns must increase with beta: a table row copied into the wrong
    # ensemble would otherwise pass every individual equality above.
    for surmise in (False, True):
        column = [mean_gap_ratio_reference(name, surmise=surmise) for name in wigner_dyson]
        assert column == sorted(column)
        assert min(column) > mean_gap_ratio_reference("poisson", surmise=surmise)


@pytest.mark.parametrize(("alias", "canonical"), [("coe", "goe"), ("cue", "gue"), ("cse", "gse")])
@pytest.mark.parametrize("surmise", [False, True])
def test_circular_ensembles_alias_their_gaussian_gap_ratio(
    alias: Ensemble, canonical: Ensemble, surmise: bool
) -> None:
    """The ratio distribution only depends on the Dyson index, so COE == GOE."""
    assert mean_gap_ratio_reference(alias, surmise=surmise) == mean_gap_ratio_reference(
        canonical, surmise=surmise
    )


def test_mean_gap_ratio_reference_rejects_unknown_ensembles_and_non_bool_surmise() -> None:
    with pytest.raises(
        ValidationError, match=r"'poisson', 'goe', 'gue', 'cue', 'coe', 'gse', 'cse'"
    ):
        mean_gap_ratio_reference("cpe")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="surmise must be a bool"):
        mean_gap_ratio_reference("goe", surmise=1)  # type: ignore[arg-type]


def test_mean_gap_ratio_reference_matches_measured_haar_and_poisson_spectra() -> None:
    """The published constants must reproduce what this pipeline actually measures.

    200 Haar unitaries of dimension 64 give 12800 circular gap ratios. Measured
    over 25 seeds the run mean is ``0.6005 +/- 0.0024`` against the large-``N``
    CUE value ``0.5996``, worst deviation ``0.0053``. The same configuration on
    uniform spectra gives ``0.3871 +/- 0.0024`` against ``2*ln(2)-1``, worst
    deviation ``0.0057``. Larger Haar samples converge onto the same constant:
    40 matrices at ``N=256`` give ``0.5996 +/- 0.0031`` and 20 at ``N=512`` give
    ``0.5997 +/- 0.0023``. The 0.012 bound is about five standard errors.

    ``degeneracy_tolerance=0.0`` switches off the degeneracy diagnostic: a genuine
    uniform spectrum occasionally contains a gap below the 1e-10 default, which is
    real and harmless for a gap-ratio mean but would raise under the suite's
    error-on-warning policy.
    """
    rng = np.random.default_rng(126)
    cue_ratios: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    poisson_ratios: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    for _ in range(200):
        phases = np.angle(np.linalg.eigvals(_haar_unitary(rng, 64)))
        cue = prepare_eigenphases(phases, symmetry_sector="cue")
        cue_ratios.append(adjacent_gap_ratios(cue).values)
        uniform = prepare_eigenphases(
            rng.uniform(0.0, 2.0 * np.pi, 64),
            symmetry_sector="poisson",
            degeneracy_tolerance=0.0,
        )
        poisson_ratios.append(adjacent_gap_ratios(uniform).values)

    measured_cue = float(np.mean(np.concatenate(cue_ratios)))
    measured_poisson = float(np.mean(np.concatenate(poisson_ratios)))

    assert abs(measured_cue - mean_gap_ratio_reference("cue")) <= 0.012
    assert abs(measured_poisson - mean_gap_ratio_reference("poisson")) <= 0.012
    # The measurement must actually discriminate: Poisson and CUE are 0.21 apart,
    # far outside the bound, so a swapped reference table cannot pass.
    assert abs(measured_cue - mean_gap_ratio_reference("poisson")) > 0.15
    assert abs(measured_poisson - mean_gap_ratio_reference("cue")) > 0.15


def test_symplectic_mean_gap_ratio_requires_dropping_one_level_per_doublet() -> None:
    """The beta=4 reference is only reachable after the Kramers doublets are removed.

    Nothing in this library produces a symplectic spectrum, so the ensemble is
    generated here: ``U = V^D V`` with ``V`` Haar on ``U(2n)`` and ``X^D = J X^T
    J^-1`` the quaternion dual is self-dual, hence Kramers degenerate. Both members
    of a doublet come out within 1e-14 of each other, so sorting puts each pair at
    ``(2k, 2k+1)`` and ``phases[0::2]`` keeps one level per doublet.

    **What the doublets do to ``<r>`` is not a small bias, it is a different
    number.** With them in place every other circular gap is zero, so every
    adjacent-gap ratio has a vanishing numerator: the measured mean is ``1.3e-14``
    rather than 0.6744, and under the default ``degeneracy="drop"`` policy *every*
    ratio is discarded and the result is an empty array. Removing them recovers
    0.6661 to 0.6788 over seeds 4242, 11 and 808 (40 members at ``2N=128``,
    standard error 0.0055 per run), against the tabulated 0.6744.
    """
    rng = np.random.default_rng(4242)
    kept: list[float] = []
    with_doublets: list[float] = []
    for _ in range(40):
        unitary = _haar_unitary(rng, 128)
        symplectic = np.zeros_like(unitary)
        symplectic[:64, 64:] = np.eye(64)
        symplectic[64:, :64] = -np.eye(64)
        self_dual = (symplectic @ unitary.T @ (-symplectic)) @ unitary
        phases = np.sort(np.mod(np.angle(np.linalg.eigvals(self_dual)), 2.0 * np.pi))
        assert float(np.max(phases[1::2] - phases[0::2])) < 1e-10

        distinct = prepare_eigenphases(phases[0::2], symmetry_sector="cse")
        kept.append(float(np.mean(adjacent_gap_ratios(distinct).values)))

        # Preparation already flags this: half the circular gaps are exactly zero,
        # so the degeneracy diagnostic fires even at ``tolerance=0.0``.
        with pytest.warns(NumericalWarning, match="circular eigenphase gaps at or below 0.0"):
            degenerate = prepare_eigenphases(
                phases, symmetry_sector="cse", degeneracy_tolerance=0.0
            )
        with_doublets.append(
            float(np.mean(adjacent_gap_ratios(degenerate, degeneracy="zero", tolerance=0.0).values))
        )
        # The default policy throws every ratio away rather than reporting a number.
        assert adjacent_gap_ratios(degenerate).values.size == 0

    assert float(np.mean(kept)) == pytest.approx(mean_gap_ratio_reference("gse"), abs=0.02)
    assert float(np.mean(kept)) > mean_gap_ratio_reference("gue")
    assert float(np.mean(with_doublets)) <= 1e-10


@pytest.mark.parametrize(
    "operation",
    [
        lambda: prepare_eigenphases([[0.0, 1.0]], symmetry_sector="all"),
        lambda: unfold(
            prepare_eigenphases([0.0, 1.0], symmetry_sector="all"),
            method="bad",  # type: ignore[arg-type]
        ),
        lambda: spacing_distribution(
            prepare_eigenphases([0.0, 1.0], symmetry_sector="all"), bins=0
        ),
    ],
)
def test_spectral_parameter_validation(operation: object) -> None:
    with pytest.raises(ValidationError):
        operation()  # type: ignore[operator]


def _unnormalized_surmise(
    ratios: np.ndarray, index: float
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Atas et al. 3x3 surmise before normalization, written out independently."""
    return np.asarray(
        (ratios + ratios**2) ** index / (1.0 + ratios + ratios**2) ** (1.0 + 1.5 * index),
        dtype=np.float64,
    )


@pytest.mark.parametrize("ensemble", ["poisson", "goe", "coe", "gue", "cue", "gse", "cse"])
def test_gap_ratio_distribution_reference_reproduces_the_mean_gap_ratio_reference(
    ensemble: Ensemble,
) -> None:
    """One integral fixes both the normalization and the shape of ``P(r)``.

    ``adjacent_gap_ratios`` measures ``r_tilde = min(s_i,s_j)/max(s_i,s_j)``, so the
    reference is the density of that folded ratio on ``[0, 1]`` and its first moment
    has to equal ``mean_gap_ratio_reference(ensemble, surmise=True)``. Nothing else
    is asserted about ``Z_beta``: if the normalization were wrong the mass would
    move, and if the exponent structure were wrong the mean would, so requiring both
    at once leaves no freedom.

    Measured on a two-million-point grid: mass ``1.000000000000`` for every
    ensemble, and ``int r P(r) dr`` equal to ``0.3862943611`` (Poisson, matching
    ``2*ln(2)-1`` to 4e-14), ``0.5358983849`` against the published ``0.5359``, and
    ``0.6026577908`` against ``0.6027``, and ``0.6761683102`` against ``0.6762``.
    The beta=1 and beta=2 references are literature constants quoted to four
    decimals, which is why the tolerance is ``1e-4`` and not tighter; the beta=4
    one was produced *by* this integral, so for GSE the check is that the table
    still carries the rounded value of the curve.
    """
    grid = np.linspace(0.0, 1.0, 2_000_001)
    density = np.asarray(rmt_reference("gap_ratio_distribution", ensemble, grid).values)

    assert float(trapezoid(density, grid)) == pytest.approx(1.0, rel=1e-9)
    assert float(trapezoid(grid * density, grid)) == pytest.approx(
        mean_gap_ratio_reference(ensemble, surmise=True), abs=1e-4
    )


@pytest.mark.parametrize("ensemble", ["goe", "gue", "gse"])
def test_gap_ratio_distribution_normalization_matches_the_closed_form(
    ensemble: Ensemble,
) -> None:
    """``Z_1 = 8/27``, ``Z_2 = 4*pi/(81*sqrt(3))``, ``Z_4 = 4*pi/(729*sqrt(3))``: checked.

    The implementation carries the reciprocals as literals; this test recovers them
    by integrating the *unnormalized* surmise over ``(0, inf)`` with adaptive
    quadrature and comparing against the ratio the reference curve implies. All
    three agree to better than 1e-12 relative, the beta=4 one to 3.3e-16.
    """
    index = {"goe": 1.0, "gue": 2.0, "gse": 4.0}[ensemble]
    closed_form = {
        "goe": 8.0 / 27.0,
        "gue": 4.0 * np.pi / (81.0 * np.sqrt(3.0)),
        "gse": 4.0 * np.pi / (729.0 * np.sqrt(3.0)),
    }[ensemble]
    integral = quad(_unnormalized_surmise, 0.0, 1.0, args=(index,), epsabs=1e-14, epsrel=1e-14)[0]
    integral += quad(_unnormalized_surmise, 1.0, np.inf, args=(index,), epsabs=1e-14, epsrel=1e-14)[
        0
    ]

    assert integral == pytest.approx(closed_form, rel=1e-12)

    probe = np.asarray([0.1, 0.4, 0.75, 1.0])
    unfolded = np.asarray(
        rmt_reference("gap_ratio_distribution", ensemble, probe, folded=False).values
    )
    np.testing.assert_allclose(
        unfolded, _unnormalized_surmise(probe, index) / closed_form, rtol=1e-14
    )


def test_gap_ratio_distribution_folds_through_the_inversion_symmetry() -> None:
    """``P(1/r)/r^2 = P(r)`` is what makes the folded curve twice the unfolded one.

    The identity holds for the surmise at every ``beta`` and for the exact Poisson
    density, so the two branches of ``r_tilde = min(r, 1/r)`` coincide and folding
    multiplies by exactly two below ``r=1``. Above ``r=1`` the folded density is
    zero, because ``min(r, 1/r)`` cannot land there -- it is returned as zero rather
    than rejected so that a histogram grid may run slightly past the support.
    """
    inside = np.asarray([0.05, 0.3, 0.6, 1.0])
    outside = np.asarray([1.0000001, 1.5, 4.0])
    for ensemble in ("poisson", "goe", "gue"):
        unfolded = np.asarray(
            rmt_reference("gap_ratio_distribution", ensemble, inside, folded=False).values
        )
        folded = np.asarray(rmt_reference("gap_ratio_distribution", ensemble, inside).values)
        mirrored = np.asarray(
            rmt_reference("gap_ratio_distribution", ensemble, 1.0 / inside, folded=False).values
        )

        np.testing.assert_allclose(folded, 2.0 * unfolded, rtol=1e-14)
        np.testing.assert_allclose(mirrored / inside**2, unfolded, rtol=1e-13)
        np.testing.assert_array_equal(
            rmt_reference("gap_ratio_distribution", ensemble, outside).values,
            np.zeros_like(outside),
        )

        # The unfolded density carries unit mass on (0, inf) as well; the Poisson
        # tail decays only as 1/r^2, so a finite grid leaves 1/(1+R) behind.
        wide = np.linspace(0.0, 4000.0, 2_000_001)
        tail = np.asarray(
            rmt_reference("gap_ratio_distribution", ensemble, wide, folded=False).values
        )
        assert float(trapezoid(tail, wide)) == pytest.approx(1.0, abs=3e-4)

    assert rmt_reference("gap_ratio_distribution", "gue", inside).metadata.parameters["folded"]
    assert (
        rmt_reference("gap_ratio_distribution", "gue", inside, folded=False).metadata.parameters[
            "folded"
        ]
        is False
    )


def test_gap_ratio_distribution_poisson_is_exact_and_not_the_beta_zero_surmise() -> None:
    """Poisson is ``2/(1+r)^2``, which is *not* what ``beta -> 0`` gives.

    It is often said that the Atas surmise reduces to the Poisson ratio density as
    ``beta -> 0``. It does not. Setting ``beta=0`` leaves ``1/(1+r+r^2)``, whose
    normalization on ``(0, inf)`` is ``Z_0 = 2*pi/(3*sqrt(3)) = 1.209200`` and whose
    folded mean is ``0.408545`` -- 5.8% above the exact Poisson
    ``2*ln(2)-1 = 0.386294``, and far outside the ``+/-1e-4`` that
    ``mean_gap_ratio_reference`` claims. The reference therefore uses the exact
    ratio density of independent exponential gaps for ``"poisson"``, and this test
    pins the difference so that nobody "simplifies" the implementation by taking the
    limit.
    """
    grid = np.linspace(0.0, 1.0, 2_000_001)
    folded = np.asarray(rmt_reference("gap_ratio_distribution", "poisson", grid).values)

    np.testing.assert_allclose(folded, 2.0 / (1.0 + grid) ** 2, rtol=1e-15)
    assert float(trapezoid(grid * folded, grid)) == pytest.approx(2.0 * np.log(2.0) - 1.0, abs=1e-9)

    limit_normalization = 2.0 * np.pi / (3.0 * np.sqrt(3.0))
    assert limit_normalization == pytest.approx(1.2091995761561452, rel=1e-14)
    limit = 2.0 * _unnormalized_surmise(grid, 0.0) / limit_normalization
    assert float(trapezoid(limit, grid)) == pytest.approx(1.0, rel=1e-9)
    limit_mean = float(trapezoid(grid * limit, grid))
    assert limit_mean == pytest.approx(0.408545, abs=1e-5)
    assert abs(limit_mean - (2.0 * np.log(2.0) - 1.0)) == pytest.approx(0.02225, abs=1e-4)


def test_gap_ratio_distribution_shows_the_expected_level_repulsion_powers() -> None:
    """``P(r) ~ r^beta`` at small ``r``: ``r``, ``r^2``, ``r^4`` for beta=1,2,4."""
    tiny = np.asarray([1e-4, 2e-4])
    poisson = np.asarray(rmt_reference("gap_ratio_distribution", "poisson", tiny).values)

    for ensemble, index in (("goe", 1.0), ("gue", 2.0), ("gse", 4.0)):
        density = np.asarray(
            rmt_reference("gap_ratio_distribution", ensemble, tiny).values  # type: ignore[arg-type]
        )
        assert float(density[1] / density[0]) == pytest.approx(2.0**index, rel=1e-3)

    # Poisson does not repel: the density is finite and equal to 2 at the origin.
    assert float(poisson[0]) == pytest.approx(2.0, rel=1e-3)


@pytest.mark.parametrize(("label", "other"), [("cue", "goe"), ("coe", "gue")])
def test_measured_gap_ratio_histograms_track_their_own_surmise(
    label: Ensemble, other: Ensemble
) -> None:
    """A histogram of measured ratios must sit on its own reference within counting noise.

    80 circular-ensemble members of dimension 128 give 10240 ratios, so a width-0.05
    density bin carries about 0.05 of counting noise near the peak. The maximum
    deviation over the 20 bins runs 0.047-0.115 across four seeds, i.e. 0.9-2.6
    times that per-bin sigma -- consistent with the ~2.2 expected maximum of 20
    standard normals, so the comparison is noise-dominated and says nothing about
    how good the surmise is. Resolving the 3x3 surmise against the exact ratio
    density would need a far larger sample, and this library does not ship the exact
    density.

    What makes the test discriminating is the other two assertions: the same
    histogram misses the *other* Dyson index by at least 0.42 and the Poisson curve
    by at least 1.71, so a swapped ``beta``, a missing factor of two from the fold,
    or a wrong ``Z_beta`` cannot pass.
    """
    rng = np.random.default_rng(321)
    edges = np.linspace(0.0, 1.0, 21)
    densities: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    for _ in range(80):
        unitary = _haar_unitary(rng, 128)
        if label == "coe":
            unitary = unitary.T @ unitary
        prepared = prepare_eigenphases(
            np.angle(np.linalg.eigvals(unitary)),
            symmetry_sector=label,
            degeneracy_tolerance=0.0,
        )
        ratios = np.asarray(adjacent_gap_ratios(prepared).values)
        counts, _ = np.histogram(ratios, bins=edges)
        densities.append(counts / (ratios.size * np.diff(edges)))
    measured = np.mean(densities, axis=0)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def deviation(ensemble: Ensemble) -> float:
        reference = np.asarray(rmt_reference("gap_ratio_distribution", ensemble, centers).values)
        return float(np.max(np.abs(measured - reference)))

    assert deviation(label) <= 0.16
    assert deviation(other) > 0.3
    assert deviation("poisson") > 1.0


def test_mean_unfolding_anchors_the_first_level_at_zero() -> None:
    """``values`` is a public array and its origin is part of the convention.

    ``unfold(method="mean")`` subtracts the first phase before dividing by the mean
    spacing, so ``values[0]`` is exactly zero and ``values`` counts mean spacings
    from the first level. Nothing pinned that: every consumer in the library is
    shift-invariant -- the form factor centres the levels itself, the number
    variance and the rigidity wrap onto a circle -- so dropping the subtraction
    changed a documented public array while the whole suite stayed green.
    """
    for offset in (0.0, 1.0, 3.5):
        phases = np.mod(offset + np.arange(12) * 2.0 * np.pi / 12.0, 2.0 * np.pi)
        prepared = prepare_eigenphases(
            np.sort(phases), symmetry_sector="rigid", degeneracy_tolerance=0.0
        )
        result = unfold(prepared, method="mean")

        assert float(result.values[0]) == 0.0
        # And the scale really is the mean spacing: a rigid spectrum unfolds to the
        # integers, which also fixes that the divisor is 2*pi/count.
        np.testing.assert_allclose(result.values, np.arange(12.0), rtol=0.0, atol=1e-12)

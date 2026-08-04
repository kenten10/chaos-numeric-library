from __future__ import annotations

import numpy as np
import pytest

from chaos_numerics.core import NumericalWarning, ValidationError
from chaos_numerics.spectral import (
    adjacent_gap_ratios,
    prepare_eigenphases,
    spacing_distribution,
    unfold,
)


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


def test_seeded_poisson_circular_spectrum_has_expected_gap_ratio() -> None:
    rng = np.random.default_rng(124)
    spacings = rng.exponential(size=100_000)
    phases = 2.0 * np.pi * np.cumsum(np.concatenate(([0.0], spacings[:-1]))) / np.sum(spacings)
    prepared = prepare_eigenphases(phases, symmetry_sector="poisson")
    ratios = adjacent_gap_ratios(prepared)
    expected = 2.0 * np.log(2.0) - 1.0

    assert abs(float(np.mean(ratios.values)) - expected) <= 0.02


def test_seeded_cue_samples_have_expected_gap_ratio() -> None:
    rng = np.random.default_rng(125)
    samples: list[np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = []
    for _ in range(50):
        gaussian = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        unitary, triangular = np.linalg.qr(gaussian)
        diagonal = np.diag(triangular)
        unitary *= (diagonal / np.abs(diagonal)).conj()
        phases = np.angle(np.linalg.eigvals(unitary))
        prepared = prepare_eigenphases(phases, symmetry_sector="cue")
        samples.append(adjacent_gap_ratios(prepared).values)
    mean_ratio = float(np.mean(np.concatenate(samples)))

    assert abs(mean_ratio - 0.60266) <= 0.02


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

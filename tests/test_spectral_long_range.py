from __future__ import annotations

import numpy as np
import pytest

from chaos_numerics.core import ValidationError
from chaos_numerics.spectral import (
    number_variance,
    prepare_eigenphases,
    rmt_reference,
    spectral_form_factor,
    unfold,
)


def test_spectral_form_factor_matches_direct_complex_sum() -> None:
    prepared = prepare_eigenphases([0.0, 0.7, 2.2, 5.0], symmetry_sector="all")
    unfolded = unfold(prepared, method="mean")
    times = np.asarray([0.0, 0.125, 0.5, 1.0])
    centered = unfolded.values - np.mean(unfolded.values)
    expected = np.abs(np.exp(2j * np.pi * np.outer(times, centered)).sum(axis=1)) ** 2 / 4

    result = spectral_form_factor(unfolded, times, connected=False)

    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(result.x, times)
    assert result.metadata.parameters["normalization"] == "sum(w^2); plateau=1"
    assert not result.values.flags.writeable


def test_connected_sff_bootstrap_is_seeded_and_returns_plot_ready_uncertainty() -> None:
    prepared = prepare_eigenphases(np.linspace(0.0, 2.0 * np.pi, 65)[:-1], symmetry_sector="all")
    times = np.linspace(0.0, 1.5, 12)

    first = spectral_form_factor(
        prepared,
        times,
        window="hann",
        bootstrap=32,
        seed=7,
    )
    second = spectral_form_factor(
        prepared,
        times,
        window="hann",
        bootstrap=32,
        seed=7,
    )

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.uncertainty, second.uncertainty)
    np.testing.assert_array_equal(first.variance, second.variance)
    assert first.values[0] == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(ValidationError, match="seed is required"):
        spectral_form_factor(prepared, times, bootstrap=2)


def test_number_variance_matches_direct_circular_window_counting() -> None:
    prepared = prepare_eigenphases([0.1, 1.0, 2.0, 4.0, 5.5], symmetry_sector="all")
    unfolded = unfold(prepared, method="mean")
    lengths = np.asarray([0.0, 0.75, 2.0, 4.0])
    samples = 256
    result = number_variance(unfolded, lengths, samples=samples)

    levels = np.sort(np.mod(unfolded.values, unfolded.count))
    origins = np.linspace(0.0, unfolded.count, samples, endpoint=False)
    doubled = np.concatenate((levels, levels + unfolded.count))
    starts = np.searchsorted(levels, origins, side="left")
    expected = []
    for length in lengths:
        counts = np.searchsorted(doubled, origins + length, side="left") - starts
        expected.append(np.mean((counts - length) ** 2))

    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-14)
    assert result.uncertainty is not None
    assert result.variance is not None


def test_number_variance_bootstrap_and_finite_size_correction() -> None:
    prepared = prepare_eigenphases(np.linspace(0.0, 2.0 * np.pi, 33)[:-1], symmetry_sector="all")
    raw = number_variance(prepared, [1.5, 4.0], samples=128, bootstrap=24, seed=9)
    corrected = number_variance(
        prepared,
        [1.5, 4.0],
        samples=128,
        finite_size_correction=True,
    )

    np.testing.assert_allclose(
        corrected.values,
        raw.values / (1.0 - np.asarray([1.5, 4.0]) / prepared.count),
        rtol=1e-12,
    )
    assert raw.uncertainty is not None
    assert bool(np.all(raw.uncertainty >= 0.0))


def test_rmt_form_factor_and_number_variance_references() -> None:
    times = np.asarray([0.0, 0.25, 0.5, 1.0, 2.0])
    poisson = rmt_reference("spectral_form_factor", "poisson", times)
    gue = rmt_reference("spectral_form_factor", "gue", times)
    cue = rmt_reference("spectral_form_factor", "cue", times, dimension=64)
    goe = rmt_reference("spectral_form_factor", "goe", times)

    np.testing.assert_array_equal(poisson.values, [0.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(gue.values, np.minimum(times, 1.0))
    np.testing.assert_array_equal(cue.values, gue.values)
    assert bool(np.all(goe.values[1:3] >= gue.values[1:3]))
    assert bool(np.all(goe.values[3:] <= gue.values[3:]))

    lengths = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0])
    poisson_variance = rmt_reference("number_variance", "poisson", lengths)
    goe_variance = rmt_reference("number_variance", "goe", lengths)
    gue_variance = rmt_reference("number_variance", "gue", lengths)
    cue_variance = rmt_reference("number_variance", "cue", lengths, dimension=64)

    np.testing.assert_array_equal(poisson_variance.values, lengths)
    assert bool(np.all(goe_variance.values[1:] > gue_variance.values[1:]))
    np.testing.assert_allclose(cue_variance.values, gue_variance.values, rtol=0.02, atol=1e-3)


def test_seeded_poisson_number_variance_has_expected_trend() -> None:
    rng = np.random.default_rng(1250)
    spacings = rng.exponential(size=50_000)
    phases = 2.0 * np.pi * np.cumsum(np.concatenate(([0.0], spacings[:-1]))) / np.sum(spacings)
    spectrum = unfold(prepare_eigenphases(phases, symmetry_sector="poisson"))
    lengths = np.asarray([1.0, 2.0, 4.0, 8.0])
    result = number_variance(spectrum, lengths, samples=8192)

    np.testing.assert_allclose(result.values, lengths, rtol=0.08, atol=0.08)


def test_seeded_cue_sff_follows_ramp_and_poisson_is_flat() -> None:
    rng = np.random.default_rng(1251)
    dimension = 24
    times = np.asarray([0.25, 0.5, 1.0])
    cue_samples = []
    poisson_samples = []
    for _ in range(80):
        gaussian = rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal(
            (dimension, dimension)
        )
        unitary, triangular = np.linalg.qr(gaussian)
        diagonal = np.diag(triangular)
        unitary *= (diagonal / np.abs(diagonal)).conj()
        cue = prepare_eigenphases(np.angle(np.linalg.eigvals(unitary)), symmetry_sector="cue")
        cue_samples.append(spectral_form_factor(cue, times).values)

        poisson = prepare_eigenphases(
            rng.uniform(0.0, 2.0 * np.pi, dimension), symmetry_sector="poisson"
        )
        poisson_samples.append(spectral_form_factor(poisson, times).values)

    cue_mean = np.mean(cue_samples, axis=0)
    poisson_mean = np.mean(poisson_samples, axis=0)

    np.testing.assert_allclose(cue_mean, np.minimum(times, 1.0), rtol=0.25, atol=0.12)
    np.testing.assert_allclose(poisson_mean, np.ones_like(times), rtol=0.25, atol=0.12)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: rmt_reference("bad", "gue", [1.0]),  # type: ignore[arg-type]
        lambda: rmt_reference("number_variance", "bad", [1.0]),  # type: ignore[arg-type]
        lambda: number_variance(prepare_eigenphases([0.0, 1.0], symmetry_sector="all"), [3.0]),
        lambda: spectral_form_factor(
            prepare_eigenphases([0.0, 1.0], symmetry_sector="all"), [1.0], window="hann"
        ),
    ],
)
def test_long_range_parameter_validation(operation: object) -> None:
    with pytest.raises(ValidationError):
        operation()  # type: ignore[operator]

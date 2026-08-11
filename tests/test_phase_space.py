from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from chaos_numerics.core import (
    ConvergenceInfo,
    ExperimentMetadata,
    SerializableResult,
    ValidationError,
)
from chaos_numerics.quantum import (
    BoundaryPhases,
    HusimiResult,
    KickedRotor,
    WignerResult,
    basis_state,
    coherent_state,
    husimi_distribution,
    inverse_participation_ratio,
    normalize_state,
    participation_ratio,
    phase_space,
    shannon_entropy,
    wigner_distribution,
)


@pytest.mark.parametrize("dimension", [2, 8, 31, 32])
def test_periodized_coherent_state_is_normalized_and_periodic(dimension: int) -> None:
    phases = BoundaryPhases(0.5, 0.25)
    state = coherent_state(
        dimension=dimension,
        position=0.97,
        momentum=0.13,
        boundary_phases=phases,
    )
    translated = coherent_state(
        dimension=dimension,
        position=1.97,
        momentum=-0.87,
        boundary_phases=phases,
    )

    assert state.dtype == np.dtype(np.complex128)
    assert abs(np.linalg.norm(state) - 1.0) <= 1e-12
    np.testing.assert_allclose(translated, state, rtol=1e-13, atol=1e-14)


def test_coherent_state_matches_independent_periodized_sum() -> None:
    dimension = 6
    position = 0.2
    momentum = 0.7
    phases = BoundaryPhases(0.25, 0.5)
    q_basis = (np.arange(dimension) + phases.position) / dimension
    expected = np.zeros(dimension, dtype=np.complex128)
    for index, q_value in enumerate(q_basis):
        for image in range(-4, 5):
            displacement = q_value - position + image
            expected[index] += (
                np.exp(-np.pi * dimension * displacement**2)
                * np.exp(2j * np.pi * dimension * momentum * displacement)
                * np.exp(-2j * np.pi * phases.momentum * image)
            )
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(
        coherent_state(
            dimension=dimension,
            position=position,
            momentum=momentum,
            boundary_phases=phases,
        ),
        expected,
        rtol=1e-13,
        atol=1e-14,
    )


def test_husimi_is_normalized_plot_ready_and_supports_batches() -> None:
    states = np.stack(
        (
            coherent_state(dimension=16, position=0.25, momentum=0.75),
            basis_state(dimension=16, index=4),
        )
    )
    result = husimi_distribution(states, grid_shape=(24, 20), chunk_size=37)

    assert result.values.shape == (2, 24, 20)
    assert result.grid_shape == (24, 20)
    assert result.grid_points.shape == (24, 20, 2)
    # ``integral == 1`` is an algebraic identity once ``normalize=True`` divided
    # by the mean, so it cannot detect a wrong quadrature. It is kept only as a
    # consistency check on the normalization step itself.
    np.testing.assert_allclose(result.integral, np.ones(2), atol=1e-14)
    assert result.raw_integral.shape == (2,)
    assert result.cell_area == pytest.approx(1.0 / (24 * 20))
    assert bool(np.all(result.grid_points[..., 0] == result.positions[:, None]))
    assert bool(np.all(result.grid_points[..., 1] == result.momenta[None, :]))
    assert not result.values.flags.writeable

    # The substantive statement is that the *unnormalized* midpoint quadrature
    # already integrates to one on a grid that resolves the state.
    raw = husimi_distribution(states, grid_shape=(24, 20), chunk_size=37, normalize=False)
    np.testing.assert_allclose(raw.raw_integral, np.ones(2), atol=1e-13)


def test_raw_husimi_integral_detects_an_unresolved_grid() -> None:
    """Show the ``raw_integral`` check has teeth that ``integral`` does not.

    A (3, 5) grid cannot resolve either state, and ``raw_integral`` says so,
    while the normalized ``integral`` is still exactly one by construction.
    """
    states = np.stack(
        (
            coherent_state(dimension=16, position=0.25, momentum=0.75),
            basis_state(dimension=16, index=4),
        )
    )

    coarse = husimi_distribution(states, grid_shape=(3, 5), normalize=False)
    normalized = husimi_distribution(states, grid_shape=(3, 5))

    np.testing.assert_allclose(coarse.raw_integral, [0.99829106, 0.94164262], atol=1e-8)
    assert abs(float(coarse.raw_integral[1]) - 1.0) > 5e-2
    np.testing.assert_allclose(normalized.integral, np.ones(2), atol=1e-14)


def test_husimi_raw_quadrature_converges_with_grid_refinement() -> None:
    phases = BoundaryPhases(0.5, 0.5)
    state = coherent_state(
        dimension=32,
        position=0.97,
        momentum=0.13,
        boundary_phases=phases,
    )
    coarse = husimi_distribution(
        state,
        grid_shape=(8, 8),
        boundary_phases=phases,
        normalize=False,
    )
    fine = husimi_distribution(
        state,
        grid_shape=(32, 32),
        boundary_phases=phases,
        normalize=False,
    )

    coarse_error = abs(float(coarse.raw_integral) - 1.0)
    fine_error = abs(float(fine.raw_integral) - 1.0)
    assert fine_error < coarse_error
    # Measured: 3.9e-3 on the (8, 8) grid and 0.0 on (32, 32).
    assert coarse_error >= 1e-3
    assert fine_error <= 1e-13


@pytest.mark.parametrize("dimension", [64, 128, 256])
def test_coherent_state_localization_matches_closed_form(dimension: int) -> None:
    """Check the Gaussian width against two independent closed forms.

    A minimum-uncertainty torus Gaussian of width ``1/sqrt(N)`` in each
    coordinate has participation ratio ``sqrt(N)`` and Shannon entropy
    ``ln(e N / 2) / 2`` in the position basis. Both would move if the Gaussian
    exponent, the ``hbar_eff = 1 / N`` scale, or the normalization were wrong.
    """
    state = coherent_state(dimension=dimension, position=0.25, momentum=0.75)

    assert participation_ratio(state) == pytest.approx(np.sqrt(dimension), rel=1e-14)
    assert shannon_entropy(state) == pytest.approx(0.5 * np.log(np.e * dimension / 2.0), abs=1e-14)
    assert inverse_participation_ratio(state) == pytest.approx(1.0 / np.sqrt(dimension), rel=1e-14)


def test_husimi_peak_sits_on_the_requested_phase_space_point() -> None:
    """Detect a swapped ``(q, p)`` order or a flipped momentum sign.

    The peak must sit within one cell of the requested centre. It is **not**
    pinned to a particular cell, because on this grid it cannot be: a 64x64 grid
    has midpoints at ``(j + 0.5) / 64``, so the requested ``0.25`` falls exactly
    halfway between ``0.2421875`` and ``0.2578125``, and the two cells hold
    bit-identical values here. Which one ``argmax`` returns is then decided by the
    implementation rather than by the physics, and this test previously asserted
    ``0.2421875`` and failed on the NumPy 1.26 floor, which returned the other one.

    One cell of slack still discriminates what the test is for by a wide margin:
    swapping the axes would report ``(0.7421875, 0.2421875)`` and negating the
    momentum ``(0.2421875, 0.2421875)``, both about half a period away.
    """
    result = husimi_distribution(
        coherent_state(dimension=64, position=0.25, momentum=0.75),
        grid_shape=(64, 64),
    )

    peak = np.unravel_index(int(np.argmax(result.values)), result.values.shape)
    spacing = 1.0 / 64.0
    assert abs(float(result.positions[peak[0]]) - 0.25) <= spacing
    assert abs(float(result.momenta[peak[1]]) - 0.75) <= spacing


def test_localization_measures_for_basis_uniform_and_batch_states() -> None:
    localized = basis_state(dimension=8, index=3)
    uniform = normalize_state(np.ones(8))
    batch = np.stack((localized, uniform))

    assert inverse_participation_ratio(localized) == pytest.approx(1.0, abs=1e-14)
    assert participation_ratio(localized) == pytest.approx(1.0, abs=1e-14)
    assert shannon_entropy(localized) == pytest.approx(0.0, abs=1e-14)
    assert inverse_participation_ratio(uniform) == pytest.approx(1.0 / 8.0, rel=1e-12)
    assert participation_ratio(uniform) == pytest.approx(8.0, rel=1e-12)
    assert shannon_entropy(uniform) == pytest.approx(np.log(8.0), abs=1e-12)
    np.testing.assert_allclose(inverse_participation_ratio(batch), [1.0, 1.0 / 8.0])
    np.testing.assert_allclose(participation_ratio(batch), [1.0, 8.0])
    np.testing.assert_allclose(shannon_entropy(batch), [0.0, np.log(8.0)], atol=1e-14)


@pytest.mark.parametrize(
    "roundtrip", [lambda item: pickle.loads(pickle.dumps(item)), copy.deepcopy]
)
def test_husimi_result_survives_pickle_and_deepcopy(roundtrip: object) -> None:
    """A parallel sweep has to be able to ship this container between processes.

    ``slots=True`` gives the dataclass a field-by-field reduction that skips
    ``__post_init__``, so a round trip used to hand back **writable** arrays and
    break the read-only contract that every other result type keeps. ``eq=False``
    meant the restored object also compared unequal to the original, which made
    the breakage invisible to any test that only checked equality.
    """
    original = husimi_distribution(
        coherent_state(dimension=8, position=0.3, momentum=0.7),
        grid_shape=(6, 5),
    )
    assert callable(roundtrip)
    restored = roundtrip(original)

    assert isinstance(restored, HusimiResult)
    assert type(restored) is HusimiResult
    assert restored == original
    for array in (
        restored.positions,
        restored.momenta,
        restored.values,
        restored.raw_integral,
    ):
        assert not array.flags.writeable
    assert restored.metadata.to_dict() == original.metadata.to_dict()
    assert restored.grid_shape == (6, 5)
    np.testing.assert_array_equal(restored.values, original.values)


def test_husimi_result_equality_is_value_based() -> None:
    """Same numbers means equal, one different number means not equal."""
    state = coherent_state(dimension=8, position=0.3, momentum=0.7)
    original = husimi_distribution(state, grid_shape=(6, 5))
    identical = husimi_distribution(state, grid_shape=(6, 5))
    different = husimi_distribution(
        coherent_state(dimension=8, position=0.4, momentum=0.7),
        grid_shape=(6, 5),
    )

    assert original == identical
    assert original != different
    assert original != object()
    # Neither side of the comparison may accidentally succeed by identity alone.
    assert original is not identical


@pytest.mark.parametrize(
    "operation",
    [
        lambda: coherent_state(dimension=0, position=0.0, momentum=0.0),
        lambda: coherent_state(dimension=4, position=np.nan, momentum=0.0),
        lambda: coherent_state(dimension=4, position=0.0, momentum=0.0, images=0),
        lambda: husimi_distribution([1.0, 0.0], grid_shape=(0, 4)),
        lambda: husimi_distribution([1.0, 0.0], grid_shape=[4, 4]),  # type: ignore[arg-type]
        lambda: inverse_participation_ratio([1.0, 1.0]),
    ],
)
def test_phase_space_parameter_validation(operation: object) -> None:
    with pytest.raises(ValidationError):
        operation()  # type: ignore[operator]


def _wigner_reference(state: np.ndarray, *, beta: float) -> np.ndarray:
    """Transcribe the half-integer definition directly, in ``O(N**3)``.

    Written from ``docs/design/quantum-phase-space.md`` rather than from the
    implementation, and left complex so that the realness of the result is a
    measured property instead of an assumption.
    """
    size = state.shape[-1]
    weyl = np.zeros((2 * size, 2 * size), dtype=np.complex128)
    for midpoint in range(2 * size):
        for momentum_index in range(2 * size):
            total = 0.0 + 0.0j
            for left in range(size):
                partner = midpoint - left
                extended = state[partner % size] * np.exp(2j * np.pi * beta * (partner // size))
                total += (
                    state[left]
                    * np.conj(extended)
                    * np.exp(
                        -2j * np.pi * (momentum_index / 2.0 + beta) * (2 * left - midpoint) / size
                    )
                )
            weyl[midpoint, momentum_index] = total / (2 * size)
    return weyl


@pytest.mark.parametrize("dimension", [2, 4, 6, 8, 16])
@pytest.mark.parametrize(
    "phases",
    [
        BoundaryPhases(),
        BoundaryPhases(0.25, 0.13),
        BoundaryPhases(0.5, 0.5),
        BoundaryPhases(0.0, 0.5),
        BoundaryPhases(0.5, 0.0),
    ],
)
def test_wigner_matches_the_definition_and_is_real(dimension: int, phases: BoundaryPhases) -> None:
    """Pin the convention against an independent transcription of the formula.

    The reference builds the ``2N x 2N`` half-integer lattice; the library
    returns its ``2 x 2`` block sums. Measured worst case over this grid is
    2.5e-16 for the values and 3.9e-16 for the discarded imaginary part. The
    reference also fixes the ghost identity ``A[m + N, n] = (-1)**n A[m, n]``
    that the block sum exists to cancel: worst case 3.6e-16.
    """
    rng = np.random.default_rng(500 + dimension)
    state = normalize_state(rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension))
    weyl = _wigner_reference(state, beta=phases.momentum)

    result = wigner_distribution(state, boundary_phases=phases)

    assert float(np.max(np.abs(weyl.imag))) <= 1e-15
    blocked = weyl.real.reshape(dimension, 2, dimension, 2).sum(axis=(1, 3))
    np.testing.assert_allclose(result.values, blocked, atol=1e-15)
    assert result.values.dtype == np.dtype(np.float64)
    assert result.grid_shape == (dimension, dimension)

    signs = np.where(np.arange(2 * dimension) % 2 == 0, 1.0, -1.0)
    np.testing.assert_allclose(np.roll(weyl.real, dimension, axis=0), signs * weyl.real, atol=1e-15)


@pytest.mark.parametrize("dimension", [2, 4, 6, 8, 16, 32])
@pytest.mark.parametrize("phases", [BoundaryPhases(), BoundaryPhases(0.25, 0.13)])
def test_wigner_marginals_are_exact_in_both_bases(dimension: int, phases: BoundaryPhases) -> None:
    """The decisive check: both marginals, exactly, on the same lattice.

    Summing over ``p`` must give ``abs(psi_j)**2`` and summing over ``q`` must
    give the momentum distribution produced by
    :meth:`KickedRotor.to_momentum_basis` in its own storage order. Measured
    worst case 2.2e-16 and 5.6e-16 over the tested grid; essentially any sign,
    factor, index, or missing-twist error breaks one of the two.
    """
    rng = np.random.default_rng(900 + dimension)
    state = normalize_state(rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension))
    rotor = KickedRotor(dimension, 1.0, boundary_phases=phases)

    result = wigner_distribution(state, boundary_phases=phases)
    values = np.asarray(result.values)

    np.testing.assert_allclose(values.sum(axis=-1), np.abs(state) ** 2, atol=1e-14)
    np.testing.assert_allclose(
        values.sum(axis=-2),
        np.abs(rotor.to_momentum_basis(state)) ** 2,
        atol=1e-14,
    )
    assert float(result.total) == pytest.approx(1.0, abs=1e-14)
    # The axes are the model's own grids, so a marginal can be plotted against
    # them without a second convention.
    np.testing.assert_allclose(
        result.positions, (np.arange(dimension) + phases.position) / dimension, atol=1e-15
    )
    np.testing.assert_allclose(
        result.momenta, (np.arange(dimension) + phases.momentum) / dimension, atol=1e-15
    )


@pytest.mark.parametrize("dimension", [4, 8, 16])
def test_wigner_quadratic_form_reproduces_the_squared_overlap(dimension: int) -> None:
    """``N * sum W_psi W_phi == abs(<psi|phi>)**2``, so purity reads exactly 1.

    The factor is ``1 / N`` and nothing else; measured worst case 2.2e-16.
    """
    rng = np.random.default_rng(31 + dimension)
    phases = BoundaryPhases(0.25, 0.13)
    left = normalize_state(rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension))
    right = normalize_state(rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension))

    first = np.asarray(wigner_distribution(left, boundary_phases=phases).values)
    second = np.asarray(wigner_distribution(right, boundary_phases=phases).values)

    assert dimension * float(np.sum(first * second)) == pytest.approx(
        abs(np.vdot(left, right)) ** 2, abs=1e-14
    )
    assert dimension * float(np.sum(first**2)) == pytest.approx(1.0, abs=1e-13)
    assert dimension * float(np.sum(second**2)) == pytest.approx(1.0, abs=1e-13)


def test_wigner_and_husimi_agree_on_where_a_coherent_state_sits() -> None:
    """The two phase-space pictures must peak on the same point.

    The Wigner lattice contains ``(0.25, 0.75)`` exactly; the Husimi midpoint
    grid has no cell there and puts its peak on the nearest one, 1/(2N) away.
    Requiring agreement within one cell is what makes this a check on the
    ``(q, p)`` ordering and on the block sum, which before it was applied gave
    four tied maxima at ``(0.25|0.75, 0.25|0.75)``.
    """
    dimension = 64
    state = coherent_state(dimension=dimension, position=0.25, momentum=0.75)

    wigner = wigner_distribution(state)
    husimi = husimi_distribution(state, grid_shape=(dimension, dimension))
    values = np.asarray(wigner.values)

    assert int(np.sum(values >= values.max() - 1e-15)) == 1
    peak = np.unravel_index(int(np.argmax(values)), values.shape)
    smooth = np.unravel_index(int(np.argmax(np.asarray(husimi.values))), (dimension, dimension))
    assert float(wigner.positions[peak[0]]) == pytest.approx(0.25, abs=1e-15)
    assert float(wigner.momenta[peak[1]]) == pytest.approx(0.75, abs=1e-15)
    assert abs(float(wigner.positions[peak[0]]) - float(husimi.positions[smooth[0]])) <= 1.0 / 64
    assert abs(float(wigner.momenta[peak[1]]) - float(husimi.momenta[smooth[1]])) <= 1.0 / 64


@pytest.mark.parametrize(("dimension", "trough"), [(16, -3.9822e-2), (32, -2.5063e-2)])
def test_wigner_goes_negative_for_a_cat_state_where_husimi_cannot(
    dimension: int, trough: float
) -> None:
    """The reason for having a Wigner function at all.

    A superposition of two coherent states shows interference fringes with
    ``W < 0``; the Husimi density of the very same state is positive everywhere,
    because it is a Gaussian smoothing. The measured trough is two thirds of the
    peak height, so this is a physical feature and not rounding.
    """
    left = coherent_state(dimension=dimension, position=0.25, momentum=0.5)
    right = coherent_state(dimension=dimension, position=0.75, momentum=0.5)
    cat = normalize_state(left + right)

    result = wigner_distribution(cat)
    values = np.asarray(result.values)
    smooth = np.asarray(husimi_distribution(cat, grid_shape=(dimension, dimension)).values)

    assert float(values.min()) == pytest.approx(trough, rel=1e-3)
    assert float(values.min()) < -0.5 * float(values.max())
    assert float(smooth.min()) > 0.0

    # A single coherent state is far less negative, and a basis state is exactly
    # non-negative, so ``negative_weight`` really does track interference.
    single = wigner_distribution(left)
    basis = wigner_distribution(basis_state(dimension=dimension, index=3))
    assert float(basis.negative_weight) == 0.0
    assert float(np.asarray(basis.values).min()) >= 0.0
    # Measured 0.555 against 0.192 at N = 16 and 0.607 against 0.133 at N = 32.
    assert float(result.negative_weight) > 2.5 * float(single.negative_weight)


def test_wigner_supports_batches_chunking_and_the_result_contract() -> None:
    dimension = 8
    states = np.stack(
        (
            coherent_state(dimension=dimension, position=0.25, momentum=0.75),
            basis_state(dimension=dimension, index=4),
        )
    )

    result = wigner_distribution(states, chunk_size=3)
    reference = wigner_distribution(states)

    assert result.values.shape == (2, dimension, dimension)
    assert result.negative_weight.shape == (2,)
    assert result.grid_points.shape == (dimension, dimension, 2)
    # ``chunk_size`` is a memory knob only: the numbers must not move at all.
    np.testing.assert_array_equal(result.values, reference.values)
    np.testing.assert_allclose(result.total, np.ones(2), atol=1e-14)
    assert not result.values.flags.writeable
    assert not result.negative_weight.flags.writeable
    np.testing.assert_array_equal(
        result.values[1], np.asarray(wigner_distribution(states[1]).values)
    )
    payload = result.array_payload()
    described = result.metadata_payload()["arrays"]
    assert isinstance(described, dict)
    assert set(payload) == set(described)
    assert result.metadata_payload()["result_type"] == "wigner"
    # ``tests/test_results.py`` asserts this over a hand-built list of every
    # container; assert it here too so the new type is covered either way.
    assert isinstance(result, SerializableResult)


@pytest.mark.parametrize(
    "roundtrip", [lambda item: pickle.loads(pickle.dumps(item)), copy.deepcopy]
)
def test_wigner_result_survives_pickle_and_deepcopy(roundtrip: object) -> None:
    original = wigner_distribution(coherent_state(dimension=6, position=0.3, momentum=0.7))
    assert callable(roundtrip)
    restored = roundtrip(original)

    assert isinstance(restored, WignerResult)
    assert restored == original
    assert restored != wigner_distribution(coherent_state(dimension=6, position=0.4, momentum=0.7))
    assert restored != object()
    for array in (
        restored.positions,
        restored.momenta,
        restored.values,
        restored.negative_weight,
    ):
        assert not array.flags.writeable


def test_wigner_rejects_an_odd_dimension_instead_of_guessing() -> None:
    """An odd ``N`` has no consistent 2 x 2 block pairing, so it is refused."""
    with pytest.raises(ValidationError, match="requires an even dimension"):
        wigner_distribution(basis_state(dimension=7, index=0))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: wigner_distribution(basis_state(dimension=4, index=0), chunk_size=0),
        lambda: wigner_distribution(
            basis_state(dimension=4, index=0),
            boundary_phases=(0.0, 0.0),  # type: ignore[arg-type]
        ),
        lambda: wigner_distribution([1.0, 1.0]),
    ],
)
def test_wigner_parameter_validation(operation: object) -> None:
    with pytest.raises(ValidationError):
        operation()  # type: ignore[operator]


def test_wigner_result_rejects_inconsistent_arrays() -> None:
    axis = np.array([0.0, 0.5])
    with pytest.raises(ValidationError, match="trailing grid shape"):
        WignerResult(axis, axis, np.zeros((3, 3)), np.zeros(()))
    with pytest.raises(ValidationError, match="only finite values"):
        WignerResult(axis, axis, np.full((2, 2), np.nan), np.zeros(()))
    with pytest.raises(ValidationError, match="metadata must be"):
        WignerResult(axis, axis, np.zeros((2, 2)), np.zeros(()), metadata=object())  # type: ignore[arg-type]
    # Unlike a Husimi density, negative values are the point and are accepted.
    signed = WignerResult(axis, axis, np.array([[0.75, -0.25], [0.25, 0.25]]), np.zeros(()))
    assert float(signed.total) == pytest.approx(1.0, abs=1e-15)


def test_husimi_coherent_grid_cache_does_not_change_a_single_bit() -> None:
    """The grid cache is a pure memoization, so numbers must be bit-identical.

    ``_CACHE_BUDGET_BYTES = 0`` disables it, which is exactly the pre-cache code
    path, and the two are compared with ``array_equal`` rather than a tolerance.
    That is the invariant the cache has to satisfy and it is asserted exactly.

    ``chunk_size`` is a different matter. It must reuse the same cache *entry*,
    which is asserted on the cache itself, but it does **not** leave the numbers
    bit-identical: the block width is the shape of the array handed to the matrix
    product, so changing it changes BLAS's blocking and therefore the order of a
    floating-point reduction. This test used to assert bit-equality across chunk
    sizes and passed on the development machine while failing on the CI Linux and
    macOS runners at a relative 1e-15, which is what a different BLAS kernel
    looks like. ``numerical-standards.md`` says bitwise identity is not promised
    across BLAS implementations or reduction orders; the comparison across chunk
    sizes is therefore made at rounding level.
    """
    state = coherent_state(dimension=32, position=0.3, momentum=0.7)
    phases = BoundaryPhases(0.25, 0.13)
    budget = phase_space._CACHE_BUDGET_BYTES
    try:
        phase_space._clear_coherent_cache()
        phase_space._CACHE_BUDGET_BYTES = 0
        uncached = husimi_distribution(
            state, grid_shape=(24, 20), boundary_phases=phases, chunk_size=37
        )
        assert not phase_space._coherent_cache
        phase_space._CACHE_BUDGET_BYTES = budget
        cached = husimi_distribution(
            state, grid_shape=(24, 20), boundary_phases=phases, chunk_size=37
        )
        reused = husimi_distribution(
            state, grid_shape=(24, 20), boundary_phases=phases, chunk_size=256
        )

        # The cache itself: bit-identical, no tolerance.
        np.testing.assert_array_equal(uncached.values, cached.values)
        # Across chunk sizes: rounding only, and bounded against the *peak* rather
        # than per element. A reduction-order change perturbs each cell by about
        # eps times the largest term in its own sum, so the far tail of the
        # distribution -- cells at 1e-10 while the peak is 30 -- moves by a
        # relative 1e-11 while moving by an absolute 7e-18. A per-element `rtol`
        # therefore measures the wrong thing and rejected a correct result on the
        # CI macOS runner. Measured worst absolute difference there 6.6e-18, which
        # is 2.2e-19 of the peak, and exactly zero on this machine.
        scale = float(np.max(uncached.values))
        drift = float(np.max(np.abs(np.asarray(uncached.values) - np.asarray(reused.values))))
        assert drift <= 1e-13 * scale
        assert abs(float(uncached.raw_integral) - float(reused.raw_integral)) <= 1e-13 * scale
        # What `chunk_size` really must not do is add a cache entry.
        assert len(phase_space._coherent_cache) == 1
        entry = next(iter(phase_space._coherent_cache.values()))
        # A writable cache entry would let one caller corrupt the next one's grid.
        assert not entry.flags.writeable
        assert entry.shape == (24 * 20, 32)

        # Grids of equal size but different shape are different grids.
        husimi_distribution(state, grid_shape=(8, 4), boundary_phases=phases)
        husimi_distribution(state, grid_shape=(4, 8), boundary_phases=phases)
        assert len(phase_space._coherent_cache) == 3
        # Every model parameter that changes the grid must change the key.
        husimi_distribution(state, grid_shape=(24, 20), boundary_phases=phases, images=2)
        husimi_distribution(state, grid_shape=(24, 20))
        assert len(phase_space._coherent_cache) == phase_space._CACHE_MAX_ENTRIES
    finally:
        phase_space._CACHE_BUDGET_BYTES = budget
        phase_space._clear_coherent_cache()


def test_husimi_cache_stays_inside_its_memory_budget() -> None:
    """An oversized grid is recomputed rather than retained."""
    state = coherent_state(dimension=32, position=0.3, momentum=0.7)
    budget = phase_space._CACHE_BUDGET_BYTES
    try:
        phase_space._clear_coherent_cache()
        phase_space._CACHE_BUDGET_BYTES = 32 * 32 * 16  # room for 32 centers
        big = husimi_distribution(state, grid_shape=(16, 16))
        assert not phase_space._coherent_cache

        small = husimi_distribution(state, grid_shape=(4, 8))
        assert len(phase_space._coherent_cache) == 1
        assert sum(e.nbytes for e in phase_space._coherent_cache.values()) <= (
            phase_space._CACHE_BUDGET_BYTES
        )
        np.testing.assert_allclose(big.integral, 1.0, atol=1e-14)
        np.testing.assert_allclose(small.integral, 1.0, atol=1e-14)
    finally:
        phase_space._CACHE_BUDGET_BYTES = budget
        phase_space._clear_coherent_cache()


def test_husimi_accepts_a_stack_of_states_in_one_call() -> None:
    """The documented batching route has to keep working.

    Batching is the only way to amortize the coherent-state grid when it is too
    large to cache: measured 0.55 s for one state and 0.57 s for twenty at
    ``N = 1024`` on a (64, 64) grid.
    """
    dimension = 32
    rng = np.random.default_rng(3)
    states = normalize_state(
        rng.standard_normal((20, dimension)) + 1j * rng.standard_normal((20, dimension))
    )

    batched = husimi_distribution(states, grid_shape=(12, 10))

    assert batched.values.shape == (20, 12, 10)
    assert batched.raw_integral.shape == (20,)
    for index in (0, 7, 19):
        single = husimi_distribution(states[index], grid_shape=(12, 10))
        # Not bit-identical: a (20, N) matrix product takes a different BLAS
        # path than a (1, N) one. Measured worst case 2.7e-15.
        np.testing.assert_allclose(batched.values[index], single.values, atol=1e-13)


def test_husimi_metadata_payload_describes_the_convergence_history() -> None:
    """Both halves of the persistence split must list the same arrays.

    ``array_payload`` adds ``convergence_history`` whenever the metadata carries
    one, so a descriptor that omits it makes the storage layer fail with a
    ``KeyError`` from a missing digest rather than a ``ValidationError``. No
    library path attaches a history to these containers today, which is exactly
    why the asymmetry survived.
    """
    metadata = ExperimentMetadata(
        convergence=ConvergenceInfo(
            converged=True,
            iterations=3,
            residual=0.0,
            tolerance=1e-12,
            history=np.array([2.0, 1.0]),
            reason="converged",
        )
    )
    axis = np.array([0.25, 0.75])
    containers = (
        HusimiResult(axis, axis, np.zeros((2, 2)), np.zeros(()), metadata=metadata),
        WignerResult(axis, axis, np.zeros((2, 2)), np.zeros(()), metadata=metadata),
    )

    for container in containers:
        described = container.metadata_payload()["arrays"]
        assert isinstance(described, dict)
        assert "convergence_history" in described
        assert set(container.array_payload()) == set(described)
        assert described["convergence_history"] == {"shape": [2], "dtype": "float64"}

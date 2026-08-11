from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from scipy.special import jv  # type: ignore[import-untyped]

from chaos_numerics.core import LinearOperatorLike, QuantumMap, ValidationError
from chaos_numerics.core.types import BoolArray, FloatArray
from chaos_numerics.quantum import (
    BoundaryPhases,
    CylinderKickedRotor,
    DenseUnitary,
    KickedRotor,
    QuantumBakerMap,
    QuantumCatMap,
    basis_state,
    desymmetrize,
    eigenstates,
    evolve,
    inverse_participation_ratio,
    normalize_state,
    unitarity_defect,
)
from chaos_numerics.spectral import (
    adjacent_gap_ratios,
    mean_gap_ratio_reference,
    prepare_eigenphases,
)


@pytest.mark.parametrize("dimension", [7, 8, 15, 16])
@pytest.mark.parametrize(
    "phases",
    [BoundaryPhases(), BoundaryPhases(0.25, 0.375)],
)
def test_dense_and_fft_actions_match_for_odd_even_dimensions(
    dimension: int, phases: BoundaryPhases
) -> None:
    model = KickedRotor(dimension, 7.25, boundary_phases=phases)
    rng = np.random.default_rng(1000 + dimension)
    states = normalize_state(
        rng.standard_normal((10, dimension)) + 1j * rng.standard_normal((10, dimension))
    )

    # Measured worst case over this grid is 2.5e-15; 1e-14 keeps a safety
    # factor of four without letting a real regression through.
    np.testing.assert_allclose(
        model.apply_fft(states),
        model.apply_dense(states),
        rtol=1e-14,
        atol=1e-14,
    )
    assert isinstance(model, QuantumMap)
    assert model.parameters["fft_normalization"] == "ortho"


def test_dense_reference_is_unitary_and_linear_operator_has_correct_adjoint() -> None:
    model = KickedRotor(12, -2.5, boundary_phases=BoundaryPhases(0.1, 0.3))
    state = normalize_state(np.arange(12) + 1j * np.arange(12)[::-1])
    linear = model.as_linear_operator()

    assert unitarity_defect(model) <= 1e-12
    np.testing.assert_allclose(linear.matvec(state), model.apply_fft(state), atol=1e-14)
    np.testing.assert_allclose(
        linear.rmatvec(linear.matvec(state)),
        state,
        rtol=1e-14,
        atol=1e-14,
    )


def test_fft_evolution_preserves_norm_without_dense_materialization() -> None:
    model = KickedRotor(257, 8.0, boundary_phases=0.125)
    initial = basis_state(dimension=model.dimension, index=17)

    # ``norm_tolerance`` is passed explicitly and well below the asserted bound:
    # with the default 5e-11 the assertion would be a tautology, because evolve
    # already raises NumericalError once drift exceeds its own tolerance.
    run = evolve(
        model,
        initial,
        steps=1000,
        method="fft",
        return_history=True,
        norm_tolerance=1e-12,
    )

    history = run.history
    assert history is not None
    assert history.shape == (1001, model.dimension)
    drift = np.abs(np.linalg.norm(history, axis=-1) - 1.0)
    # Measured drift after 1000 steps is 1.8e-13, growing linearly as ~1.8e-16*T.
    assert float(np.max(drift)) <= 5e-13
    np.testing.assert_allclose(model.apply(initial), model.apply_fft(initial), atol=1e-14)


def test_dense_and_fft_evolve_use_the_common_api() -> None:
    model = KickedRotor(16, 4.0, boundary_phases=0.2)
    state = normalize_state(np.linspace(1.0, 2.0, model.dimension) * (1.0 + 0.5j))

    dense = evolve(model, state, steps=5, method="dense", dense_limit=16)
    fft = evolve(model, state, steps=5, method="fft")

    np.testing.assert_allclose(fft.final_state, dense.final_state, rtol=5e-12, atol=5e-13)


def _circular_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the |shortest arc| between two phase arrays.

    Eigenphases are reduced into the half-open interval ``[-pi, pi)``, so an
    eigenvalue a rounding error away from ``-1`` may land on either end. A plain
    difference would then report ``2*pi`` instead of ``0``.
    """
    difference = np.abs(np.mod(left - right + np.pi, 2.0 * np.pi) - np.pi)
    return np.asarray(difference, dtype=np.float64)


@pytest.mark.parametrize("dimension", [8, 9, 16])
def test_free_rotor_eigenphases_match_the_closed_form(dimension: int) -> None:
    """Pin the sign and scale of the free propagator against exact theory.

    At ``K = 0`` the Floquet operator is diagonal in momentum with eigenvalues
    ``exp(-i hbar n^2 / 2)``, ``hbar = 2 pi / N``. Flipping the sign of that
    exponent, or rescaling it, is not a gauge choice: it moves the eigenphases
    by up to 0.48 rad while leaving unitarity, norm conservation, and
    dense/FFT agreement untouched, so those properties cannot detect it.
    """
    model = KickedRotor(dimension, 0.0)
    mode_numbers = np.fft.fftfreq(dimension) * dimension
    expected = np.angle(np.exp(-0.5j * (2.0 * np.pi / dimension) * mode_numbers**2))

    result = eigenstates(model)

    distance = _circular_distance(np.sort(expected), np.sort(result.eigenphases))
    assert float(np.max(distance)) <= 1e-14


def test_one_step_matches_an_independently_written_split_operator() -> None:
    """Pin the kick to ``cos`` and the free phase to ``exp(-i hbar n^2 / 2)``.

    The reference below is written from the definition rather than reused from
    the implementation, so replacing ``cos`` by ``sin`` in the kick, or
    flipping either exponent sign, fails here.
    """
    dimension = 12
    strength = 3.7
    model = KickedRotor(dimension, strength)
    hbar = 2.0 * np.pi / dimension

    positions = np.arange(dimension) / dimension
    kick = np.exp(-1j * strength * np.cos(2.0 * np.pi * positions) / hbar)
    mode_numbers = np.fft.fftfreq(dimension) * dimension
    free = np.exp(-0.5j * hbar * mode_numbers**2)
    fourier = np.exp(
        -2j * np.pi * np.outer(np.arange(dimension), np.arange(dimension)) / dimension
    ) / np.sqrt(dimension)
    expected = fourier.conj().T @ (free[:, None] * fourier) @ np.diag(kick)

    rng = np.random.default_rng(7)
    state = normalize_state(rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension))

    np.testing.assert_allclose(model.to_dense(), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(model.apply(state), expected @ state, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("dimension", [7, 8, 16])
@pytest.mark.parametrize(
    "phases",
    [
        BoundaryPhases(0.0, 0.0),
        BoundaryPhases(0.0, 0.5),
        BoundaryPhases(0.5, 0.0),
        BoundaryPhases(0.5, 0.5),
    ],
)
def test_parity_symmetry_is_published_exactly_for_self_reflecting_twists(
    dimension: int, phases: BoundaryPhases
) -> None:
    """All four twists with ``2 alpha`` and ``2 beta`` integer carry a reflection.

    ``(1/2, 0)`` and ``(0, 1/2)`` used to publish nothing, which left anyone who
    picked one of those twists comparing a two-sector superposition against RMT.
    The reflection index map is fixed by ``alpha`` alone; ``beta`` only decides
    the sign picked up where the reflection wraps past the boundary, which is why
    ``(0, 1/2)`` needs a sign-dressed operator rather than a bare permutation.
    """
    model = KickedRotor(dimension, 7.25, boundary_phases=phases)
    operators = model.symmetry_operators

    assert set(operators) == {"parity"}
    parity = operators["parity"]
    indices = np.arange(dimension)
    shift = int(2.0 * phases.position)
    targets = np.mod(-indices - shift, dimension)
    winding = (targets + indices + shift) // dimension
    signs = np.where(int(2.0 * phases.momentum) * winding % 2 == 1, -1.0, 1.0)
    expected = np.zeros((dimension, dimension), dtype=np.complex128)
    expected[targets, indices] = signs * signs[0]
    np.testing.assert_array_equal(parity, expected)
    # Global sign is normalized away, so the operator stays a plain permutation
    # except at ``(0, 1/2)``, where ``j = 0`` is the only positive entry.
    negatives = int(np.count_nonzero(parity.real < 0.0))
    assert negatives == (dimension - 1 if phases == BoundaryPhases(0.0, 0.5) else 0)

    dense = model.to_dense()
    defect = float(np.linalg.norm(dense @ parity - parity @ dense) / np.sqrt(dimension))
    assert defect <= 1e-12
    identity = np.eye(dimension, dtype=np.complex128)
    np.testing.assert_array_equal(parity @ parity, identity)
    np.testing.assert_array_equal(parity, parity.conj().T)
    reported = eigenstates(model).metadata.parameters["symmetry_defects"]
    assert isinstance(reported, Mapping)
    assert dict(reported) == {"parity": pytest.approx(defect, abs=1e-15)}


def test_antiperiodic_momentum_twist_needs_a_sign_dressed_reflection() -> None:
    """At ``(0, 1/2)`` the bare permutation ``j -> -j`` does not commute.

    This is the measurement that a permutation-only search misses. Measured at
    ``N = 128``, ``K = 10``: the published sign-dressed operator has commutator
    defect ``6.75e-14`` while the bare permutation sits at ``0.2491`` -- twelve
    orders of magnitude above any rounding scale, but far enough below the
    ``1.579`` of a reflection applied at the wrong twist to look like a bug rather
    than a convention error.
    """
    dimension = 128
    model = KickedRotor(dimension, 10.0, boundary_phases=BoundaryPhases(0.0, 0.5))
    dense = model.to_dense()
    published = model.symmetry_operators["parity"]
    indices = np.arange(dimension)
    bare = np.zeros((dimension, dimension), dtype=np.complex128)
    bare[np.mod(-indices, dimension), indices] = 1.0

    def defect(symmetry: np.ndarray) -> float:
        return float(np.linalg.norm(dense @ symmetry - symmetry @ dense) / np.sqrt(dimension))

    assert defect(published) <= 1e-12
    assert 0.2 <= defect(bare) <= 0.3
    # The two differ only in the sign of the wrapped entries.
    np.testing.assert_array_equal(np.abs(published), bare)


@pytest.mark.parametrize(
    "phases",
    [BoundaryPhases(0.25, 0.0), BoundaryPhases(0.25, 0.13), BoundaryPhases(0.0, 0.25)],
)
def test_generic_bloch_phases_publish_no_parity_symmetry(phases: BoundaryPhases) -> None:
    """A generic twist genuinely breaks parity, so nothing is published.

    "Genuinely" is checked by brute force over every reflection ``j -> c - j``
    and every diagonal sign dressing of it, which together exhaust the operators
    that could reflect ``q -> -q`` on this grid. Measured smallest defect over
    that whole family is at least ``0.6`` for each twist here, against ``5e-14``
    for the four twists that do carry a reflection.
    """
    dimension = 16
    model = KickedRotor(dimension, 7.25, boundary_phases=phases)

    assert model.symmetry_operators == {}
    dense = model.to_dense()
    indices = np.arange(dimension)
    worst = np.inf
    for centre in range(dimension):
        targets = np.mod(centre - indices, dimension)
        for signs in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)):
            candidate = np.zeros((dimension, dimension), dtype=np.complex128)
            wrapped = (centre - indices) < 0
            candidate[targets, indices] = np.where(wrapped, signs[1], signs[0])
            worst = min(
                worst,
                float(np.linalg.norm(dense @ candidate - candidate @ dense) / np.sqrt(dimension)),
            )
    assert worst > 0.5


def test_dense_reference_is_memoized_and_read_only() -> None:
    model = KickedRotor(32, 5.0)

    first = model.to_dense()
    second = model.to_dense()

    assert first is not second
    assert first.flags.writeable
    np.testing.assert_array_equal(first, second)
    # apply_dense reuses the same cached matrix instead of reassembling it.
    cached = model._dense_matrix()
    assert model._dense_matrix() is cached
    assert not cached.flags.writeable
    state = basis_state(dimension=32, index=3)
    np.testing.assert_array_equal(model.apply_dense(state), state @ cached.T)


def test_kicked_rotor_scalar_boundary_phases_is_common_twist() -> None:
    model = KickedRotor(8, 1.0, boundary_phases=1.25)

    assert model.boundary_phases == BoundaryPhases(position=0.25, momentum=0.25)
    assert model.effective_hbar == pytest.approx(2.0 * np.pi / 8)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: KickedRotor(8, 1.0, boundary_phase=0.25),  # type: ignore[call-arg]
        lambda: QuantumCatMap(4, boundary_phase=0.0),  # type: ignore[call-arg]
        lambda: QuantumBakerMap(4, boundary_phase=0.5),  # type: ignore[call-arg]
    ],
)
def test_singular_boundary_phase_keyword_is_gone(constructor: object) -> None:
    """The attribute and the keyword now share one spelling, ``boundary_phases``.

    This is a pre-v0.1 breaking rename with deliberately no compatibility shim
    and no ``**kwargs`` interception. The interpreter's own message is already
    the message a shim would print -- on CPython 3.13+ it reads
    ``got an unexpected keyword argument 'boundary_phase'. Did you mean
    'boundary_phases'?`` -- and accepting ``**kwargs`` purely to reword it would
    make every other misspelled keyword silently legal to static checkers.
    """
    with pytest.raises(TypeError, match="boundary_phase"):
        constructor()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0, 1.0, 0.0), "dimension must be positive"),
        ((4.0, 1.0, 0.0), "positive integer"),
        ((4, float("nan"), 0.0), "kick_strength must be a finite real"),
        ((4, 1.0, float("inf")), "position phase must be a finite real"),
        ((4, 1.0, True), "position phase must be a finite real"),
    ],
)
def test_kicked_rotor_validates_quantization_before_dense_allocation(
    args: tuple[object, object, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        KickedRotor(*args)  # type: ignore[arg-type]


def test_fft_method_requires_an_fft_capable_model() -> None:
    with pytest.raises(ValidationError, match="does not provide apply_fft"):
        evolve(
            DenseUnitary(np.eye(2, dtype=np.complex128)),
            basis_state(dimension=2, index=0),
            steps=1,
            method="fft",
        )


def test_cylinder_rotor_satisfies_the_protocols_and_is_unitary() -> None:
    model = CylinderKickedRotor(64, 3.0, effective_hbar=0.7)
    # Two handles on the same operator: one narrowed to the protocol the library
    # promises, one left as the SciPy object whose extra methods are also wired.
    protocol_view = model.as_linear_operator()
    linear = model.as_linear_operator()
    rng = np.random.default_rng(11)
    states = normalize_state(
        rng.standard_normal((5, 64)) + 1j * rng.standard_normal((5, 64)),
    )

    assert isinstance(model, QuantumMap)
    assert isinstance(protocol_view, LinearOperatorLike)
    assert model.parameters == {
        "dimension": 64,
        "kick_strength": 3.0,
        "effective_hbar": 0.7,
        "basis": "momentum",
        "geometry": "cylinder",
        "floquet_order": "free_after_kick",
        "fft_normalization": "ortho",
    }

    evolved = model.apply(states)
    # Measured worst norm drift over this batch is 2.2e-16.
    assert float(np.max(np.abs(np.linalg.norm(evolved, axis=-1) - 1.0))) <= 1e-14
    # Measured dense defect is 3.6e-16.
    assert unitarity_defect(model, dense_limit=64) <= 1e-14
    # Measured worst adjoint round-trip error is 1.5e-16.
    np.testing.assert_allclose(model.apply_adjoint(evolved), states, rtol=1e-14, atol=1e-14)

    np.testing.assert_allclose(linear.matvec(states[0]), model.apply(states[0]), atol=1e-15)
    # matmat routes through the same FFT as apply, but transposing the batch
    # changes the memory layout, and FFT results are not bit-identical across
    # layouts on every BLAS/NumPy build. Compare at rounding level, not exactly.
    np.testing.assert_allclose(linear.matmat(states.T), evolved.T, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        linear.rmatvec(states[0]), model.apply_adjoint(states[0]), rtol=0.0, atol=1e-15
    )


def test_cylinder_momentum_numbers_are_the_symmetric_read_only_branch() -> None:
    model = CylinderKickedRotor(8, 1.0, effective_hbar=2.0)

    np.testing.assert_array_equal(
        model.momentum_numbers,
        np.asarray([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0]),
    )
    assert not model.momentum_numbers.flags.writeable


@pytest.mark.parametrize("dimension", [16, 17, 32])
def test_cylinder_free_rotor_eigenphases_match_the_closed_form(dimension: int) -> None:
    """At ``K = 0`` the cylinder rotor is diagonal in its own momentum basis.

    This pins the sign and the scale of ``exp(-i hbar n^2 / 2)`` against exact
    theory with ``hbar`` decoupled from ``dimension``, so a stray ``2 pi / N``
    creeping back into the free phase fails here.
    """
    hbar = 0.55
    model = CylinderKickedRotor(dimension, 0.0, effective_hbar=hbar)
    expected = np.angle(np.exp(-0.5j * hbar * model.momentum_numbers**2))

    result = eigenstates(model, dense_limit=dimension)

    distance = _circular_distance(np.sort(expected), np.sort(result.eigenphases))
    # Measured worst circular distance over this grid is 1.3e-15.
    assert float(np.max(distance)) <= 1e-14


def test_cylinder_rotor_reproduces_dynamical_localization() -> None:
    r"""The reason this class exists: ``<n^2>`` leaves ``D t`` and saturates.

    Two estimates of the *distribution* decay length must agree with each other.
    For an exponential profile ``exp(-|n| / ell_dist)`` the saturated moment obeys
    ``<n^2>_sat = 2 ell_dist^2``, so ``sqrt(<n^2>_sat / 2)`` and a log-linear fit
    of the profile measure the same ``ell_dist`` by different routes. Neither of
    them measures ``D / 2``, which is the *eigenfunction* localization length;
    see ``test_cylinder_rotor_separates_its_two_localization_lengths``.

    Measured at ``N = 2048``, ``K = 10``, ``hbar = 1`` after 400 kicks:
    ``<n^2>`` equals ``D t`` exactly at ``t = 1``, is already half of it by
    ``t = 5``, and reaches only 4% of it at ``t = 400``. The fitted ``ell_dist``
    is 20.6 and ``sqrt(<n^2>_sat / 2)`` is 20.2. The 10% tolerance below is the
    band sensitivity of the log-linear fit, not statistical noise: the run is
    deterministic.
    """
    dimension, kick, hbar, steps = 2048, 10.0, 1.0, 400
    # In the <(Delta n)^2> = D t convention the quasilinear rate is K^2 / 2 for
    # hbar = 1, which is the exact one-kick moment of a momentum eigenstate.
    diffusion = kick**2 / (2.0 * hbar**2)
    model = CylinderKickedRotor(dimension, kick, effective_hbar=hbar)
    run = evolve(
        model,
        basis_state(dimension=dimension, index=0),
        steps=steps,
        return_history=True,
    )
    assert run.history is not None
    probability = np.abs(run.history) ** 2
    modes = model.momentum_numbers
    second_moment = probability @ modes**2

    # The finite lattice must not be contaminating the result: measured peak
    # probability beyond |n| > 900 over the whole run is 8.5e-28.
    assert float(np.max(np.sum(probability[:, np.abs(modes) > 900], axis=-1))) <= 1e-20

    # One kick follows classical diffusion exactly, then the quantum curve
    # departs; by the end it is two orders of magnitude below the classical line.
    assert float(second_moment[1]) == pytest.approx(diffusion, rel=1e-12)
    assert float(second_moment[5]) <= 0.6 * diffusion * 5
    assert float(second_moment[steps]) <= 0.1 * diffusion * steps
    # Saturation, not slow growth: the second half gains almost nothing.
    assert float(second_moment[steps]) <= 1.5 * float(second_moment[steps // 2])

    profile = probability[-1]
    order = np.argsort(modes)
    axis, tail = modes[order], profile[order]
    band = (np.abs(axis) > 5.0) & (np.abs(axis) < 150.0) & (tail > 0.0)
    slope = np.polyfit(np.abs(axis[band]), np.log(tail[band]), 1)[0]
    ell_fit = -1.0 / float(slope)
    ell_moment = float(np.sqrt(second_moment[-1] / 2.0))

    assert ell_fit == pytest.approx(ell_moment, rel=0.10)
    # Both routes measure the distribution length, which is around D rather than
    # D / 2; the band below is wide because K = 10 sits near an accelerator
    # resonance where the Bessel correction to D is large.
    assert 0.5 <= ell_fit / (diffusion / 2.0) <= 2.0
    # Localized means ell is far below the basis size, which is exactly what the
    # torus KickedRotor cannot arrange.
    assert ell_fit < dimension / 20.0


def _self_consistent_decay_length(axis: FloatArray, log_profile: FloatArray, guess: float) -> float:
    """Return the log-linear decay length fitted on the band ``ell < |n| < 6 ell``.

    Tying the fit band to the answer removes the free parameter that makes a
    hand-picked band look like a measurement. The iteration is damped, so it
    converges from either side. ``axis`` is an integer lattice, so the band is a
    step function of ``ell`` and there is no exact fixed point; convergence is
    declared once the band membership repeats, at which point the fit is
    bit-for-bit reproducible.
    """
    ell = guess
    previous: BoolArray | None = None
    for _ in range(50):
        band = (np.abs(axis) > ell) & (np.abs(axis) < 6.0 * ell)
        slope = np.polyfit(np.abs(axis[band]), log_profile[band], 1)[0]
        updated = -1.0 / float(slope)
        if previous is not None and np.array_equal(band, previous):
            return updated
        previous = band
        ell = 0.5 * (ell + updated)
    raise AssertionError("decay-length fit band did not settle")


def test_cylinder_rotor_separates_its_two_localization_lengths() -> None:
    r"""``ell ~ D / 2`` is the eigenfunction length, not the distribution length.

    The saturated distribution left behind by a wave packet is a superposition of
    eigenfunctions and decays about twice as slowly as one of them, so fitting the
    distribution and calling the answer ``D / 2`` overestimates ``D`` by about a
    factor of two. That trap is exactly what the class docstring used to set.

    Measured at ``K = 5``, ``hbar = 1`` with the Bessel-corrected quasilinear rate
    ``D = (K^2 / 2)(1 - 2 J_2(K)) = 11.336``, so ``D / 2 = 5.668``:

    * eigenfunction length ``ell_eig = 4.48``, which is ``0.79 (D / 2)``
    * distribution length ``ell_dist = 8.59``, which is ``1.52 (D / 2) = 0.76 D``
    * ``ell_dist / ell_eig = 1.92``
    * ``<n^2>_sat = 117.9``, which is ``0.92 D^2`` -- **not** ``D^2 / 2 = 64.3``

    Every number here is deterministic; the bands below are set by the fit-band
    and finite-lattice sensitivity, which is a few percent, not by sampling noise.
    """
    kick, hbar = 5.0, 1.0
    # Leading Bessel correction to the random-walk rate K^2 / 2. Without it the
    # rate is 12.5 instead of 11.34 and every ratio below shifts by 10%.
    diffusion = kick**2 / (2.0 * hbar**2) * (1.0 - 2.0 * float(jv(2, kick / hbar)))
    assert diffusion == pytest.approx(11.336, abs=0.001)

    dimension, steps = 1024, 1500
    model = CylinderKickedRotor(dimension, kick, effective_hbar=hbar)
    run = evolve(
        model,
        basis_state(dimension=dimension, index=0),
        steps=steps,
        return_history=True,
    )
    assert run.history is not None
    modes = model.momentum_numbers
    saturated = np.abs(run.history[steps // 2 :]) ** 2
    # Measured leaked probability beyond a quarter of the lattice is 2.9e-17.
    assert float(np.max(np.sum(saturated[:, np.abs(modes) > dimension / 4], axis=-1))) <= 1e-12
    profile = np.mean(saturated, axis=0)
    second_moment = float(np.mean(saturated @ modes**2))
    order = np.argsort(modes)
    ell_dist = _self_consistent_decay_length(
        modes[order], np.log(np.maximum(profile[order], 1e-300)), diffusion
    )

    eigen_dimension = 512
    eigen_model = CylinderKickedRotor(eigen_dimension, kick, effective_hbar=hbar)
    system = eigenstates(eigen_model, dense_limit=eigen_dimension)
    eigen_modes = eigen_model.momentum_numbers
    eigen_order = np.argsort(eigen_modes)
    columns = np.abs(system.eigenstates[eigen_order, :]) ** 2
    centre = eigen_dimension // 2
    # Geometric mean of the peak-centred profiles: the typical exponential decay.
    # An arithmetic mean is dominated by the widest state in the spectrum and does
    # not have a decay length at all.
    centred = np.stack(
        [
            np.roll(columns[:, index], centre - int(np.argmax(columns[:, index])))
            for index in range(eigen_dimension)
        ],
        axis=1,
    )
    typical = np.mean(np.log(np.maximum(centred, 1e-300)), axis=1)
    ell_eig = _self_consistent_decay_length(
        (np.arange(eigen_dimension) - centre).astype(np.float64), typical, diffusion / 2.0
    )

    assert ell_eig == pytest.approx(4.48, abs=0.15)
    assert ell_dist == pytest.approx(8.59, abs=0.3)
    assert second_moment == pytest.approx(117.9, rel=0.05)

    # The eigenfunctions are the length D / 2 refers to.
    assert 0.65 <= ell_eig / (diffusion / 2.0) <= 1.0
    # The distribution is about twice that, so about D.
    assert 1.7 <= ell_dist / ell_eig <= 2.2
    assert 0.65 <= ell_dist / diffusion <= 0.95
    # And the moment follows the distribution: ~D^2, decidedly not D^2 / 2.
    assert 0.8 <= second_moment / diffusion**2 <= 1.2
    assert second_moment / (diffusion**2 / 2.0) >= 1.6
    # Which is the same statement as sqrt(<n^2>_sat / 2) being far from D / 2.
    assert np.sqrt(second_moment / 2.0) / (diffusion / 2.0) == pytest.approx(1.355, abs=0.05)


def test_cylinder_rotor_accepts_the_fft_evolution_method() -> None:
    """``evolve(..., method="fft")`` is what the README teaches for long runs.

    This is an FFT implementation with no dense matrix at all, yet it used to be
    the one model that rejected ``method="fft"``, so anyone following the
    localization recipe hit ``does not provide apply_fft`` on their first call.
    ``apply_fft`` is an exact alias of ``apply`` here, and the assertions below
    pin that: same object contract, bit-for-bit identical output.
    """
    model = CylinderKickedRotor(64, 4.5, effective_hbar=0.8)
    state = basis_state(dimension=64, index=3)

    # Compared at rounding level rather than bit for bit. On the oldest supported
    # NumPy (1.26.4) a complex128 elementwise multiply is not reproducible call to
    # call: measured on the phase product inside `apply`, eight invocations with the
    # same input array spread over 2e-17, and two `evolve` runs with identical
    # arguments differ by 3e-16. NumPy 2.5 is stable here. The alias is exact by
    # construction -- `apply_fft` returns `self.apply(state)` -- so what this pins is
    # that `method="fft"` reaches the same operator, not that floating point repeats.
    np.testing.assert_allclose(model.apply_fft(state), model.apply(state), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        evolve(model, state, steps=12, method="fft").final_state,
        evolve(model, state, steps=12).final_state,
        rtol=0.0,
        atol=1e-14,
    )
    # The batched contract survives the alias too, again at rounding level.
    batch = np.stack([basis_state(dimension=64, index=index) for index in (0, 7)])
    np.testing.assert_allclose(model.apply_fft(batch), model.apply(batch), rtol=0.0, atol=1e-15)


def test_desymmetrizing_the_mixed_twists_recovers_coe_statistics() -> None:
    """The payoff of publishing parity at ``(1/2, 0)`` and ``(0, 1/2)``.

    Before the fix these two twists reported no symmetry, so a caller comparing
    them against RMT was handed the superposition of two parity blocks: mean
    adjacent gap ratio 0.42 against a COE reference of 0.5307, which reads as a
    result rather than as a mistake. Measured at ``N = 512``, ``K = 10``:

    ============= ======== ============= =============
    ``(a, b)``    raw      even sector   odd sector
    ============= ======== ============= =============
    ``(1/2, 0)``  0.4229   0.5409        0.5432
    ``(0, 1/2)``  0.4182   0.5092        0.5321
    ============= ======== ============= =============

    The standard error of a 256-level sector mean is about 0.016, so the raw
    value is 7 standard errors low and both sectors are within two of COE.
    """
    dimension = 512
    for phases in (BoundaryPhases(0.5, 0.0), BoundaryPhases(0.0, 0.5)):
        model = KickedRotor(dimension, 10.0, boundary_phases=phases)
        system = eigenstates(model, dense_limit=dimension)
        raw = adjacent_gap_ratios(prepare_eigenphases(system, symmetry_sector="mixed-parity"))
        assert float(np.mean(raw.values)) == pytest.approx(0.42, abs=0.03)

        parity = model.symmetry_operators["parity"]
        for sector in ("even", "odd"):
            block = desymmetrize(system, parity, sector=sector)
            assert block.count == dimension // 2
            ratios = adjacent_gap_ratios(prepare_eigenphases(block, symmetry_sector=sector))
            assert float(np.mean(ratios.values)) == pytest.approx(
                mean_gap_ratio_reference("coe"), abs=0.045
            )


def test_cylinder_rotor_evolves_a_batch_of_initial_momenta_independently() -> None:
    model = CylinderKickedRotor(256, 5.0, effective_hbar=1.0)
    indices = (0, 3, 40, 255)
    batch = np.stack([basis_state(dimension=256, index=index) for index in indices])

    batched = evolve(model, batch, steps=20)

    assert batched.final_state.shape == (len(indices), 256)
    assert batched.batch_shape == (len(indices),)
    for row, index in enumerate(indices):
        single = evolve(model, basis_state(dimension=256, index=index), steps=20)
        np.testing.assert_allclose(
            batched.final_state[row], single.final_state, rtol=1e-14, atol=1e-15
        )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0, 1.0, 1.0), "dimension must be positive"),
        ((4.0, 1.0, 1.0), "positive integer"),
        ((4, float("nan"), 1.0), "kick_strength must be a finite real"),
        ((4, 1.0, 0.0), "effective_hbar must be positive"),
        ((4, 1.0, -0.5), "effective_hbar must be positive"),
        ((4, 1.0, float("inf")), "effective_hbar must be a finite real"),
        ((4, 1.0, float("nan")), "effective_hbar must be a finite real"),
        ((4, 1.0, True), "effective_hbar must be a finite real"),
    ],
)
def test_cylinder_rotor_validates_its_quantization(
    args: tuple[object, object, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        CylinderKickedRotor(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize("dimension", [7, 8, 63, 64, 65, 128])
@pytest.mark.parametrize(
    "phases",
    [BoundaryPhases(), BoundaryPhases(0.25, 0.13), BoundaryPhases(0.5, 0.5)],
)
def test_basis_change_is_unitary_and_round_trips(dimension: int, phases: BoundaryPhases) -> None:
    """The published transform must preserve the norm and invert exactly.

    Measured worst case over this grid is 4.5e-16 for the norm and 2.5e-16 for
    the round trip, so 1e-14 keeps a factor of about 20 of headroom.
    """
    model = KickedRotor(dimension, 3.0, boundary_phases=phases)
    rng = np.random.default_rng(4000 + dimension)
    states = normalize_state(
        rng.standard_normal((5, dimension)) + 1j * rng.standard_normal((5, dimension))
    )

    momentum = model.to_momentum_basis(states)

    assert momentum.shape == states.shape
    assert momentum.dtype == np.dtype(np.complex128)
    np.testing.assert_allclose(np.linalg.norm(momentum, axis=-1), np.ones(5), atol=1e-14)
    np.testing.assert_allclose(model.to_position_basis(momentum), states, atol=1e-14)
    # A single state must go through the same code path as a batch.
    np.testing.assert_allclose(model.to_momentum_basis(states[0]), momentum[0], atol=1e-15)


@pytest.mark.parametrize(
    "phases",
    [BoundaryPhases(), BoundaryPhases(0.25, 0.13), BoundaryPhases(0.5, 0.0)],
)
def test_basis_change_matches_the_twisted_dft_built_from_to_dense(
    phases: BoundaryPhases,
) -> None:
    """Pin the transform against the same matrix the dense reference assembles.

    ``to_dense()`` is ``F* D_free F D_kick``, so ``F`` can be recovered from it
    without trusting the FFT path: at ``K = 0`` the kick is the identity and
    ``F* D_free F`` is diagonalized by exactly the matrix under test. The check
    here is the more direct one -- the explicit DFT entries of
    ``docs/design/kicked-rotor.md`` -- plus the ``K = 0`` consistency of the
    dense operator with the transform. Measured worst case 1.2e-14.
    """
    dimension = 24
    model = KickedRotor(dimension, 6.5, boundary_phases=phases)
    indices = np.arange(dimension, dtype=np.float64)
    fourier = np.exp(
        -2j * np.pi * np.outer(indices + phases.momentum, indices + phases.position) / dimension
    ) / np.sqrt(dimension)
    rng = np.random.default_rng(77)
    state = normalize_state(rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension))

    np.testing.assert_allclose(model.to_momentum_basis(state), fourier @ state, atol=1e-14)
    np.testing.assert_allclose(model.to_position_basis(state), fourier.conj().T @ state, atol=1e-14)

    # The free rotation is diagonal in this basis, which is what makes the
    # transform the right one for the model rather than merely unitary.
    free = KickedRotor(dimension, 0.0, boundary_phases=phases)
    np.testing.assert_allclose(
        free.to_momentum_basis(free.apply_dense(state)),
        np.exp(-0.5j * free.effective_hbar * free.momentum_numbers**2)
        * free.to_momentum_basis(state),
        atol=1e-14,
    )


def test_momentum_numbers_match_the_cylinder_convention_and_are_read_only() -> None:
    """Both rotors must label momentum the same way, up to the torus twist."""
    torus = KickedRotor(8, 1.0, boundary_phases=BoundaryPhases(0.25, 0.13))
    plain = KickedRotor(8, 1.0)
    cylinder = CylinderKickedRotor(8, 1.0)

    np.testing.assert_array_equal(plain.momentum_numbers, cylinder.momentum_numbers)
    np.testing.assert_allclose(torus.momentum_numbers, cylinder.momentum_numbers + 0.13, atol=1e-15)
    np.testing.assert_allclose(plain.momentum_numbers, [0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
    for numbers in (torus.momentum_numbers, cylinder.momentum_numbers):
        assert not numbers.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        torus.momentum_numbers[0] = 0.0


def test_bare_numpy_fft_is_wrong_by_a_measurable_amount_at_nonzero_twist() -> None:
    """Fix the size of the mistake the docstring warns about.

    ``alpha`` only rotates phases, so a momentum *histogram* cannot detect a
    missing position twist; ``beta`` moves probability between bins and a
    histogram can. Both measured at ``N = 64`` on one seeded random state.
    """
    dimension = 64
    rng = np.random.default_rng(9)
    state = normalize_state(rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension))
    bare = np.fft.fft(state, norm="ortho")

    # Zero twist: the two are the same transform, bit for bit. This is why the
    # mistake survives tests written at the default.
    np.testing.assert_array_equal(KickedRotor(dimension, 1.0).to_momentum_basis(state), bare)

    # Dropping ``norm="ortho"`` inflates every probability by exactly N.
    assert float(np.linalg.norm(np.fft.fft(state))) == pytest.approx(np.sqrt(dimension), rel=1e-13)

    position_only = KickedRotor(
        dimension, 1.0, boundary_phases=BoundaryPhases(0.25, 0.0)
    ).to_momentum_basis(state)
    amplitude_error = float(np.linalg.norm(position_only - bare))
    probability_error = float(np.max(np.abs(np.abs(position_only) ** 2 - np.abs(bare) ** 2)))
    # Measured 0.757 of a unit norm, against 4.2e-17 in the probabilities: the
    # amplitudes are three quarters wrong and the histogram cannot tell.
    assert amplitude_error == pytest.approx(0.757, abs=2e-3)
    assert probability_error < 1e-15

    for momentum_twist, expected in ((0.13, 0.1051), (0.5, 0.3759)):
        twisted = KickedRotor(
            dimension, 1.0, boundary_phases=BoundaryPhases(0.25, momentum_twist)
        ).to_momentum_basis(state)
        total_variation = 0.5 * float(np.sum(np.abs(np.abs(twisted) ** 2 - np.abs(bare) ** 2)))
        assert total_variation == pytest.approx(expected, abs=1e-3)


def test_momentum_basis_localization_differs_from_the_position_basis() -> None:
    """A weakly kicked rotor is momentum localized and position delocalized.

    At ``K = 0.2`` the Floquet eigenstates are near-momentum eigenstates, so the
    momentum-basis IPR is large while the position-basis IPR sits at the
    delocalized floor ``1/N``. Measured medians at ``N = 64``: position 0.0158
    against the ``1/N = 0.0156`` floor, momentum 0.274, a factor of 17. At
    ``K = 8`` both bases are chaotic and the two agree to about 10% (0.0329 and
    0.0295), which is what makes the small-``K`` split evidence about the
    transform rather than about the measure.
    """
    dimension = 64
    phases = BoundaryPhases(0.25, 0.13)

    weak = KickedRotor(dimension, 0.2, boundary_phases=phases)
    vectors = np.asarray(eigenstates(weak, dense_limit=dimension).eigenstates).T
    position = np.asarray(inverse_participation_ratio(vectors))
    momentum = np.asarray(inverse_participation_ratio(weak.to_momentum_basis(vectors)))

    assert float(np.median(position)) == pytest.approx(0.0158, abs=5e-4)
    assert float(np.median(momentum)) == pytest.approx(0.274, abs=5e-3)
    assert float(np.median(momentum)) > 10.0 * float(np.median(position))

    strong = KickedRotor(dimension, 8.0, boundary_phases=phases)
    chaotic = np.asarray(eigenstates(strong, dense_limit=dimension).eigenstates).T
    chaotic_position = float(np.median(np.asarray(inverse_participation_ratio(chaotic))))
    chaotic_momentum = float(
        np.median(np.asarray(inverse_participation_ratio(strong.to_momentum_basis(chaotic))))
    )
    assert chaotic_momentum == pytest.approx(chaotic_position, rel=0.2)


def test_cylinder_basis_change_is_the_mirror_image_of_the_torus_one() -> None:
    """The cylinder stores momentum, so the same method names swap roles."""
    dimension = 32
    model = CylinderKickedRotor(dimension, 4.0, effective_hbar=0.7)
    stored = basis_state(dimension=dimension, index=5)

    angles = model.to_position_basis(stored)

    # A momentum eigenstate is a plane wave of unit modulus in the angle basis.
    np.testing.assert_allclose(np.abs(angles), np.full(dimension, dimension**-0.5), atol=1e-15)
    np.testing.assert_allclose(model.to_momentum_basis(angles), stored, atol=1e-15)
    np.testing.assert_allclose(
        angles,
        np.fft.ifft(stored, norm="ortho"),
        atol=1e-15,
    )

    # The kick is diagonal in the angle basis for this class, the mirror of the
    # torus rotor where the free rotation is diagonal in the momentum basis.
    free_only = CylinderKickedRotor(dimension, 0.0, effective_hbar=0.7)
    batch = np.stack([stored, basis_state(dimension=dimension, index=17)])
    np.testing.assert_allclose(
        free_only.apply(batch),
        np.exp(-0.5j * 0.7 * free_only.momentum_numbers**2) * batch,
        atol=1e-15,
    )
    assert free_only.to_position_basis(batch).shape == batch.shape


@pytest.mark.parametrize(
    "operation",
    [
        lambda: KickedRotor(8, 1.0).to_momentum_basis(np.ones(7)),
        lambda: KickedRotor(8, 1.0).to_position_basis(np.ones(9)),
        lambda: CylinderKickedRotor(8, 1.0).to_position_basis(np.ones(4)),
        lambda: CylinderKickedRotor(8, 1.0).to_momentum_basis(np.ones(4)),
    ],
)
def test_basis_change_rejects_a_mismatched_dimension(operation: object) -> None:
    with pytest.raises(ValidationError, match="trailing dimension 8"):
        operation()  # type: ignore[operator]

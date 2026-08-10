"""Torus and cylinder kicked rotors with dense and FFT operator actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]

from chaos_numerics.core import ValidationError
from chaos_numerics.core.types import ArrayLike, ComplexArray, FloatArray
from chaos_numerics.quantum.states import BoundaryPhases, _reflection_operator, quantum_state


@dataclass(frozen=True, slots=True, eq=False, init=False)
class KickedRotor:
    """Finite-dimensional torus quantization of the standard kicked rotor.

    States use the position ordering ``q_j = (j + alpha) / N``. One Floquet
    step applies the kick and then the free rotation,
    ``U = F.H @ D_free @ F @ D_kick``, with ``hbar_eff = 2 pi / N``.
    ``F`` is the unitary twisted DFT determined by ``(alpha, beta)``.

    The kick amplitude the literature calls ``K`` is the ``kick_strength``
    argument; there is deliberately no ``K`` alias, so ``KickedRotor(64, K=5.0)``
    is a plain ``TypeError``.

    A scalar ``boundary_phases`` is shorthand for ``alpha = beta``. Pass a
    :class:`BoundaryPhases` value to control the two twists independently. The
    constructor argument and the stored attribute share the plural spelling, so
    a keyword taken from one is accepted by the other.

    ``apply_dense`` and ``to_dense`` share one memoized dense matrix, so the
    ``O(N**3)`` assembly happens at most once per instance and an ``O(N**2)``
    buffer is retained afterwards. ``apply``/``apply_fft`` never touch it.

    **States are position amplitudes**, and :meth:`to_momentum_basis` /
    :meth:`to_position_basis` are the twisted DFT that moves between the two
    bases. :class:`CylinderKickedRotor` stores *momentum* amplitudes instead, so
    the same two method names take and return the opposite bases there; see
    :meth:`to_momentum_basis` for the comparison table.

    Tying ``hbar_eff`` to ``N`` is what makes this a finite unitary with a full
    eigenphase spectrum, and it is also why dynamical localization cannot appear
    here: the localization length always exceeds the basis size. Use
    :class:`CylinderKickedRotor` for that regime.
    """

    dimension: int
    kick_strength: float
    boundary_phases: BoundaryPhases
    effective_hbar: float
    _momentum_numbers: FloatArray = field(repr=False)
    _kick_phase: ComplexArray = field(repr=False)
    _kick_conjugate: ComplexArray = field(repr=False)
    _free_phase: ComplexArray = field(repr=False)
    _free_conjugate: ComplexArray = field(repr=False)
    _fourier_input_phase: ComplexArray = field(repr=False)
    _fourier_input_conjugate: ComplexArray = field(repr=False)
    _fourier_output_phase: ComplexArray = field(repr=False)
    _fourier_output_conjugate: ComplexArray = field(repr=False)
    _dense_cache: ComplexArray | None = field(repr=False)
    _parity_cache: ComplexArray | None = field(repr=False)

    def __init__(
        self,
        dimension: int,
        kick_strength: float,
        boundary_phases: float | BoundaryPhases = 0.0,
    ) -> None:
        size = _positive_int(dimension, name="dimension")
        strength = _finite_float(kick_strength, name="kick_strength")
        phases = (
            boundary_phases
            if isinstance(boundary_phases, BoundaryPhases)
            else BoundaryPhases(position=boundary_phases, momentum=boundary_phases)
        )
        hbar = 2.0 * np.pi / size
        positions = (np.arange(size, dtype=np.float64) + phases.position) / size
        mode_numbers = np.fft.fftfreq(size) * size + phases.momentum
        fft_indices = np.arange(size, dtype=np.float64)

        kick = np.exp(-1j * strength * np.cos(2.0 * np.pi * positions) / hbar)
        free = np.exp(-0.5j * hbar * mode_numbers**2)
        fourier_input = np.exp(-2j * np.pi * phases.momentum * fft_indices / size)
        fourier_output = np.exp(
            -2j * np.pi * phases.position * (fft_indices + phases.momentum) / size
        )
        precomputed = {
            "_kick_phase": kick,
            "_kick_conjugate": kick.conj(),
            "_free_phase": free,
            "_free_conjugate": free.conj(),
            "_fourier_input_phase": fourier_input,
            "_fourier_input_conjugate": fourier_input.conj(),
            "_fourier_output_phase": fourier_output,
            "_fourier_output_conjugate": fourier_output.conj(),
        }
        for array in precomputed.values():
            array.setflags(write=False)
        mode_numbers.setflags(write=False)

        object.__setattr__(self, "dimension", size)
        object.__setattr__(self, "kick_strength", strength)
        object.__setattr__(self, "boundary_phases", phases)
        object.__setattr__(self, "effective_hbar", hbar)
        object.__setattr__(self, "_momentum_numbers", mode_numbers)
        for name, array in precomputed.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "_dense_cache", None)
        object.__setattr__(self, "_parity_cache", None)

    @property
    def momentum_numbers(self) -> FloatArray:
        """Return the read-only momentum quantum numbers in storage order.

        The ordering is the same as :attr:`CylinderKickedRotor.momentum_numbers`
        -- the symmetric ``np.fft.fftfreq`` branch scaled by ``dimension``,
        ``0, 1, ..., N/2 - 1, -N/2, ..., -1`` for even ``N`` -- **plus the
        momentum boundary phase**: ``n_k = fftfreq(N) * N + beta``. The cylinder
        has no twist, so the two agree exactly whenever ``beta == 0``.

        These are the labels of the axis produced by :meth:`to_momentum_basis`,
        and they are the modes the free-rotation phase ``exp(-i hbar n^2 / 2)``
        is built from, so plotting ``abs(to_momentum_basis(psi))**2`` against
        this array is guaranteed to be self-consistent. The torus momentum
        coordinate in turns is ``p_k = n_k / N``, which is what
        :func:`~chaos_numerics.quantum.husimi_distribution` and
        :func:`~chaos_numerics.quantum.wigner_distribution` use.
        """
        return self._momentum_numbers

    def to_momentum_basis(self, state: ArrayLike, /) -> ComplexArray:
        """Return scalar or batched ``(..., N)`` position amplitudes as momentum ones.

        This is the unitary twisted DFT ``F`` of the class docstring,

        ``psi_tilde[k] = sum_j exp(-2 pi i (k + beta)(j + alpha) / N) psi[j] / sqrt(N)``,

        with ``alpha = boundary_phases.position`` and
        ``beta = boundary_phases.momentum``. Output index ``k`` carries momentum
        quantum number :attr:`momentum_numbers`\\ ``[k]``.

        **Do not substitute a bare** ``np.fft.fft``. Three separate things go
        wrong, measured on a random state at ``N = 64``:

        * Dropping ``norm="ortho"`` scales the amplitudes by ``sqrt(N)``, so
          every momentum probability comes out ``N = 64`` times too large.
        * The twist ``alpha`` enters as the output phase
          ``exp(-2 pi i alpha (k + beta) / N)``. It changes **only** phases: at
          ``(alpha, beta) = (0.25, 0)``, ``norm(psi_tilde - fft(psi,
          norm="ortho"))`` has median ``0.85`` over 200 random unit states
          (range ``0.74`` to ``0.99``) -- most of the norm of the state -- while
          ``abs(psi_tilde)**2`` agrees to ``4.2e-17``. A momentum *histogram*
          therefore cannot detect a missing ``alpha``, but any interference or
          overlap computation can.
        * The twist ``beta`` enters as the input phase
          ``exp(-2 pi i beta j / N)`` and **does** move the distribution. The
          total variation distance between the correct ``abs(psi_tilde)**2`` and
          the bare-FFT one has median ``0.119`` at ``beta = 0.13`` and ``0.390``
          at ``beta = 0.5`` over the same 200 states -- that is, nearly 40% of
          the probability sits in the wrong bin at the antiperiodic twist.

        With ``(alpha, beta) = (0, 0)`` the two agree bit for bit, which is
        exactly why the mistake survives a test written at the default twist.

        :meth:`to_position_basis` is the exact inverse. Round trips are unitary
        to ``2.5e-16`` and the momentum norm matches the position norm to
        ``4.5e-16``, measured over ``N`` in ``{7, 8, 15, 16, 63, 64, 65, 128,
        257}`` and three twists.

        **The two rotor classes store opposite bases**, so the same method name
        consumes a different input in each:

        ============================= ==================== ====================
        method                        :class:`KickedRotor` cylinder
        ============================= ==================== ====================
        stored basis                  position             momentum
        ``to_momentum_basis`` input    stored state         angle amplitudes
        ``to_position_basis`` input    momentum amplitudes  stored state
        ============================= ==================== ====================
        """
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        return self._position_to_momentum(values)

    def to_position_basis(self, state: ArrayLike, /) -> ComplexArray:
        """Return scalar or batched ``(..., N)`` momentum amplitudes as position ones.

        The exact inverse of :meth:`to_momentum_basis`, which documents the
        convention, the measured round-trip error, and why ``np.fft.ifft`` alone
        is not a substitute.
        """
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        return self._momentum_to_position(values)

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "kick_strength": self.kick_strength,
            "boundary_phases": self.boundary_phases.to_dict(),
            "effective_hbar": self.effective_hbar,
            "basis": "position",
            "floquet_order": "free_after_kick",
            "fft_normalization": "ortho",
        }

    @property
    def symmetry_operators(self) -> dict[str, ComplexArray]:
        """Return the exact parity symmetry, or nothing when parity is broken.

        The Floquet operator commutes with a position-space reflection for **all
        four** twists with ``2 alpha`` and ``2 beta`` integer, that is
        ``alpha, beta in {0, 1/2}``. The reflection index map is set by ``alpha``
        alone, ``j -> (-j - 2 alpha) mod N``, and ``beta`` decides whether the
        reflected amplitudes are multiplied by ``-1`` where the reflection wraps
        past the boundary:

        =========== =========== =================== =====================
        ``alpha``   ``beta``    index map           wrap factor
        =========== =========== =================== =====================
        ``0``       ``0``       ``j -> -j mod N``   ``+1``
        ``0``       ``1/2``     ``j -> -j mod N``   ``-1`` for ``j != 0``
        ``1/2``     ``0``       ``j -> N - 1 - j``  ``+1``
        ``1/2``     ``1/2``     ``j -> N - 1 - j``  ``+1`` (global sign fixed)
        =========== =========== =================== =====================

        Measured ``||[U, S]||_F / sqrt(N)`` at ``N = 128``, ``K = 10`` is
        ``6.73e-14``, ``6.75e-14``, ``6.51e-14``, and ``6.48e-14`` for those four
        rows. The ``(0, 1/2)`` entry is why this used to be under-reported: the
        plain permutation ``j -> -j`` has defect ``0.2491`` there and only the
        sign-dressed operator commutes, so a search restricted to permutations
        finds nothing and the sector split silently goes missing.

        For every other ``(alpha, beta)`` the twisted grid is genuinely not
        invariant under ``q -> -q`` -- the smallest defect over all ``N``
        reflections ``j -> c - j`` at ``N = 128`` is ``1.156`` for
        ``(0.25, 0)`` and ``1.218`` for ``(0.25, 0.13)`` -- so this mapping is
        empty rather than reporting a large commutator defect. For reference, a
        reflection applied at the wrong twist sits at ``1.579``.

        Practical consequence: whenever this mapping is non-empty the raw
        eigenphase spectrum is the superposition of two independent parity
        blocks, so raw spectral statistics sit *below* the COE reference (mean
        adjacent gap ratio ``0.42`` against ``0.5307``) and only match it after
        desymmetrizing into even and odd sectors. Choosing a generic Bloch phase
        such as ``BoundaryPhases(0.25, 0.13)`` breaks parity and time reversal
        and gives a single CUE-like spectrum that needs no desymmetrization.

        The matrix is built on demand and then cached, because it costs
        ``O(N**2)`` memory that the matrix-free FFT path never needs.
        """
        cached = self._parity_cache
        if cached is None:
            cached = _reflection_operator(self.dimension, phases=self.boundary_phases)
            if cached is None:
                return {}
            object.__setattr__(self, "_parity_cache", cached)
        return {"parity": cached}

    def apply(self, state: ArrayLike, /) -> ComplexArray:
        """Apply one Floquet step with the matrix-free FFT algorithm."""
        return self.apply_fft(state)

    def apply_fft(self, state: ArrayLike, /) -> ComplexArray:
        """Apply one step in ``O(N log N)`` time and ``O(N)`` auxiliary memory."""
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        momentum = self._position_to_momentum(values * self._kick_phase)
        return self._momentum_to_position(momentum * self._free_phase)

    def apply_dense(self, state: ArrayLike, /) -> ComplexArray:
        """Apply the memoized dense reference Floquet matrix.

        The dense matrix is assembled once per instance, so repeated calls cost
        one ``O(N**2)`` matrix-vector product instead of re-running the
        ``O(N**3)`` assembly.
        """
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        return np.asarray(values @ self._dense_matrix().T, dtype=np.complex128)

    def apply_adjoint(self, state: ArrayLike, /) -> ComplexArray:
        """Apply the exact adjoint using the reverse split-operator sequence."""
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        momentum = self._position_to_momentum(values)
        rotated = self._momentum_to_position(momentum * self._free_conjugate)
        return np.asarray(rotated * self._kick_conjugate, dtype=np.complex128)

    def to_dense(self) -> ComplexArray:
        """Return an independent writable copy of the dense reference matrix."""
        return self._dense_matrix().copy()

    def _dense_matrix(self) -> ComplexArray:
        """Return the shared read-only dense reference, assembling it once."""
        cached = self._dense_cache
        if cached is not None:
            return cached
        dense = self._assemble_dense()
        dense.setflags(write=False)
        object.__setattr__(self, "_dense_cache", dense)
        return dense

    def _assemble_dense(self) -> ComplexArray:
        """Assemble the ``O(N**2)`` dense reference in the position basis."""
        indices = np.arange(self.dimension, dtype=np.float64)
        exponent = np.outer(
            indices + self.boundary_phases.momentum,
            indices + self.boundary_phases.position,
        )
        fourier = np.exp(-2j * np.pi * exponent / self.dimension) / np.sqrt(self.dimension)
        dense = fourier.conj().T @ (self._free_phase[:, None] * fourier)
        return np.asarray(dense * self._kick_phase[None, :], dtype=np.complex128)

    def as_linear_operator(self) -> LinearOperator:
        """Return an FFT-backed SciPy ``LinearOperator`` with an adjoint."""
        shape = (self.dimension, self.dimension)
        return LinearOperator(
            shape=shape,
            dtype=np.dtype(np.complex128),
            matvec=self.apply_fft,
            rmatvec=self.apply_adjoint,
            matmat=lambda values: self.apply_fft(values.T).T,
            rmatmat=lambda values: self.apply_adjoint(values.T).T,
        )

    def _position_to_momentum(self, state: ComplexArray) -> ComplexArray:
        transformed = np.fft.fft(
            state * self._fourier_input_phase,
            axis=-1,
            norm="ortho",
        )
        return np.asarray(transformed * self._fourier_output_phase, dtype=np.complex128)

    def _momentum_to_position(self, state: ComplexArray) -> ComplexArray:
        transformed = np.fft.ifft(
            state * self._fourier_output_conjugate,
            axis=-1,
            norm="ortho",
        )
        return np.asarray(transformed * self._fourier_input_conjugate, dtype=np.complex128)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CylinderKickedRotor:
    r"""Kicked rotor on the momentum cylinder with ``hbar`` free of the basis size.

    One Floquet step is ``U = exp(-i hbar n^2 / 2) exp(-i (K / hbar) cos(theta))``
    applied in that order: kick first, free rotation second, matching
    :class:`KickedRotor`.

    **States are momentum amplitudes.** ``psi[j]`` is the amplitude of the
    momentum eigenstate ``n = momentum_numbers[j]``, not a position amplitude.
    This is the opposite convention from :class:`KickedRotor`, which stores
    position amplitudes, and it is the single easiest thing to get wrong when
    moving a state between the two classes. :meth:`to_position_basis` and
    :meth:`to_momentum_basis` name the basis they *return*, so the pair reads
    the same way on both classes even though the stored basis is reversed.

    **The momentum lattice is finite, so results are physical only before the
    wave packet reaches its edges.** The lattice spans
    ``-dimension/2 <= n < dimension/2`` and wraps periodically, so probability
    that leaves one edge reappears at the other and every moment of ``n``
    silently becomes wrong. Keep ``dimension`` several times the expected
    spread: the ``|psi_n|^2`` tail at the edge is the diagnostic to watch.

    **Choosing between the two rotors.** :class:`KickedRotor` quantizes the
    torus with ``hbar_eff = 2 pi / N``, so its localization length always
    exceeds the basis size and dynamical localization cannot appear; use it for
    spectral statistics, where a finite unitary with a full eigenphase spectrum
    is exactly what is needed. This class decouples ``hbar`` from ``dimension``,
    which is what makes the localized regime ``ell << dimension`` reachable; use
    it for dynamical localization and momentum transport. Its eigenphases can be
    computed but should not be fed to level statistics: the momentum lattice is a
    truncation of an infinite one rather than a torus quantization, so the
    spectrum belongs to the truncation as much as to the rotor.

    **Localization length: two different lengths, both called ``ell``.** The
    saturated momentum profile is exponential, ``mean(|psi_n|^2) ~ exp(-|n| /
    ell_dist)``, and so is a single Floquet eigenfunction around its own centre,
    ``|phi_n|^2 ~ exp(-|n - n_0| / ell_eig)``. These are *not* the same length:
    the distribution left behind by a wave packet is a superposition of
    eigenfunctions and decays about twice as slowly as one of them.

    **It is ``ell_eig``, not ``ell_dist``, that the textbook estimate
    ``ell ~ D / 2`` predicts**, with ``D`` defined by ``<(Delta n)^2> = D t`` --
    one factor of ``t``, no factor of two. Fitting the saturated profile and
    calling the result ``D / 2`` therefore overestimates ``D`` by roughly a
    factor of two.

    Measured with ``hbar = 1`` and the Bessel-corrected quasilinear rate
    ``D = (K^2 / 2 hbar^2) (1 - 2 J_2(K / hbar))``, time-averaging the profile
    over the second half of the run:

    ===== ======= ============ ============= ============= =============
    ``K`` ``D``   ``ell_eig``  ``ell_dist``  ``<n^2>_sat`` ``D**2``
    ===== ======= ============ ============= ============= =============
    5     11.336  4.48         8.59          117.9         128.5
    8     39.231  24.33        30.38         1741.6        1539.1
    ===== ======= ============ ============= ============= =============

    So ``ell_eig`` is within about 25% of ``D / 2`` (0.79 and 1.24 times it),
    ``ell_dist`` is ``0.76`` and ``0.77`` times ``D`` -- that is, close to ``D``
    and nowhere near ``D / 2`` -- and ``ell_dist / ell_eig`` is 1.9 and 1.2. The
    moment follows the distribution, not the eigenfunctions: ``<n^2>_sat`` is
    ``0.92`` and ``1.13`` times ``D**2``, so ``sqrt(<n^2>_sat / 2)`` comes out at
    ``1.35`` and ``1.50`` times ``D / 2``. Reading ``<n^2>_sat = 2 (D / 2)^2 =
    D**2 / 2`` off the exponential ansatz is off by that same factor of two.

    ``ell_eig`` was measured as the log-linear slope of the geometric mean over
    the whole spectrum of the peak-centred ``|phi_n|^2`` at ``N = 512``
    (``K = 5``) and ``N = 1024`` (``K = 8``); ``ell_dist`` from the same
    self-consistent fit band ``ell < |n| < 6 ell`` applied to the time-averaged
    profile at ``N = 1024`` / 1500 kicks and ``N = 2048`` / 2000 kicks. The
    geometric mean is deliberate: the arithmetic mean of the eigenfunction
    profiles is dominated by the widest state in the spectrum and has no decay
    length at all.

    The transport helpers in :mod:`chaos_numerics.classical` use the momentum
    convention ``<Delta P^2> = 2 D_P t`` instead, so converting requires
    ``D = 2 D_P / hbar**2``. Mixing the two conventions misses ``ell`` by a
    factor of two, which is larger than the accuracy of the estimate itself.
    """

    dimension: int
    kick_strength: float
    effective_hbar: float
    _momentum_numbers: FloatArray = field(repr=False)
    _kick_phase: ComplexArray = field(repr=False)
    _kick_conjugate: ComplexArray = field(repr=False)
    _free_phase: ComplexArray = field(repr=False)
    _free_conjugate: ComplexArray = field(repr=False)

    def __init__(
        self,
        dimension: int,
        kick_strength: float,
        effective_hbar: float = 1.0,
    ) -> None:
        size = _positive_int(dimension, name="dimension")
        strength = _finite_float(kick_strength, name="kick_strength")
        hbar = _finite_float(effective_hbar, name="effective_hbar")
        if hbar <= 0.0:
            raise ValidationError(f"effective_hbar must be positive; got {hbar}")
        momentum_numbers = np.fft.fftfreq(size) * size
        angles = 2.0 * np.pi * np.arange(size, dtype=np.float64) / size

        kick = np.exp(-1j * strength * np.cos(angles) / hbar)
        free = np.exp(-0.5j * hbar * momentum_numbers**2)
        precomputed = {
            "_momentum_numbers": momentum_numbers,
            "_kick_phase": kick,
            "_kick_conjugate": kick.conj(),
            "_free_phase": free,
            "_free_conjugate": free.conj(),
        }
        for array in precomputed.values():
            array.setflags(write=False)

        object.__setattr__(self, "dimension", size)
        object.__setattr__(self, "kick_strength", strength)
        object.__setattr__(self, "effective_hbar", hbar)
        for name, array in precomputed.items():
            object.__setattr__(self, name, array)

    @property
    def momentum_numbers(self) -> FloatArray:
        """Return the read-only momentum quantum numbers in storage order.

        The ordering is the symmetric ``np.fft.fftfreq`` branch scaled by
        ``dimension``: ``0, 1, ..., N/2 - 1, -N/2, ..., -1`` for even ``N``.
        :attr:`KickedRotor.momentum_numbers` uses the same ordering with the
        momentum boundary phase added, so the two agree when that twist is zero.
        """
        return self._momentum_numbers

    def to_position_basis(self, state: ArrayLike, /) -> ComplexArray:
        """Return scalar or batched ``(..., N)`` momentum amplitudes as angle ones.

        The angle grid is ``theta_j = 2 pi j / N`` and the transform is the
        untwisted unitary inverse DFT, ``np.fft.ifft(..., norm="ortho")``: this
        cylinder has no boundary phases, so unlike
        :meth:`KickedRotor.to_momentum_basis` there is no ``(alpha, beta)`` to
        get wrong.

        **This class stores momentum amplitudes**, so this method consumes the
        stored state and :meth:`to_momentum_basis` produces one -- the opposite
        of :class:`KickedRotor`, where the storage basis is position. Moving a
        state between the two classes without transforming it is the single
        easiest mistake to make; the naming here says which basis comes *out*,
        not which one goes in.
        """
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        return self._momentum_to_angle(values)

    def to_momentum_basis(self, state: ArrayLike, /) -> ComplexArray:
        """Return scalar or batched ``(..., N)`` angle amplitudes as momentum ones.

        The exact inverse of :meth:`to_position_basis`, which documents the
        convention and the basis-role contrast with :class:`KickedRotor`. Output
        index ``k`` carries momentum quantum number
        :attr:`momentum_numbers`\\ ``[k]``.
        """
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        return self._angle_to_momentum(values)

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "kick_strength": self.kick_strength,
            "effective_hbar": self.effective_hbar,
            "basis": "momentum",
            "geometry": "cylinder",
            "floquet_order": "free_after_kick",
            "fft_normalization": "ortho",
        }

    def apply(self, state: ArrayLike, /) -> ComplexArray:
        """Apply one Floquet step to scalar or batched momentum amplitudes."""
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        angle_amplitudes = self._momentum_to_angle(values) * self._kick_phase
        return np.asarray(
            self._angle_to_momentum(angle_amplitudes) * self._free_phase,
            dtype=np.complex128,
        )

    def apply_fft(self, state: ArrayLike, /) -> ComplexArray:
        """Apply one step in ``O(N log N)`` time; identical to :meth:`apply`.

        This class has no dense reference matrix, so unlike
        :class:`KickedRotor` -- where ``apply``, ``apply_fft``, and
        ``apply_dense`` are three named routes to the same operator -- there is
        only one algorithm and ``apply_fft`` is an exact alias. It exists so that
        ``evolve(..., method="fft")``, which the README and the examples teach as
        the way to run long matrix-free evolutions, works on every FFT-backed
        model instead of failing here with ``does not provide apply_fft``.
        """
        return self.apply(state)

    def apply_adjoint(self, state: ArrayLike, /) -> ComplexArray:
        """Apply the exact adjoint using the reverse split-operator sequence."""
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        angle_amplitudes = self._momentum_to_angle(values * self._free_conjugate)
        return np.asarray(
            self._angle_to_momentum(angle_amplitudes * self._kick_conjugate),
            dtype=np.complex128,
        )

    def as_linear_operator(self) -> LinearOperator:
        """Return an FFT-backed SciPy ``LinearOperator`` with an adjoint."""
        return LinearOperator(
            shape=(self.dimension, self.dimension),
            dtype=np.dtype(np.complex128),
            matvec=self.apply,
            rmatvec=self.apply_adjoint,
            matmat=lambda values: self.apply(values.T).T,
            rmatmat=lambda values: self.apply_adjoint(values.T).T,
        )

    def _momentum_to_angle(self, state: ComplexArray) -> ComplexArray:
        return np.asarray(np.fft.ifft(state, axis=-1, norm="ortho"), dtype=np.complex128)

    def _angle_to_momentum(self, state: ComplexArray) -> ComplexArray:
        return np.asarray(np.fft.fft(state, axis=-1, norm="ortho"), dtype=np.complex128)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a positive integer; got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValidationError(f"{name} must be positive; got {result}")
    return result


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    return result


__all__ = ["CylinderKickedRotor", "KickedRotor"]

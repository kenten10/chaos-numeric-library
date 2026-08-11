"""Periodized torus coherent states, Husimi data, and localization measures."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import TypeAlias

import numpy as np

from chaos_numerics.core import ExperimentMetadata, ValidationError
from chaos_numerics.core._payload import (
    ArrayPayload,
    add_convergence_history,
    metadata_payload,
)
from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.types import ArrayLike, ComplexArray, FloatArray
from chaos_numerics.quantum.states import BoundaryPhases, normalize_state, quantum_state


@dataclass(frozen=True, slots=True, eq=False)
class HusimiResult:
    """Plot-ready normalized density on a half-open ``(q, p)`` midpoint grid."""

    positions: FloatArray
    momenta: FloatArray
    values: FloatArray
    raw_integral: FloatArray
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def __reduce__(self) -> tuple[type[HusimiResult], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only.

        The default ``slots=True`` reduction restores the ``__dict__``-free state
        field by field, which bypasses ``__post_init__`` and hands back writable
        arrays: a ``copy.deepcopy`` round trip used to break the read-only
        contract that every other result type in the library keeps.
        """
        return (
            self.__class__,
            (self.positions, self.momenta, self.values, self.raw_integral, self.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HusimiResult):
            return NotImplemented
        return (
            np.array_equal(self.positions, other.positions)
            and np.array_equal(self.momenta, other.momenta)
            and np.array_equal(self.values, other.values)
            and np.array_equal(self.raw_integral, other.raw_integral)
            and self.metadata == other.metadata
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable copies of every stored array."""
        payload: ArrayPayload = {
            "positions": self.positions.copy(),
            "momenta": self.momenta.copy(),
            "values": self.values.copy(),
            "raw_integral": self.raw_integral.copy(),
        }
        add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return the JSON descriptor, with array shapes but no array contents.

        The convergence history is described here as well, because
        :meth:`array_payload` writes it: a descriptor that omits an array the
        payload carries makes the two halves of the persistence split disagree,
        and the storage layer fails with a ``KeyError`` instead of a
        ``ValidationError``.
        """
        arrays: ArrayPayload = {
            "positions": self.positions,
            "momenta": self.momenta,
            "values": self.values,
            "raw_integral": self.raw_integral,
        }
        add_convergence_history(arrays, self.metadata, copy=False)
        return metadata_payload("husimi", self.metadata, arrays)

    def __post_init__(self) -> None:
        positions = _readonly_axis(self.positions, name="positions")
        momenta = _readonly_axis(self.momenta, name="momenta")
        values = as_float_array(self.values, name="Husimi values", copy=True)
        if values.ndim < 2 or values.shape[-2:] != (positions.size, momenta.size):
            raise ValidationError(
                "Husimi values must have trailing grid shape "
                f"({positions.size}, {momenta.size}); got {values.shape}"
            )
        if np.any(values < 0.0):
            raise ValidationError("Husimi values must be non-negative")
        raw_integral = _readonly_integral(
            self.raw_integral,
            shape=values.shape[:-2],
        )
        if not isinstance(self.metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        values.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "momenta", momenta)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "raw_integral", raw_integral)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.positions.size), int(self.momenta.size))

    @property
    def cell_area(self) -> float:
        return 1.0 / (self.positions.size * self.momenta.size)

    @property
    def integral(self) -> float | FloatArray:
        """Return the midpoint quadrature of the stored ``values``.

        Under the default ``normalize=True`` of :func:`husimi_distribution` this
        is 1 by construction, to within the summation rounding, and it is
        therefore **not** a check on the quadrature. :attr:`raw_integral` is the
        pre-normalization value and is the quantity that actually says whether
        the grid and the ``images`` count resolved the state.
        """
        result = np.sum(self.values, axis=(-2, -1)) * self.cell_area
        return float(result) if result.ndim == 0 else np.asarray(result, dtype=np.float64)

    @property
    def grid_points(self) -> FloatArray:
        """Return ``(n_q, n_p, 2)`` points in classical ``(q, p)`` order."""
        q_grid, p_grid = np.meshgrid(self.positions, self.momenta, indexing="ij")
        return np.stack((q_grid, p_grid), axis=-1)


@dataclass(frozen=True, slots=True, eq=False)
class WignerResult:
    """Real, signed discrete Wigner weights on the ``N x N`` ``(q, p)`` torus lattice.

    Same container contract as :class:`HusimiResult`: read-only arrays,
    value-based ``__eq__``, and a ``__reduce__`` that rebuilds through
    ``__init__``. ``negative_weight`` is ``sum max(-W, 0)`` per state, which is
    ``0.0`` exactly for a basis state and grows with interference; see
    :func:`wigner_distribution` for measured values and for why it may only be
    compared between states of the same dimension.
    """

    positions: FloatArray
    momenta: FloatArray
    values: FloatArray
    negative_weight: FloatArray
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def __reduce__(self) -> tuple[type[WignerResult], tuple[object, ...]]:
        """Rebuild through ``__init__`` so unpickled arrays stay read-only.

        Identical reasoning to :meth:`HusimiResult.__reduce__`: the default
        ``slots=True`` reduction restores fields one by one, skips
        ``__post_init__``, and hands back writable arrays.
        """
        return (
            self.__class__,
            (self.positions, self.momenta, self.values, self.negative_weight, self.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WignerResult):
            return NotImplemented
        return (
            np.array_equal(self.positions, other.positions)
            and np.array_equal(self.momenta, other.momenta)
            and np.array_equal(self.values, other.values)
            and np.array_equal(self.negative_weight, other.negative_weight)
            and self.metadata == other.metadata
        )

    def array_payload(self) -> ArrayPayload:
        """Return independent writable copies of every stored array."""
        payload: ArrayPayload = {
            "positions": self.positions.copy(),
            "momenta": self.momenta.copy(),
            "values": self.values.copy(),
            "negative_weight": self.negative_weight.copy(),
        }
        add_convergence_history(payload, self.metadata)
        return payload

    def metadata_payload(self) -> dict[str, object]:
        """Return the JSON descriptor, with array shapes but no array contents.

        Includes the convergence history whenever :meth:`array_payload` does; see
        :meth:`HusimiResult.metadata_payload` for why the two must agree.
        """
        arrays: ArrayPayload = {
            "positions": self.positions,
            "momenta": self.momenta,
            "values": self.values,
            "negative_weight": self.negative_weight,
        }
        add_convergence_history(arrays, self.metadata, copy=False)
        return metadata_payload("wigner", self.metadata, arrays)

    def __post_init__(self) -> None:
        positions = _readonly_axis(self.positions, name="positions")
        momenta = _readonly_axis(self.momenta, name="momenta")
        values = as_float_array(self.values, name="Wigner values", copy=True)
        if values.ndim < 2 or values.shape[-2:] != (positions.size, momenta.size):
            raise ValidationError(
                "Wigner values must have trailing grid shape "
                f"({positions.size}, {momenta.size}); got {values.shape}"
            )
        # Deliberately no sign check, unlike HusimiResult: a Wigner function that
        # cannot go negative would carry no more information than a Husimi
        # density. ``as_float_array`` still rejects non-finite entries.
        negative_weight = _readonly_integral(self.negative_weight, shape=values.shape[:-2])
        if not isinstance(self.metadata, ExperimentMetadata):
            raise ValidationError("metadata must be an ExperimentMetadata instance")
        values.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "momenta", momenta)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "negative_weight", negative_weight)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (int(self.positions.size), int(self.momenta.size))

    @property
    def total(self) -> float | FloatArray:
        """Return the plain grid sum, which is 1 for any normalized state.

        Unlike :attr:`HusimiResult.integral` there is no cell area: the stored
        values are dimensionless weights, not a density, and the sum is an exact
        identity rather than the result of a quadrature. Measured departure from
        1 is at most ``2.2e-16``, so this is a check on the summation and on
        nothing else.
        """
        result = np.sum(self.values, axis=(-2, -1))
        return float(result) if result.ndim == 0 else np.asarray(result, dtype=np.float64)

    @property
    def grid_points(self) -> FloatArray:
        """Return ``(N, N, 2)`` points in classical ``(q, p)`` order."""
        q_grid, p_grid = np.meshgrid(self.positions, self.momenta, indexing="ij")
        return np.stack((q_grid, p_grid), axis=-1)


def coherent_state(
    *,
    dimension: int,
    position: float,
    momentum: float,
    boundary_phases: BoundaryPhases | None = None,
    images: int = 4,
) -> ComplexArray:
    """Return one normalized periodized Gaussian in the position basis.

    ``position`` and ``momentum`` are torus coordinates in turns and are **wrapped
    into ``[0, 1)`` modulo one**, silently and by design: the torus has no outside,
    so ``position=1.2`` names the same point as ``position=0.2`` and ``-0.3`` the
    same point as ``0.7``. A coordinate outside the unit square is therefore not
    an error. Only non-finite values are rejected.
    """
    size = _positive_int(dimension, name="dimension")
    q_center = _phase_point(position, name="position")
    p_center = _phase_point(momentum, name="momentum")
    phases = BoundaryPhases() if boundary_phases is None else boundary_phases
    if not isinstance(phases, BoundaryPhases):
        raise ValidationError("boundary_phases must be a BoundaryPhases instance")
    image_count = _positive_int(images, name="images")
    states = _coherent_batch(
        size,
        np.asarray([q_center]),
        np.asarray([p_center]),
        boundary_phases=phases,
        images=image_count,
    )
    return np.asarray(states[0], dtype=np.complex128)


def husimi_distribution(
    state: ArrayLike,
    *,
    grid_shape: tuple[int, int] = (64, 64),
    boundary_phases: BoundaryPhases | None = None,
    images: int = 4,
    normalize: bool = True,
    chunk_size: int = 256,
) -> HusimiResult:
    """Return ``N |<q,p|psi>|^2`` on a configurable midpoint torus grid.

    **Pass every state you need in one call.** Almost all of the cost is
    rebuilding the ``(n_q * n_p, N)`` grid of coherent states, which depends only
    on ``dimension``, ``boundary_phases``, ``grid_shape``, and ``images`` -- not
    on the state. A batch ``(..., N)`` builds it once and then spends one
    ``complex128`` matrix product per state. Measured at ``N = 1024``, best of
    three in one process: on a ``(64, 64)`` grid, ``0.55 s`` for one state and
    ``0.57 s`` for a stack of 20, a **19.3x** per-state saving; on
    ``(128, 128)``, ``2.25 s`` for one and ``2.24 s`` for 20, **20.1x**. Looping
    over eigenstates one call at a time is the single most expensive mistake
    available here: pass the whole spectrum,
    ``np.asarray(eigenstates(model).eigenstates).T``, as one ``(N, N)`` stack of
    rows instead.

    Repeated calls with identical model parameters also hit a small module-level
    LRU cache of that grid, so a second call at ``N = 1024`` / ``(64, 64)`` costs
    ``0.002 s`` instead of ``0.55 s``. The cache holds at most four grids and at
    most 128 MiB, is keyed on exactly the four parameters above, and hands out
    read-only arrays, so results are bit-for-bit identical whether or not an
    entry was reused. A grid larger than the budget -- ``N = 1024`` on
    ``(128, 128)`` needs 256 MiB -- is never cached, and batching stays the only
    saving available there.
    """
    values = quantum_state(state, normalized=True)
    size = int(values.shape[-1])
    n_q, n_p = _grid_shape(grid_shape)
    phases = BoundaryPhases() if boundary_phases is None else boundary_phases
    if not isinstance(phases, BoundaryPhases):
        raise ValidationError("boundary_phases must be a BoundaryPhases instance")
    image_count = _positive_int(images, name="images")
    block = _positive_int(chunk_size, name="chunk_size")
    if not isinstance(normalize, bool):
        raise ValidationError("normalize must be a bool")

    positions = (np.arange(n_q, dtype=np.float64) + 0.5) / n_q
    momenta = (np.arange(n_p, dtype=np.float64) + 0.5) / n_p
    q_grid, p_grid = np.meshgrid(positions, momenta, indexing="ij")
    centers_q = q_grid.ravel()
    centers_p = p_grid.ravel()
    flattened = values.reshape((-1, size))
    density = np.empty((flattened.shape[0], centers_q.size), dtype=np.float64)
    cached_grid = _cached_coherent_grid(
        size,
        centers_q,
        centers_p,
        grid_shape=(n_q, n_p),
        boundary_phases=phases,
        images=image_count,
        chunk_size=block,
    )
    for start in range(0, centers_q.size, block):
        stop = min(start + block, centers_q.size)
        coherent = (
            cached_grid[start:stop]
            if cached_grid is not None
            else _coherent_batch(
                size,
                centers_q[start:stop],
                centers_p[start:stop],
                boundary_phases=phases,
                images=image_count,
            )
        )
        overlaps = flattened @ coherent.conj().T
        density[:, start:stop] = size * np.abs(overlaps) ** 2
    density = density.reshape((*values.shape[:-1], n_q, n_p))
    raw_integral = np.mean(density, axis=(-2, -1))
    if normalize:
        if bool(np.any(raw_integral <= np.finfo(np.float64).tiny)):
            raise ValidationError("Husimi quadrature has zero integral")
        density = density / raw_integral[..., None, None]
    metadata = ExperimentMetadata(
        parameters={
            "dimension": size,
            "grid_shape": (n_q, n_p),
            "grid": "half-open midpoint",
            "coordinate_order": "(q, p)",
            "basis": "position",
            "boundary_phases": phases.to_dict(),
            "images_each_side": image_count,
            "definition": "N * abs(<q,p|psi>)**2",
            "quadrature": "uniform midpoint",
            "normalized": normalize,
            "chunk_size": block,
        },
        precision="float64",
    )
    return HusimiResult(positions, momenta, density, raw_integral, metadata)


def wigner_distribution(
    state: ArrayLike,
    *,
    boundary_phases: BoundaryPhases | None = None,
    chunk_size: int = 256,
) -> WignerResult:
    r"""Return the real discrete Wigner function on the ``N x N`` torus lattice.

    **Even dimensions only.** An odd ``N`` is rejected rather than silently
    handed the wrong convention; see the block-sum paragraph below for why.

    **Convention.** The natural Weyl lattice of an ``N``-state torus is the
    *half-integer* one, ``2N x 2N`` points at ``q = (m/2 + alpha) / N`` and
    ``p = (n/2 + beta) / N``, because the midpoint of two position grid points
    generally is not a grid point. On it,

    ``A[m, n] = sum_j psi_j conj(psi_[m - j]) exp(-2 pi i (n/2 + beta)(2 j - m) / N) / (2 N)``

    with the index on ``conj`` extended quasi-periodically,
    ``psi_[k + N] = exp(2 pi i beta) psi_k``. The returned values are the sums of
    ``A`` over ``2 x 2`` blocks,

    ``W[j, k] = sum_(a, b in {0, 1}) A[2 j + a, 2 k + b]``,

    which lands back on the ordinary position/momentum lattice
    ``q_j = (j + alpha) / N``, ``p_k = (k + beta) / N`` -- the same ``q`` grid
    :class:`~chaos_numerics.quantum.KickedRotor` stores states on, and ``p_k``
    equal to ``KickedRotor.momentum_numbers[k] / N`` modulo one, in the DFT
    storage order of
    :meth:`~chaos_numerics.quantum.KickedRotor.to_momentum_basis`.

    **The block sum is not cosmetic, and it is why ``N`` must be even.** A
    quasi-periodic state has periodic images one full period apart, and the
    midpoint of a point and its neighbouring image is half a period away, so
    ``A`` carries a full-strength *ghost* copy of every feature at
    ``(q + 1/2, p)`` and ``(q, p + 1/2)``, obeying exactly
    ``A[m + N, n] = (-1)**n A[m, n]`` (verified to ``3.6e-16``). Those ghosts
    oscillate at the Nyquist frequency of the ``p`` axis, so summing adjacent
    ``n`` cancels them -- and the pairing ``{2k, 2k + 1}`` is compatible with the
    ``+N`` shift only when ``N`` is even. Measured on ``A`` itself, a single
    coherent state at ``N = 32`` has four *equal* maxima at ``(0.25, 0.25)``,
    ``(0.25, 0.75)``, ``(0.75, 0.25)``, ``(0.75, 0.75)`` and negative weight
    ``1.50``, indistinguishable from a cat state's ``1.47``; after the block sum
    the maximum is unique and at the requested centre, and the negative weight
    separates as ``0.13`` against ``0.61``. Reporting ``A`` directly is the
    natural-looking mistake here.

    **Exact identities** (worst case over ``N`` in ``{2, 4, 6, 8, 16, 32}``, five
    twists including ``(0.25, 0.13)``, random states):

    * the values agree with an independent ``O(N**3)`` transcription of the two
      formulas above to ``2.5e-16``, and the discarded imaginary part of ``A`` is
      at most ``3.9e-16``;
    * ``sum_k W[j, k] == abs(psi_j)**2`` to ``2.2e-16``;
    * ``sum_j W[j, k] == abs(to_momentum_basis(psi)[k])**2`` to ``5.6e-16``;
    * ``sum_(j, k) W == 1`` to ``3.3e-16``;
    * ``N * sum_(j, k) W_psi W_phi == abs(<psi|phi>)**2`` to ``2.2e-16``, so the
      purity of a pure state reads ``N * sum W**2 == 1``. **The proportionality
      constant is ``1 / N``**, i.e. ``sum W_psi W_phi = abs(<psi|phi>)**2 / N``.

    The two marginals are the reason to prefer this over an ad hoc convention:
    they fail for essentially any sign, factor, or index error, including a
    dropped boundary twist.

    **Why this and not only Husimi.** ``W`` takes negative values where a state
    interferes with itself, which is what makes fringes and scars visible;
    :func:`husimi_distribution` is a Gaussian smoothing of the same information
    and is non-negative by construction. Measured for the cat state
    ``(coherent(0.25, 0.5) + coherent(0.75, 0.5))``: ``min W = -3.98e-2`` at
    ``N = 16``, ``-2.51e-2`` at ``N = 32``, ``-1.40e-2`` at ``N = 64``, against a
    peak of ``+5.97e-2``, ``+3.05e-2``, ``+1.54e-2`` -- the fringe trough is
    two thirds of the peak height, not a rounding artifact -- while the Husimi
    density of the same states on the same grid stays positive (minimum
    ``2.2e-5``, ``7.1e-12``, ``3.3e-25``).

    A position or momentum basis state has ``W >= 0`` exactly
    (:attr:`WignerResult.negative_weight` is ``0.0``), and a coherent state's
    negative weight shrinks as ``hbar_eff = 2 pi / N`` does: ``0.19``, ``0.13``,
    ``0.092`` at ``N = 16, 32, 64``. Some negativity at finite ``N`` is
    therefore expected even for the most classical states available, and
    ``negative_weight`` is only comparable between states of equal ``N``.

    ``chunk_size`` is the number of output ``q`` rows built per pass; auxiliary
    storage is ``O(chunk_size * N**2)`` complex, and the result itself is
    ``O(N**2)`` per state. Batches ``(..., N)`` are supported and share nothing,
    so unlike :func:`husimi_distribution` there is no batching speedup.
    """
    values = quantum_state(state, normalized=True)
    size = int(values.shape[-1])
    if size % 2 != 0:
        raise ValidationError(
            "wigner_distribution requires an even dimension; got "
            f"{size}. The 2 x 2 block sum that cancels the periodic-image ghosts "
            "is only consistent with the half-period shift for even N"
        )
    phases = BoundaryPhases() if boundary_phases is None else boundary_phases
    if not isinstance(phases, BoundaryPhases):
        raise ValidationError("boundary_phases must be a BoundaryPhases instance")
    block = _positive_int(chunk_size, name="chunk_size")

    doubled = 2 * size
    indices = np.arange(size, dtype=np.float64)
    positions = (indices + phases.position) / size
    momenta = (indices + phases.momentum) / size
    flattened = values.reshape((-1, size))
    batch = flattened.shape[0]
    wigner = np.empty((batch, size, size), dtype=np.float64)
    left = np.arange(size)
    for start in range(0, size, block):
        stop = min(start + block, size)
        midpoints = np.arange(2 * start, 2 * stop)[:, None]
        chord = 2 * left[None, :] - midpoints
        partner = midpoints - left[None, :]
        conjugate = flattened[:, partner % size] * np.exp(
            2j * np.pi * phases.momentum * (partner // size)
        )
        products = (
            flattened[:, None, :]
            * conjugate.conj()
            * np.exp(-2j * np.pi * phases.momentum * chord / size)
        )
        chords = np.zeros((batch, 2 * (stop - start), doubled), dtype=np.complex128)
        np.put_along_axis(
            chords,
            np.broadcast_to(chord % doubled, products.shape),
            products,
            axis=-1,
        )
        weyl = np.fft.fft(chords, axis=-1).real / doubled
        wigner[:, start:stop] = weyl.reshape((batch, stop - start, 2, size, 2)).sum(axis=(2, 4))
    wigner = wigner.reshape((*values.shape[:-1], size, size))
    negative_weight = np.sum(np.maximum(-wigner, 0.0), axis=(-2, -1))
    metadata = ExperimentMetadata(
        parameters={
            "dimension": size,
            "grid_shape": (size, size),
            "grid": "twisted position/momentum lattice",
            "coordinate_order": "(q, p)",
            "basis": "position",
            "boundary_phases": phases.to_dict(),
            "definition": (
                "2x2 block sum of sum_j psi_j conj(psi_[m-j]) "
                "exp(-2 pi i (n/2 + beta) (2 j - m) / N) / (2 N)"
            ),
            "normalization": "sum over the N x N lattice equals one",
            "chunk_size": block,
        },
        precision="float64",
    )
    return WignerResult(positions, momenta, wigner, negative_weight, metadata)


def inverse_participation_ratio(state: ArrayLike) -> float | FloatArray:
    """Return ``sum_j |psi_j|^4`` along the trailing Hilbert-space axis."""
    probabilities = np.abs(quantum_state(state, normalized=True)) ** 2
    result = np.sum(probabilities**2, axis=-1)
    return _scalar_or_array(result)


def participation_ratio(state: ArrayLike) -> float | FloatArray:
    """Return the effective occupied basis size ``1 / IPR``."""
    ipr = np.asarray(inverse_participation_ratio(state), dtype=np.float64)
    return _scalar_or_array(1.0 / ipr)


def shannon_entropy(state: ArrayLike) -> float | FloatArray:
    """Return basis Shannon entropy in nats with ``0 log(0) = 0``."""
    probabilities = np.abs(quantum_state(state, normalized=True)) ** 2
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return _scalar_or_array(-np.sum(terms, axis=-1))


_CACHE_BUDGET_BYTES = 128 * 1024 * 1024
"""Largest total size of retained coherent-state grids, in bytes.

128 MiB admits the common ``N = 1024`` / ``(64, 64)`` grid, which is exactly
64 MiB, and refuses ``(128, 128)`` at 256 MiB. It is a module attribute rather
than a parameter because it bounds a hidden allocation: a caller who wants to
opt out sets it to ``0``, which is also how the tests prove that the cached and
uncached paths agree bit for bit."""

_CACHE_MAX_ENTRIES = 4

_CacheKey: TypeAlias = tuple[int, int, int, int, float, float]

_coherent_cache: OrderedDict[_CacheKey, ComplexArray] = OrderedDict()
_coherent_cache_lock = Lock()


def _clear_coherent_cache() -> None:
    """Drop every retained coherent-state grid."""
    with _coherent_cache_lock:
        _coherent_cache.clear()


def _cached_coherent_grid(
    dimension: int,
    positions: FloatArray,
    momenta: FloatArray,
    *,
    grid_shape: tuple[int, int],
    boundary_phases: BoundaryPhases,
    images: int,
    chunk_size: int,
) -> ComplexArray | None:
    """Return the read-only full coherent-state grid, or ``None`` if too large.

    The grid depends only on the key below, so an entry may be reused across
    calls with different states and different ``chunk_size`` values: each row is
    an independent elementwise expression reduced over the image axis alone, so
    assembling it in blocks of any width gives bit-identical numbers.
    """
    key = (
        dimension,
        grid_shape[0],
        grid_shape[1],
        images,
        boundary_phases.position,
        boundary_phases.momentum,
    )
    nbytes = dimension * int(positions.size) * 16
    if nbytes > _CACHE_BUDGET_BYTES:
        return None
    with _coherent_cache_lock:
        cached = _coherent_cache.get(key)
        if cached is not None:
            _coherent_cache.move_to_end(key)
            return cached
    grid = np.empty((positions.size, dimension), dtype=np.complex128)
    for start in range(0, positions.size, chunk_size):
        stop = min(start + chunk_size, positions.size)
        grid[start:stop] = _coherent_batch(
            dimension,
            positions[start:stop],
            momenta[start:stop],
            boundary_phases=boundary_phases,
            images=images,
        )
    grid.setflags(write=False)
    with _coherent_cache_lock:
        _coherent_cache[key] = grid
        _coherent_cache.move_to_end(key)
        while len(_coherent_cache) > _CACHE_MAX_ENTRIES or (
            sum(entry.nbytes for entry in _coherent_cache.values()) > _CACHE_BUDGET_BYTES
        ):
            _coherent_cache.popitem(last=False)
    return grid


def _coherent_batch(
    dimension: int,
    positions: FloatArray,
    momenta: FloatArray,
    *,
    boundary_phases: BoundaryPhases,
    images: int,
) -> ComplexArray:
    basis_positions = (
        np.arange(dimension, dtype=np.float64) + boundary_phases.position
    ) / dimension
    image_indices = np.arange(-images, images + 1, dtype=np.float64)
    displacement = (
        basis_positions[None, :, None] - positions[:, None, None] + image_indices[None, None, :]
    )
    gaussian = np.exp(-np.pi * dimension * displacement**2)
    plane_wave = np.exp(2j * np.pi * dimension * momenta[:, None, None] * displacement)
    boundary_twist = np.exp(-2j * np.pi * boundary_phases.momentum * image_indices)
    states = np.sum(gaussian * plane_wave * boundary_twist[None, None, :], axis=-1)
    return normalize_state(states)


def _readonly_axis(value: ArrayLike, *, name: str) -> FloatArray:
    axis = as_float_array(value, name=name, ndim=1, copy=True)
    if axis.size == 0 or np.any(np.diff(axis) <= 0.0):
        raise ValidationError(f"{name} must be non-empty and strictly increasing")
    axis.setflags(write=False)
    return axis


def _readonly_integral(value: ArrayLike, *, shape: tuple[int, ...]) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValidationError("raw_integral must contain real numeric values")
    integral = np.asarray(raw, dtype=np.float64)
    if integral.shape != shape:
        raise ValidationError(f"raw_integral must have shape {shape}; got {integral.shape}")
    if not bool(np.all(np.isfinite(integral))) or bool(np.any(integral < 0.0)):
        raise ValidationError("raw_integral must be finite and non-negative")
    integral = integral.copy()
    integral.setflags(write=False)
    return integral


def _grid_shape(value: object) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValidationError("grid_shape must be a (n_q, n_p) tuple")
    return (
        _positive_int(value[0], name="grid_shape[0]"),
        _positive_int(value[1], name="grid_shape[1]"),
    )


def _phase_point(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValidationError(f"{name} must be a finite real number; got {value!r}")
    return result % 1.0


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be a positive integer; got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValidationError(f"{name} must be positive; got {result}")
    return result


def _scalar_or_array(
    value: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> float | FloatArray:
    return float(value) if value.ndim == 0 else np.asarray(value, dtype=np.float64)


__all__ = [
    "HusimiResult",
    "WignerResult",
    "coherent_state",
    "husimi_distribution",
    "inverse_participation_ratio",
    "participation_ratio",
    "shannon_entropy",
    "wigner_distribution",
]

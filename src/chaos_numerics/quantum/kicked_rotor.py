"""Finite-torus kicked rotor with dense and FFT operator actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator  # type: ignore[import-untyped]

from chaos_numerics.core import ValidationError
from chaos_numerics.core.types import ArrayLike, ComplexArray
from chaos_numerics.quantum.states import BoundaryPhases, quantum_state


@dataclass(frozen=True, slots=True, eq=False, init=False)
class KickedRotor:
    """Finite-dimensional torus quantization of the standard kicked rotor.

    States use the position ordering ``q_j = (j + alpha) / N``. One Floquet
    step applies the kick and then the free rotation,
    ``U = F.H @ D_free @ F @ D_kick``, with ``hbar_eff = 2 pi / N``.
    ``F`` is the unitary twisted DFT determined by ``(alpha, beta)``.

    A scalar ``boundary_phase`` is shorthand for ``alpha = beta``. Pass a
    :class:`BoundaryPhases` value to control the two twists independently.
    """

    dimension: int
    kick_strength: float
    boundary_phases: BoundaryPhases
    effective_hbar: float
    _kick_phase: ComplexArray = field(repr=False)
    _free_phase: ComplexArray = field(repr=False)
    _fourier_input_phase: ComplexArray = field(repr=False)
    _fourier_output_phase: ComplexArray = field(repr=False)

    def __init__(
        self,
        dimension: int,
        kick_strength: float,
        boundary_phase: float | BoundaryPhases = 0.0,
    ) -> None:
        size = _positive_int(dimension, name="dimension")
        strength = _finite_float(kick_strength, name="kick_strength")
        phases = (
            boundary_phase
            if isinstance(boundary_phase, BoundaryPhases)
            else BoundaryPhases(position=boundary_phase, momentum=boundary_phase)
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
        arrays = (kick, free, fourier_input, fourier_output)
        for array in arrays:
            array.setflags(write=False)

        object.__setattr__(self, "dimension", size)
        object.__setattr__(self, "kick_strength", strength)
        object.__setattr__(self, "boundary_phases", phases)
        object.__setattr__(self, "effective_hbar", hbar)
        object.__setattr__(self, "_kick_phase", kick)
        object.__setattr__(self, "_free_phase", free)
        object.__setattr__(self, "_fourier_input_phase", fourier_input)
        object.__setattr__(self, "_fourier_output_phase", fourier_output)

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

    def apply(self, state: ArrayLike, /) -> ComplexArray:
        """Apply one Floquet step with the matrix-free FFT algorithm."""
        return self.apply_fft(state)

    def apply_fft(self, state: ArrayLike, /) -> ComplexArray:
        """Apply one step in ``O(N log N)`` time and ``O(N)`` auxiliary memory."""
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        momentum = self._position_to_momentum(values * self._kick_phase)
        return self._momentum_to_position(momentum * self._free_phase)

    def apply_dense(self, state: ArrayLike, /) -> ComplexArray:
        """Apply the independently assembled dense reference Floquet matrix."""
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        return np.asarray(values @ self.to_dense().T, dtype=np.complex128)

    def apply_adjoint(self, state: ArrayLike, /) -> ComplexArray:
        """Apply the exact adjoint using the reverse split-operator sequence."""
        values = quantum_state(state, dimension=self.dimension, normalized=False)
        momentum = self._position_to_momentum(values)
        rotated = self._momentum_to_position(momentum * self._free_phase.conj())
        return np.asarray(rotated * self._kick_phase.conj(), dtype=np.complex128)

    def to_dense(self) -> ComplexArray:
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
            state * self._fourier_output_phase.conj(),
            axis=-1,
            norm="ortho",
        )
        return np.asarray(transformed * self._fourier_input_phase.conj(), dtype=np.complex128)


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


__all__ = ["KickedRotor"]

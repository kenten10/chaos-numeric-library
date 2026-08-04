"""Periodized torus coherent states, Husimi data, and localization measures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chaos_numerics.core import ExperimentMetadata, ValidationError
from chaos_numerics.core._validation import as_float_array
from chaos_numerics.core.types import ArrayLike, FloatArray
from chaos_numerics.quantum.states import BoundaryPhases, normalize_state, quantum_state


@dataclass(frozen=True, slots=True, eq=False)
class HusimiResult:
    """Plot-ready normalized density on a half-open ``(q, p)`` midpoint grid."""

    positions: FloatArray
    momenta: FloatArray
    values: FloatArray
    raw_integral: FloatArray
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

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
        result = np.sum(self.values, axis=(-2, -1)) * self.cell_area
        return float(result) if result.ndim == 0 else np.asarray(result, dtype=np.float64)

    @property
    def grid_points(self) -> FloatArray:
        """Return ``(n_q, n_p, 2)`` points in classical ``(q, p)`` order."""
        q_grid, p_grid = np.meshgrid(self.positions, self.momenta, indexing="ij")
        return np.stack((q_grid, p_grid), axis=-1)


def coherent_state(
    *,
    dimension: int,
    position: float,
    momentum: float,
    boundary_phases: BoundaryPhases | None = None,
    images: int = 4,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
    """Return one normalized periodized Gaussian in the position basis."""
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
    """Return ``N |<q,p|psi>|^2`` on a configurable midpoint torus grid."""
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
    for start in range(0, centers_q.size, block):
        stop = min(start + block, centers_q.size)
        coherent = _coherent_batch(
            size,
            centers_q[start:stop],
            centers_p[start:stop],
            boundary_phases=phases,
            images=image_count,
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


def _coherent_batch(
    dimension: int,
    positions: FloatArray,
    momenta: FloatArray,
    *,
    boundary_phases: BoundaryPhases,
    images: int,
) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
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
    "coherent_state",
    "husimi_distribution",
    "inverse_participation_ratio",
    "participation_ratio",
    "shannon_entropy",
]

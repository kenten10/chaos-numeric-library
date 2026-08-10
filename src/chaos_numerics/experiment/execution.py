"""Experiment execution, deterministic sweeps, and minimal NPZ/JSON storage.

The built-in registry covers two model families and the analyses that apply to
each. ``_CLASSICAL_MODELS`` take the classical analyses, ``_QUANTUM_MODELS`` the
quantum ones, and crossing the two is refused with a message that says so rather
than with a bare "unknown analysis".

Quantum spectral statistics are a *chain*, not a single call: the model is
diagonalized, optionally split into a parity sector, prepared as circular
eigenphases, unfolded, and only then reduced to a curve. ``_execute_quantum``
runs that chain and takes every stage's knobs from ``Experiment.parameters``, so
that the sweep -- not a hand-written loop in a notebook -- is what averages a
Bloch-phase ensemble of spectral form factors.

Two conventions are worth stating up front because they are what makes the
quantum registry usable from a JSON parameter grid:

``boundary_phases``
    :class:`~chaos_numerics.quantum.BoundaryPhases` is not JSON, so the
    parameter accepts a scalar, a ``[position, momentum]`` pair, or a
    ``{"position": ..., "momentum": ...}`` mapping. See
    :func:`_as_boundary_phases`.

``symmetry_sector``
    Omitting it makes :func:`~chaos_numerics.spectral.prepare_eigenphases` warn
    with :class:`~chaos_numerics.core.NumericalWarning`, and the sweep lets that
    warning through untouched: comparing an unresolved spectrum with RMT is a
    real mistake, and swallowing the diagnostic inside the framework would hide
    it exactly where it is hardest to notice. The diagnostic is also carried in
    the persisted ``metadata.warnings`` of every affected result. Passing
    ``parity_sector`` supplies the label automatically.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import subprocess
import tempfile
import time
import warnings
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import scipy  # type: ignore[import-untyped]

from chaos_numerics._version import __version__
from chaos_numerics.classical import (
    BakerMap,
    CatMap,
    StandardMap,
    iterate,
    largest_lyapunov_exponent,
    lyapunov_spectrum,
)
from chaos_numerics.core import (
    AnalysisResult,
    ClassicalMap,
    ConvergenceInfo,
    Diagnostic,
    EigenstateResult,
    ExperimentMetadata,
    NumericalWarning,
    QuantumMap,
    ReproducibilityWarning,
    Spectrum,
    Trajectory,
    ValidationError,
)
from chaos_numerics.core._payload import SCHEMA_VERSION
from chaos_numerics.core.types import ArrayLike, ComplexArray, FloatArray
from chaos_numerics.experiment.config import Experiment
from chaos_numerics.operators import UniformPartition, build_ulam, stationary_density
from chaos_numerics.quantum import (
    DEFAULT_DENSE_LIMIT,
    BoundaryPhases,
    CylinderKickedRotor,
    KickedRotor,
    QuantumBakerMap,
    QuantumCatMap,
    SymmetrySector,
    desymmetrize,
    eigenstates,
)
from chaos_numerics.spectral import (
    SpectralCurve,
    UnfoldedSpectrum,
    Window,
    number_variance,
    prepare_eigenphases,
    spectral_form_factor,
    unfold,
)

Result: TypeAlias = Trajectory | Spectrum | EigenstateResult | AnalysisResult | SpectralCurve
RunStatus: TypeAlias = Literal["success", "failed"]
_UnfoldMethod: TypeAlias = Literal["mean", "polynomial"]
# The run metadata embeds the result descriptor produced by
# ``core._payload.metadata_payload`` and validates both against one number, so the
# on-disk layout has exactly one version and it is defined there.
_SEED_DERIVATION = "sha256-v1"
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_ARRAY_COUNT = 128
_MAX_ARRAY_DIMENSIONS = 32
_MAX_NPY_HEADER_BYTES = 16 * 1024
_MAX_JSON_NESTING = 64
_ALLOWED_PERSISTED_DTYPES = {np.dtype("float64"), np.dtype("complex128")}
_CLASSICAL_MODELS: tuple[str, ...] = ("baker_map", "cat_map", "standard_map")
_QUANTUM_MODELS: tuple[str, ...] = (
    "cylinder_kicked_rotor",
    "kicked_rotor",
    "quantum_baker_map",
    "quantum_cat_map",
)
_EXPERIMENT_MODELS: tuple[str, ...] = tuple(sorted(_CLASSICAL_MODELS + _QUANTUM_MODELS))
_CLASSICAL_ANALYSES: tuple[str, ...] = (
    "iterate",
    "largest_lyapunov_exponent",
    "lyapunov_spectrum",
    "trajectory",
    "ulam_stationary_density",
)
_QUANTUM_ANALYSES: tuple[str, ...] = (
    "eigenphases",
    "eigenstates",
    "number_variance",
    "spectral_form_factor",
)
_EXPERIMENT_ANALYSES: tuple[str, ...] = tuple(sorted(_CLASSICAL_ANALYSES + _QUANTUM_ANALYSES))


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """One completed or failed experiment job with provenance."""

    job_id: str
    experiment: Experiment
    parameters: Mapping[str, object]
    seed: int | None
    status: RunStatus
    result: Result | None
    duration_seconds: float
    environment: Mapping[str, object]
    git_commit: str | None
    output_path: Path | None = None
    error: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Summary of a persisted sweep."""

    runs: tuple[ExperimentRun, ...]
    output_path: Path

    @property
    def succeeded(self) -> int:
        return sum(run.status == "success" for run in self.runs)

    @property
    def failed(self) -> int:
        return sum(run.status == "failed" for run in self.runs)


def run_experiment(
    experiment: Experiment,
    *,
    parameters: Mapping[str, object] | None = None,
    output: str | os.PathLike[str] | None = None,
) -> ExperimentRun:
    """Execute one built-in experiment and optionally persist it."""
    _validate_experiment(experiment)
    merged = _merged_parameters(experiment, parameters)
    _require_seed_for_persistence(experiment, output is not None)
    job_id = _job_id(merged)
    started = time.perf_counter()
    result = _execute(experiment.model, experiment.analysis, merged, experiment.seed)
    run = ExperimentRun(
        job_id=job_id,
        experiment=experiment,
        parameters=merged,
        seed=experiment.seed,
        status="success",
        result=result,
        duration_seconds=time.perf_counter() - started,
        environment=_environment(),
        git_commit=_git_commit(),
        output_path=Path(output) if output is not None else None,
    )
    if output is not None:
        _save_run(run, Path(output))
    return run


def run_sweep(
    experiment: Experiment,
    *,
    parameters: Iterable[Mapping[str, object]],
    output: str | os.PathLike[str],
    resume: bool = True,
    continue_on_error: bool = True,
) -> SweepResult:
    """Execute a deterministic, resumable parameter sweep."""
    _validate_experiment(experiment)
    if not isinstance(resume, bool) or not isinstance(continue_on_error, bool):
        raise ValidationError("resume and continue_on_error must be bool values")
    _require_seed_for_persistence(experiment, True)
    output_path = Path(output)
    if output_path.is_symlink():
        raise ValidationError(f"refusing to persist through symbolic-link directory: {output_path}")
    rows = tuple(_merged_parameters(experiment, row) for row in parameters)
    job_ids = tuple(_job_id(row) for row in rows)
    if len(set(job_ids)) != len(job_ids):
        raise ValidationError("sweep contains duplicate parameter combinations")
    fingerprint = _fingerprint(experiment, job_ids)
    manifest_path = output_path / "manifest.json"
    manifest = _prepare_manifest(manifest_path, experiment, job_ids, fingerprint, resume=resume)
    jobs = cast(dict[str, object], manifest["jobs"])
    runs: list[ExperimentRun] = []
    output_path.mkdir(parents=True, exist_ok=True)

    for job_id, row in zip(job_ids, rows, strict=True):
        job_path = output_path / "jobs" / job_id
        current = cast(dict[str, object], jobs[job_id])
        if (
            resume
            and current.get("status") == "success"
            and (job_path / "metadata.json").is_file()
            and (job_path / "arrays.npz").is_file()
        ):
            runs.append(load_result(job_path))
            continue
        child_seed = _child_seed(experiment.seed, job_id)
        current.update({"status": "running", "seed": child_seed, "error": None})
        _atomic_json(manifest_path, manifest)
        started = time.perf_counter()
        try:
            result = _execute(experiment.model, experiment.analysis, row, child_seed)
            run = ExperimentRun(
                job_id=job_id,
                experiment=experiment,
                parameters=row,
                seed=child_seed,
                status="success",
                result=result,
                duration_seconds=time.perf_counter() - started,
                environment=_environment(),
                git_commit=_git_commit(),
                output_path=job_path,
            )
            _save_run(run, job_path)
            current.update({"status": "success", "duration_seconds": run.duration_seconds})
        except Exception as error:
            error_payload = {"type": type(error).__name__, "message": str(error)}
            run = ExperimentRun(
                job_id=job_id,
                experiment=experiment,
                parameters=row,
                seed=child_seed,
                status="failed",
                result=None,
                duration_seconds=time.perf_counter() - started,
                environment=_environment(),
                git_commit=_git_commit(),
                output_path=job_path,
                error=error_payload,
            )
            _save_run(run, job_path)
            current.update(
                {
                    "status": "failed",
                    "duration_seconds": run.duration_seconds,
                    "error": error_payload,
                }
            )
            if not continue_on_error:
                _atomic_json(manifest_path, manifest)
                raise
        runs.append(run)
        _atomic_json(manifest_path, manifest)
    return SweepResult(tuple(runs), output_path)


def load_result(path: str | os.PathLike[str]) -> ExperimentRun:
    """Reload one persisted run, validating its array descriptors."""
    run_path = Path(path)
    payload = _read_json(run_path / "metadata.json")
    _validate_schema(payload, name="run")
    experiment_data = _mapping(payload.get("experiment"), name="experiment")
    experiment = Experiment(
        model=str(experiment_data["model"]),
        analysis=str(experiment_data["analysis"]),
        parameters=_mapping(experiment_data.get("parameters", {}), name="parameters"),
        seed=_optional_int(experiment_data.get("seed"), name="seed"),
        strict_reproducibility=bool(experiment_data.get("strict_reproducibility", False)),
    )
    status = payload.get("status")
    if status not in {"success", "failed"}:
        raise ValidationError(f"unknown persisted run status {status!r}")
    result: Result | None = None
    if status == "success":
        result_payload = _mapping(payload.get("result"), name="result")
        _validate_schema(result_payload, name="result")
        archive_path = run_path / "arrays.npz"
        _validate_archive_digest(archive_path, payload)
        _preflight_archive(archive_path, result_payload)
        try:
            with np.load(archive_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            raise ValidationError(
                f"could not read persisted arrays {archive_path}: {error}"
            ) from error
        _validate_descriptors(result_payload, arrays)
        result = _restore_result(result_payload, arrays)
    error_data = payload.get("error")
    return ExperimentRun(
        job_id=str(payload["job_id"]),
        experiment=experiment,
        parameters=_mapping(payload.get("parameters"), name="parameters"),
        seed=_optional_int(payload.get("seed"), name="seed"),
        status=status,
        result=result,
        duration_seconds=_as_float(payload["duration_seconds"], name="duration_seconds"),
        environment=_mapping(payload.get("environment"), name="environment"),
        git_commit=cast(str | None, payload.get("git_commit")),
        output_path=run_path,
        error=cast(Mapping[str, str] | None, error_data),
    )


def _execute(
    model_name: str, analysis: str, parameters: Mapping[str, object], seed: int | None
) -> Result:
    if model_name in _CLASSICAL_MODELS:
        return _execute_classical(model_name, analysis, dict(parameters), seed)
    if model_name in _QUANTUM_MODELS:
        return _execute_quantum(model_name, analysis, dict(parameters), seed)
    raise ValidationError(
        f"unknown experiment model {model_name!r}; expected one of {_names(_EXPERIMENT_MODELS)}"
    )


def _execute_classical(
    model_name: str, analysis: str, values: dict[str, object], seed: int | None
) -> Result:
    model: ClassicalMap
    if model_name == "standard_map":
        model = StandardMap(
            kick_strength=_as_float(values.pop("kick_strength", 1.0), name="kick_strength")
        )
    elif model_name == "cat_map":
        model = CatMap(matrix=_as_matrix(values.pop("matrix", ((2, 1), (1, 1)))))
    else:
        model = BakerMap(cut=_as_float(values.pop("cut", 0.5), name="cut"))

    if analysis in {"iterate", "trajectory"}:
        initial = _as_array_like(values.pop("initial_state", (0.1, 0.2)), name="initial_state")
        steps = _as_int(values.pop("steps"), name="steps")
        include_initial = _as_bool(values.pop("include_initial", True), name="include_initial")
        _reject_unused(values)
        return iterate(model, initial, steps=steps, include_initial=include_initial)
    if analysis in {"largest_lyapunov_exponent", "lyapunov_spectrum"}:
        initial = _as_array_like(values.pop("initial_state", (0.1, 0.2)), name="initial_state")
        function = (
            largest_lyapunov_exponent
            if analysis == "largest_lyapunov_exponent"
            else lyapunov_spectrum
        )
        steps = _as_int(values.pop("steps"), name="steps")
        transient = _as_int(values.pop("transient", 0), name="transient")
        interval = _as_int(
            values.pop("reorthogonalization_interval", 1),
            name="reorthogonalization_interval",
        )
        history_interval = _as_int(values.pop("history_interval", 1), name="history_interval")
        rtol = _as_float(values.pop("convergence_rtol", 1e-3), name="convergence_rtol")
        atol = _as_float(values.pop("convergence_atol", 1e-8), name="convergence_atol")
        strict = _as_bool(values.pop("strict", False), name="strict")
        _reject_unused(values)
        return function(
            model,
            initial,
            steps=steps,
            transient=transient,
            reorthogonalization_interval=interval,
            history_interval=history_interval,
            convergence_rtol=rtol,
            convergence_atol=atol,
            seed=seed,
            strict=strict,
        )
    if analysis == "ulam_stationary_density":
        partition_shape = _as_int_sequence(
            values.pop("partition_shape", (8, 8)), name="partition_shape"
        )
        samples = _as_int(values.pop("samples_per_cell", 128), name="samples_per_cell")
        tolerance = _as_float(values.pop("tolerance", 1e-10), name="tolerance")
        strict = _as_bool(values.pop("strict", True), name="strict")
        _reject_unused(values)
        partition = UniformPartition(model.bounds, partition_shape, periodic=model.is_periodic)
        operator = build_ulam(model, partition, samples_per_cell=samples, seed=seed)
        return stationary_density(operator, tolerance=tolerance, strict=strict)
    raise _unsupported_analysis(model_name, analysis, family="classical")


def _execute_quantum(
    model_name: str, analysis: str, values: dict[str, object], seed: int | None
) -> Result:
    """Diagonalize a quantum map and reduce it to the requested result.

    Every analysis here starts from :func:`~chaos_numerics.quantum.eigenstates`,
    because the residual diagnostics that the persisted metadata carries are a
    by-product of the eigenvectors; there is no eigenvalue-only fast path in the
    library and there is none here either.

    The two ``SpectralCurve`` analyses then run the rest of the chain --
    optional :func:`~chaos_numerics.quantum.desymmetrize`, then
    :func:`~chaos_numerics.spectral.prepare_eigenphases`, then
    :func:`~chaos_numerics.spectral.unfold` -- so that a caller who wants an
    ensemble-averaged form factor only has to vary ``boundary_phases`` across a
    grid. Parameters, all optional unless marked:

    ``dimension`` (required)
        Hilbert-space dimension. ``quantum_cat_map`` and ``quantum_baker_map``
        require it to be even.
    ``kick_strength``, ``effective_hbar``, ``matrix``, ``boundary_phases``
        Model constructor arguments, spelled exactly as the classes spell them.
        ``boundary_phases`` uses the JSON encodings of
        :func:`_as_boundary_phases`.
    ``dense_limit``
        Guard on ``O(N**2)`` densification, defaulting to
        :data:`~chaos_numerics.quantum.DEFAULT_DENSE_LIMIT`. It is *not* raised
        to ``dimension`` automatically: a sweep that silently densifies a
        4096-dimensional model is exactly what the guard exists to prevent.
    ``parity_sector``
        ``"even"`` or ``"odd"`` to keep one block of the model's parity
        symmetry. Required for any comparison with RMT on a model that still has
        parity -- the baker map and the cat map always do, the kicked rotor does
        for ``alpha, beta in {0, 1/2}``.
    ``symmetry_sector``
        The free-text label recorded on the prepared spectrum. Defaults to
        ``"parity-<sector>"`` when ``parity_sector`` is given and to ``None``
        otherwise, which warns; see this module's docstring.
    ``unfold_method``, ``unfold_degree``
        Passed to :func:`~chaos_numerics.spectral.unfold`.
    ``times`` (required for ``spectral_form_factor``), ``window``,
    ``connected``, ``bootstrap``
        Passed to :func:`~chaos_numerics.spectral.spectral_form_factor`.
    ``lengths`` (required for ``number_variance``), ``samples``, ``bootstrap``,
    ``finite_size_correction``
        Passed to :func:`~chaos_numerics.spectral.number_variance`.
    """
    if analysis not in _QUANTUM_ANALYSES:
        raise _unsupported_analysis(model_name, analysis, family="quantum")
    model = _quantum_model(model_name, values)
    dense_limit = _as_int(values.pop("dense_limit", DEFAULT_DENSE_LIMIT), name="dense_limit")
    sector = _as_parity_sector(values.pop("parity_sector", None))
    label = _symmetry_label(values.pop("symmetry_sector", None), sector)

    if analysis in {"eigenphases", "eigenstates"}:
        _reject_unused(values)
        eigensystem = _resolved_eigensystem(
            model, model_name, dense_limit=dense_limit, sector=sector
        )
        if analysis == "eigenstates":
            return eigensystem
        # Identical to quantum.eigenphases(model), which is itself a thin wrapper
        # over eigenstates; spelling it out here is what lets parity_sector apply
        # to the phases as well as to the full eigensystem.
        return AnalysisResult(
            "eigenphases",
            eigensystem.eigenphases,
            residuals=eigensystem.residuals,
            metadata=eigensystem.metadata,
        )

    method = _as_unfold_method(values.pop("unfold_method", "mean"))
    degree = _as_int(values.pop("unfold_degree", 5), name="unfold_degree")
    bootstrap = _as_int(values.pop("bootstrap", 0), name="bootstrap")
    if model_name == "cylinder_kicked_rotor":
        # CylinderKickedRotor says this in its own docstring; the registry is where
        # somebody actually writes the pairing down, so it says it here too.
        warnings.warn(
            "cylinder_kicked_rotor is a truncation of an infinite momentum lattice, "
            "not a torus quantization, so its eigenphase statistics belong to the "
            "truncation as much as to the rotor and must not be read as a test of "
            "random-matrix universality. Use kicked_rotor for level statistics",
            NumericalWarning,
            stacklevel=4,
        )

    def unfolded() -> UnfoldedSpectrum:
        """Run the diagonalize/desymmetrize/prepare/unfold chain on demand.

        Deferred so that every curve parameter is validated first. A misspelled
        ``window`` has to be a fast refusal, not one that arrives after an
        ``O(N**3)`` diagonalization and after ``prepare_eigenphases`` has already
        warned about a missing symmetry sector.
        """
        eigensystem = _resolved_eigensystem(
            model, model_name, dense_limit=dense_limit, sector=sector
        )
        return unfold(
            prepare_eigenphases(eigensystem, symmetry_sector=label),
            method=method,
            degree=degree,
        )

    if analysis == "spectral_form_factor":
        times = _as_array_like(values.pop("times", None), name="times")
        window = _as_window(values.pop("window", "none"))
        connected = _as_bool(values.pop("connected", True), name="connected")
        _reject_unused(values)
        return spectral_form_factor(
            unfolded(),
            times,
            window=window,
            connected=connected,
            bootstrap=bootstrap,
            seed=seed,
        )
    lengths = _as_array_like(values.pop("lengths", None), name="lengths")
    samples = _as_int(values.pop("samples", 2048), name="samples")
    correction = _as_bool(
        values.pop("finite_size_correction", False), name="finite_size_correction"
    )
    _reject_unused(values)
    return number_variance(
        unfolded(),
        lengths,
        samples=samples,
        bootstrap=bootstrap,
        seed=seed,
        finite_size_correction=correction,
    )


def _resolved_eigensystem(
    model: QuantumMap,
    model_name: str,
    *,
    dense_limit: int,
    sector: SymmetrySector | None,
) -> EigenstateResult:
    eigensystem = eigenstates(model, dense_limit=dense_limit)
    if sector is None:
        return eigensystem
    return desymmetrize(eigensystem, _parity_operator(model, model_name), sector=sector)


def _quantum_model(model_name: str, values: dict[str, object]) -> QuantumMap:
    dimension = _as_int(values.pop("dimension", None), name="dimension")
    if model_name == "kicked_rotor":
        return KickedRotor(
            dimension,
            _as_float(values.pop("kick_strength", 1.0), name="kick_strength"),
            _as_boundary_phases(values.pop("boundary_phases", 0.0)),
        )
    if model_name == "cylinder_kicked_rotor":
        return CylinderKickedRotor(
            dimension,
            _as_float(values.pop("kick_strength", 1.0), name="kick_strength"),
            _as_float(values.pop("effective_hbar", 1.0), name="effective_hbar"),
        )
    if model_name == "quantum_cat_map":
        return QuantumCatMap(
            dimension,
            _as_matrix(values.pop("matrix", ((2, 1), (1, 1)))),
            _as_boundary_phases(values.pop("boundary_phases", 0.0)),
        )
    return QuantumBakerMap(dimension, _as_boundary_phases(values.pop("boundary_phases", 0.5)))


def _parity_operator(model: QuantumMap, model_name: str) -> ComplexArray:
    operators = getattr(model, "symmetry_operators", {})
    operator = operators.get("parity") if isinstance(operators, Mapping) else None
    if operator is None:
        raise ValidationError(
            f"parity_sector was requested but model {model_name!r} exposes no parity symmetry "
            "at these parameters, so there is no block to keep. The kicked rotor only has an "
            "exact reflection for boundary_phases with 2*alpha and 2*beta integer; drop "
            "parity_sector for generic Bloch phases, which break parity on purpose"
        )
    return cast(ComplexArray, operator)


def _unsupported_analysis(model_name: str, analysis: str, *, family: str) -> ValidationError:
    """Say that a model and an analysis do not go together, not just that one is unknown.

    ``kicked_rotor`` with ``lyapunov_spectrum`` used to fail deep inside the
    classical Lyapunov routine on a missing ``jacobian``, which reads as a bug in
    the model rather than as a mismatched pair.
    """
    supported = _QUANTUM_ANALYSES if family == "quantum" else _CLASSICAL_ANALYSES
    other = _CLASSICAL_ANALYSES if family == "quantum" else _QUANTUM_ANALYSES
    if analysis in other:
        other_family = "classical" if family == "quantum" else "quantum"
        return ValidationError(
            f"analysis {analysis!r} cannot be combined with model {model_name!r}: "
            f"{analysis!r} applies to {other_family} models and {model_name!r} is a "
            f"{family} model. {model_name!r} supports {_names(supported)}"
        )
    return ValidationError(
        f"unknown experiment analysis {analysis!r}; expected one of {_names(_EXPERIMENT_ANALYSES)}"
    )


def _save_run(run: ExperimentRun, path: Path) -> None:
    if path.is_symlink():
        raise ValidationError(f"refusing to persist through symbolic-link directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    result_payload = run.result.metadata_payload() if run.result is not None else None
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "job_id": run.job_id,
        "status": run.status,
        "experiment": run.experiment.to_dict(),
        "parameters": dict(run.parameters),
        "seed": run.seed,
        "seed_derivation": _SEED_DERIVATION if run.seed is not None else None,
        "duration_seconds": run.duration_seconds,
        "environment": dict(run.environment),
        "git_commit": run.git_commit,
        "result": result_payload,
        "error": dict(run.error) if run.error is not None else None,
    }
    if run.result is not None:
        arrays = run.result.array_payload()
        _add_array_digests(result_payload, arrays)
        _atomic_npz(path / "arrays.npz", arrays)
        payload["arrays_sha256"] = _file_digest(path / "arrays.npz")
    _atomic_json(path / "metadata.json", payload)


def _restore_result(
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> Result:
    metadata = _restore_metadata(_mapping(payload.get("metadata"), name="result metadata"), arrays)
    result_type = payload.get("result_type")
    if result_type == "trajectory":
        return Trajectory(
            cast(FloatArray, arrays["states"]),
            cast(FloatArray, arrays["initial_state"]),
            metadata,
        )
    if result_type == "spectrum":
        return Spectrum(
            cast(ComplexArray, arrays["eigenvalues"]),
            cast(ComplexArray | None, arrays.get("eigenvectors")),
            cast(FloatArray | None, arrays.get("residuals")),
            metadata,
        )
    if result_type == "eigenstate":
        return EigenstateResult(
            cast(FloatArray, arrays["eigenphases"]),
            cast(ComplexArray, arrays["eigenstates"]),
            cast(FloatArray, arrays["residuals"]),
            metadata,
        )
    if result_type == "analysis":
        return AnalysisResult(
            str(payload["name"]),
            cast(FloatArray, arrays["values"]),
            cast(FloatArray | None, arrays.get("uncertainty")),
            cast(FloatArray | None, arrays.get("residuals")),
            metadata,
        )
    if result_type == "spectral_curve":
        # ``uncertainty`` and ``variance`` are absent from the archive rather than
        # stored as empty arrays when the producer reported none, so ``get``
        # rebuilds them as ``None`` and the round trip preserves the distinction
        # between "no error bar" and "an error bar that happens to be zero".
        return SpectralCurve(
            cast(FloatArray, arrays["x"]),
            cast(FloatArray, arrays["values"]),
            cast(FloatArray | None, arrays.get("uncertainty")),
            cast(FloatArray | None, arrays.get("variance")),
            metadata,
        )
    raise ValidationError(f"unknown persisted result type {result_type!r}")


def _restore_metadata(
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> ExperimentMetadata:
    warning_rows = cast(list[object], payload.get("warnings", []))
    diagnostics = tuple(
        Diagnostic(str(row["code"]), str(row["message"]), str(row["category"]))
        for item in warning_rows
        for row in [_mapping(item, name="warning")]
    )
    convergence_data = payload.get("convergence")
    convergence = None
    if convergence_data is not None:
        row = _mapping(convergence_data, name="convergence")
        convergence = ConvergenceInfo(
            converged=bool(row["converged"]),
            iterations=_as_int(row["iterations"], name="convergence iterations"),
            residual=_optional_float(row.get("residual")),
            tolerance=_optional_float(row.get("tolerance")),
            history=cast(FloatArray | None, arrays.get("convergence_history")),
            reason=cast(str | None, row.get("reason")),
        )
    return ExperimentMetadata(
        parameters=_mapping(payload.get("parameters", {}), name="result parameters"),
        precision=str(payload.get("precision", "float64")),
        seed=_optional_int(payload.get("seed"), name="result seed"),
        environment=_mapping(payload.get("environment", {}), name="result environment"),
        git_commit=cast(str | None, payload.get("git_commit")),
        warnings=diagnostics,
        convergence=convergence,
    )


def _validate_descriptors(
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> None:
    descriptors = _mapping(payload.get("arrays"), name="array descriptors")
    if set(descriptors) != set(arrays):
        raise ValidationError("NPZ arrays do not match metadata descriptors")
    for name, raw in descriptors.items():
        descriptor = _mapping(raw, name=f"array descriptor {name}")
        if list(arrays[name].shape) != descriptor.get("shape") or arrays[
            name
        ].dtype.name != descriptor.get("dtype"):
            raise ValidationError(f"array {name!r} does not match its persisted descriptor")
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or not hmac.compare_digest(
            _array_digest(arrays[name]), digest
        ):
            raise ValidationError(f"array {name!r} failed its SHA-256 integrity check")


def _prepare_manifest(
    path: Path, experiment: Experiment, job_ids: tuple[str, ...], fingerprint: str, *, resume: bool
) -> dict[str, object]:
    if path.exists():
        if not resume:
            raise ValidationError(f"sweep output already exists: {path.parent}")
        manifest = _read_json(path)
        _validate_schema(manifest, name="sweep manifest")
        if manifest.get("fingerprint") != fingerprint:
            raise ValidationError(
                f"existing sweep manifest at {path} describes a different experiment or "
                f"parameter grid, so resuming would mix two studies in one directory. "
                f"Point output= at a directory of its own, pass resume=False to refuse "
                f"the existing results explicitly, or remove {path.parent} to start over"
            )
        return manifest
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "experiment": experiment.to_dict(),
        "seed_derivation": _SEED_DERIVATION,
        "jobs": {job_id: {"status": "pending", "seed": None, "error": None} for job_id in job_ids},
    }


def _merged_parameters(
    experiment: Experiment, parameters: Mapping[str, object] | None
) -> dict[str, object]:
    merged = dict(experiment.parameters)
    if parameters is not None:
        if not isinstance(parameters, Mapping):
            raise ValidationError("parameters must be a mapping")
        merged.update(parameters)
    validated = ExperimentMetadata(parameters=merged).to_dict()["parameters"]
    return _mapping(validated, name="validated parameters")


def _require_seed_for_persistence(experiment: Experiment, persisted: bool) -> None:
    if not persisted or experiment.seed is not None:
        return
    message = "persisted experiments should declare a seed for reproducibility"
    if experiment.strict_reproducibility:
        raise ValidationError(message)
    warnings.warn(message, ReproducibilityWarning, stacklevel=3)


def _job_id(parameters: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(parameters).encode()).hexdigest()[:16]


def _fingerprint(experiment: Experiment, job_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        _canonical({"experiment": experiment.to_dict(), "jobs": sorted(job_ids)}).encode()
    ).hexdigest()


def _child_seed(root_seed: int | None, job_id: str) -> int | None:
    if root_seed is None:
        return None
    digest = hashlib.sha256(f"{_SEED_DERIVATION}:{root_seed}:{job_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _environment() -> dict[str, object]:
    """Return the JSON-compatible reproducibility record for one run.

    ``docs/design/numerical-standards.md`` section 10.1 requires the Python,
    NumPy, and SciPy versions plus the platform. SciPy governs ARPACK and the
    sparse eigensolvers, so its version changes results and is recorded. The
    BLAS/LAPACK identity and any threading environment variables are recorded
    too, because they change floating-point summation order.
    """
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "chaos_numerics": __version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    blas = _blas_description()
    if blas is not None:
        environment["blas"] = blas
    threads = {
        name: value
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        if (value := os.environ.get(name)) is not None
    }
    if threads:
        environment["thread_limits"] = threads
    dirty = _git_is_dirty()
    if dirty is not None:
        environment["git_dirty"] = dirty
    return environment


def _names(values: tuple[str, ...]) -> str:
    return ", ".join(repr(value) for value in values)


def _blas_description() -> str | None:
    """Return a short BLAS/LAPACK identifier from ``numpy.__config__``."""
    show = getattr(np.__config__, "show", None)
    if not callable(show):
        return None
    try:
        config = show(mode="dicts")
    except (TypeError, ValueError, AttributeError):
        return None
    if not isinstance(config, Mapping):
        return None
    dependencies = config.get("Build Dependencies")
    if not isinstance(dependencies, Mapping):
        return None
    parts: list[str] = []
    for name in ("blas", "lapack"):
        entry = dependencies.get(name)
        if isinstance(entry, Mapping):
            label = str(entry.get("name", name))
            version = str(entry.get("version", "unknown"))
            parts.append(f"{name}={label} {version}")
    return "; ".join(parts) or None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _git_is_dirty() -> bool | None:
    """Return whether the working tree has uncommitted changes.

    Recorded separately from ``git_commit`` so that the commit field stays a
    parsable hash while a dirty tree is never silently persisted as a clean
    commit. ``None`` means the state could not be determined.
    """
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(
    path: Path, arrays: Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        cast(Any, np.savez_compressed)(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        if path.stat().st_size > _MAX_METADATA_BYTES:
            raise ValidationError(f"persisted metadata exceeds {_MAX_METADATA_BYTES} bytes: {path}")
        text = path.read_text(encoding="utf-8")
        _validate_json_nesting(text)
        value = json.loads(text)
    except ValidationError:
        raise
    except (OSError, RecursionError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"could not read persisted metadata {path}: {error}") from error
    return dict(_mapping(value, name="persisted metadata"))


def _validate_json_nesting(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                raise ValidationError(
                    f"persisted metadata exceeds {_MAX_JSON_NESTING} JSON nesting levels"
                )
        elif character in "]}":
            depth = max(0, depth - 1)


def _add_array_digests(
    payload: dict[str, object] | None,
    arrays: Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
) -> None:
    if payload is None:
        return
    descriptors = _mapping(payload.get("arrays"), name="array descriptors")
    for name, array in arrays.items():
        descriptor = _mapping(descriptors[name], name=f"array descriptor {name}")
        descriptor["sha256"] = _array_digest(array)
        descriptors[name] = descriptor
    payload["arrays"] = descriptors


def _array_digest(array: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValidationError(f"could not read persisted arrays {path}: {error}") from error
    return digest.hexdigest()


def _validate_archive_digest(path: Path, payload: Mapping[str, object]) -> None:
    expected = payload.get("arrays_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValidationError("persisted run is missing a valid arrays SHA-256 digest")
    try:
        if path.stat().st_size > _MAX_ARCHIVE_BYTES:
            raise ValidationError(f"array archive exceeds {_MAX_ARCHIVE_BYTES} bytes: {path}")
    except OSError as error:
        raise ValidationError(f"could not inspect persisted arrays {path}: {error}") from error
    if not hmac.compare_digest(_file_digest(path), expected):
        raise ValidationError("array archive failed its SHA-256 integrity check")


def _preflight_archive(path: Path, payload: Mapping[str, object]) -> None:
    descriptors = _mapping(payload.get("arrays"), name="array descriptors")
    if len(descriptors) > _MAX_ARRAY_COUNT:
        raise ValidationError(f"array archive contains more than {_MAX_ARRAY_COUNT} arrays")

    expected: dict[str, tuple[tuple[int, ...], np.dtype[np.generic]]] = {}
    expected_bytes = 0
    for name, raw in descriptors.items():
        descriptor = _mapping(raw, name=f"array descriptor {name}")
        shape = _validated_shape(descriptor.get("shape"), name=name)
        try:
            dtype = np.dtype(str(descriptor.get("dtype")))
        except TypeError as error:
            raise ValidationError(f"array {name!r} has an invalid dtype descriptor") from error
        if dtype.hasobject:
            raise ValidationError(f"array {name!r} cannot use an object dtype")
        if dtype not in _ALLOWED_PERSISTED_DTYPES:
            raise ValidationError(f"array {name!r} uses unsupported dtype {dtype.name!r}")
        count = 1
        for dimension in shape:
            count *= dimension
        array_bytes = count * dtype.itemsize
        if array_bytes > _MAX_UNCOMPRESSED_BYTES - expected_bytes:
            raise ValidationError(
                f"array descriptors exceed {_MAX_UNCOMPRESSED_BYTES} uncompressed bytes"
            )
        expected_bytes += array_bytes
        expected[f"{name}.npy"] = (shape, dtype)

    try:
        if path.stat().st_size > _MAX_ARCHIVE_BYTES:
            raise ValidationError(f"array archive exceeds {_MAX_ARCHIVE_BYTES} bytes: {path}")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            member_names = {member.filename for member in members}
            if len(members) != len(member_names):
                raise ValidationError("array archive contains duplicate member names")
            if member_names != set(expected):
                raise ValidationError("NPZ members do not match metadata descriptors")
            total_size = 0
            for member in members:
                if member.is_dir() or "/" in member.filename or "\\" in member.filename:
                    raise ValidationError("array archive contains an invalid member path")
                if member.flag_bits & 0x1:
                    raise ValidationError("array archive contains an encrypted member")
                if member.file_size > _MAX_UNCOMPRESSED_BYTES - total_size:
                    raise ValidationError(
                        f"array archive exceeds {_MAX_UNCOMPRESSED_BYTES} uncompressed bytes"
                    )
                total_size += member.file_size
                with archive.open(member) as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        header = np.lib.format.read_array_header_1_0(
                            handle, max_header_size=_MAX_NPY_HEADER_BYTES
                        )
                    elif version == (2, 0):
                        header = np.lib.format.read_array_header_2_0(
                            handle, max_header_size=_MAX_NPY_HEADER_BYTES
                        )
                    else:
                        raise ValidationError(
                            f"array archive uses unsupported NPY version {version!r}"
                        )
                shape, _fortran_order, dtype = header
                expected_shape, expected_dtype = expected[member.filename]
                if tuple(shape) != expected_shape or dtype != expected_dtype:
                    raise ValidationError(
                        f"array member {member.filename!r} does not match its descriptor"
                    )
    except ValidationError:
        raise
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValidationError(f"could not inspect persisted arrays {path}: {error}") from error


def _validated_shape(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationError(f"array {name!r} has an invalid shape descriptor")
    if len(value) > _MAX_ARRAY_DIMENSIONS:
        raise ValidationError(f"array {name!r} has more than {_MAX_ARRAY_DIMENSIONS} dimensions")
    shape = tuple(_as_int(item, name=f"array {name} dimension") for item in value)
    if any(dimension < 0 for dimension in shape):
        raise ValidationError(f"array {name!r} has a negative dimension")
    return shape


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValidationError(f"{name} must be an object with string keys")
    return {str(key): item for key, item in value.items()}


def _validate_schema(payload: Mapping[str, object], *, name: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(
            f"unsupported {name} schema version {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )


def _optional_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer or null")
    return value


def _optional_float(value: object) -> float | None:
    return None if value is None else _as_float(value, name="optional float")


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    return value


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a real number")
    return float(value)


def _as_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a bool")
    return value


def _as_array_like(value: object, *, name: str) -> ArrayLike:
    if not isinstance(value, (Sequence, int, float, complex, str, bytes)):
        raise ValidationError(f"{name} must be array-like")
    return cast(ArrayLike, value)


def _as_int_sequence(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationError(f"{name} must be an integer sequence")
    return tuple(_as_int(item, name=name) for item in value)


def _as_boundary_phases(value: object) -> BoundaryPhases:
    """Build :class:`BoundaryPhases` from one of three JSON-compatible encodings.

    ``Experiment.parameters`` has to survive ``json.dump`` and come back equal, so
    a ``BoundaryPhases`` instance cannot be stored there. Three spellings are
    accepted, and all three round-trip through JSON unchanged:

    ``0.25``
        A real number is the scalar shorthand ``alpha = beta = 0.25``, matching
        the model constructors, which accept ``float | BoundaryPhases``.
    ``[0.25, 0.13]``
        A two-element sequence is ``[position, momentum]``. This is the spelling
        to use in ``cartesian_grid``/``zip_grid``, because a list of pairs is the
        natural way to write down a Bloch-phase ensemble.
    ``{"position": 0.25, "momentum": 0.13}``
        A mapping is exactly what ``BoundaryPhases.to_dict()`` returns, and
        therefore exactly what a persisted result carries in
        ``metadata.parameters["boundary_phases"]``. A value read back off disk can
        be fed straight into the next sweep without reshaping.

    Both phases are reduced modulo one by ``BoundaryPhases`` itself, so ``1.25``
    and ``0.25`` name the same twist and would collide as two sweep rows.
    """
    if isinstance(value, Mapping):
        row = _mapping(value, name="boundary_phases")
        unknown = set(row) - {"position", "momentum"}
        if unknown:
            raise ValidationError(
                "boundary_phases mapping accepts only 'position' and 'momentum'; got "
                f"{', '.join(repr(name) for name in sorted(unknown))}"
            )
        return BoundaryPhases(
            position=_as_float(row.get("position", 0.0), name="boundary_phases position"),
            momentum=_as_float(row.get("momentum", 0.0), name="boundary_phases momentum"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValidationError(
                f"boundary_phases sequence must be [position, momentum]; got {len(value)} entries"
            )
        return BoundaryPhases(
            position=_as_float(value[0], name="boundary_phases position"),
            momentum=_as_float(value[1], name="boundary_phases momentum"),
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            "boundary_phases must be a real number, a [position, momentum] pair, or a "
            "{'position': ..., 'momentum': ...} mapping; got "
            f"{type(value).__name__}"
        )
    scalar = _as_float(value, name="boundary_phases")
    return BoundaryPhases(position=scalar, momentum=scalar)


def _as_parity_sector(value: object) -> SymmetrySector | None:
    if value is None:
        return None
    if value not in {"even", "odd"}:
        raise ValidationError(f"parity_sector must be 'even', 'odd', or null; got {value!r}")
    return value


def _symmetry_label(value: object, sector: SymmetrySector | None) -> str | None:
    if value is None:
        return None if sector is None else f"parity-{sector}"
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValidationError("symmetry_sector must be a non-empty trimmed string or null")
    return value


def _as_unfold_method(value: object) -> _UnfoldMethod:
    if value not in {"mean", "polynomial"}:
        raise ValidationError(f"unfold_method must be 'mean' or 'polynomial'; got {value!r}")
    return value


def _as_window(value: object) -> Window:
    if value not in {"none", "hann"}:
        raise ValidationError(f"window must be 'none' or 'hann'; got {value!r}")
    return value


def _as_matrix(value: object) -> tuple[tuple[int, int], tuple[int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationError("matrix must be a nested integer sequence")
    rows = tuple(_as_int_sequence(row, name="matrix") for row in value)
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ValidationError("matrix must have shape (2, 2)")
    return ((rows[0][0], rows[0][1]), (rows[1][0], rows[1][1]))


def _reject_unused(values: Mapping[str, object]) -> None:
    if values:
        raise ValidationError(f"unsupported experiment parameters: {', '.join(sorted(values))}")


def _validate_experiment(experiment: object) -> None:
    if not isinstance(experiment, Experiment):
        raise ValidationError("experiment must be an Experiment instance")


__all__ = ["ExperimentRun", "SweepResult", "load_result", "run_experiment", "run_sweep"]

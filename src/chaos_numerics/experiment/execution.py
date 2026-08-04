"""Experiment execution, deterministic sweeps, and minimal NPZ/JSON storage."""

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
    ReproducibilityWarning,
    Spectrum,
    Trajectory,
    ValidationError,
)
from chaos_numerics.core.types import ArrayLike, ComplexArray, FloatArray
from chaos_numerics.experiment.config import Experiment
from chaos_numerics.operators import UniformPartition, build_ulam, stationary_density

Result: TypeAlias = Trajectory | Spectrum | EigenstateResult | AnalysisResult
RunStatus: TypeAlias = Literal["success", "failed"]
_SCHEMA_VERSION = 1
_SEED_DERIVATION = "sha256-v1"
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_ARRAY_COUNT = 128
_MAX_ARRAY_DIMENSIONS = 32
_MAX_NPY_HEADER_BYTES = 16 * 1024
_MAX_JSON_NESTING = 64
_ALLOWED_PERSISTED_DTYPES = {np.dtype("float64"), np.dtype("complex128")}


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
    values = dict(parameters)
    model: ClassicalMap
    if model_name == "standard_map":
        model = StandardMap(
            kick_strength=_as_float(values.pop("kick_strength", 1.0), name="kick_strength")
        )
    elif model_name == "cat_map":
        model = CatMap(matrix=_as_matrix(values.pop("matrix", ((2, 1), (1, 1)))))
    elif model_name == "baker_map":
        model = BakerMap(cut=_as_float(values.pop("cut", 0.5), name="cut"))
    else:
        raise ValidationError(f"unknown experiment model {model_name!r}")

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
    raise ValidationError(f"unknown experiment analysis {analysis!r}")


def _save_run(run: ExperimentRun, path: Path) -> None:
    if path.is_symlink():
        raise ValidationError(f"refusing to persist through symbolic-link directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    result_payload = run.result.metadata_payload() if run.result is not None else None
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
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
            raise ValidationError("existing sweep manifest does not match this experiment and grid")
        return manifest
    return {
        "schema_version": _SCHEMA_VERSION,
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
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "chaos_numerics": __version__,
        "numpy": np.__version__,
    }


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
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValidationError(
            f"unsupported {name} schema version {payload.get('schema_version')!r}; "
            f"expected {_SCHEMA_VERSION}"
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

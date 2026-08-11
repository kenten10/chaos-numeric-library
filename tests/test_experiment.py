"""Integration tests for experiment execution, persistence, and resume."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import pickle
import subprocess
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import scipy  # type: ignore[import-untyped]

from chaos_numerics.core import (
    AnalysisResult,
    EigenstateResult,
    NumericalWarning,
    ReproducibilityWarning,
    Trajectory,
    ValidationError,
)
from chaos_numerics.experiment import (
    Experiment,
    cartesian_grid,
    execution,
    load_result,
    run_experiment,
    run_sweep,
    zip_grid,
)
from chaos_numerics.quantum import BoundaryPhases, KickedRotor, QuantumBakerMap
from chaos_numerics.quantum import eigenphases as quantum_eigenphases
from chaos_numerics.spectral import SpectralCurve, rmt_reference


def test_parameter_grid_order_and_validation() -> None:
    assert cartesian_grid(kick_strength=[1.0, 2.0], steps=[4, 8]) == (
        {"kick_strength": 1.0, "steps": 4},
        {"kick_strength": 1.0, "steps": 8},
        {"kick_strength": 2.0, "steps": 4},
        {"kick_strength": 2.0, "steps": 8},
    )
    assert zip_grid(kick_strength=[1.0, 2.0], steps=[4, 8]) == (
        {"kick_strength": 1.0, "steps": 4},
        {"kick_strength": 2.0, "steps": 8},
    )
    with pytest.raises(ValidationError, match="equal lengths"):
        zip_grid(kick_strength=[1.0], steps=[4, 8])
    with pytest.raises(ValidationError, match="JSON-compatible"):
        Experiment("standard_map", "iterate", parameters={"bad": np.arange(3)})


def test_single_run_round_trip_separates_arrays_from_json(tmp_path: Path) -> None:
    output = tmp_path / "single"
    experiment = Experiment(
        model="standard_map",
        analysis="iterate",
        parameters={"initial_state": [0.1, 0.2], "kick_strength": 3.0, "steps": 12},
        seed=7,
    )
    run = run_experiment(experiment, output=output)
    restored = load_result(output)

    assert isinstance(run.result, Trajectory)
    assert isinstance(restored.result, Trajectory)
    assert restored.result == run.result
    assert restored.seed == 7
    assert restored.environment["chaos_numerics"]
    assert (output / "arrays.npz").is_file()

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    encoded = json.dumps(metadata)
    assert '"shape": [13, 2]' in encoded
    assert '"states": [[' not in encoded
    assert metadata["result"]["arrays"]["states"] == {
        "dtype": "float64",
        "shape": [13, 2],
        "sha256": metadata["result"]["arrays"]["states"]["sha256"],
    }
    assert len(metadata["result"]["arrays"]["states"]["sha256"]) == 64


def test_load_result_rejects_array_value_tampering(tmp_path: Path) -> None:
    output = tmp_path / "tampered"
    experiment = Experiment("standard_map", "iterate", parameters={"steps": 2}, seed=3)
    run_experiment(experiment, output=output)

    with np.load(output / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["states"][0, 0] += 1.0
    np.savez_compressed(output / "arrays.npz", **arrays)
    _refresh_archive_digest(output)

    with pytest.raises(ValidationError, match="integrity check"):
        load_result(output)


def test_load_result_rejects_oversized_metadata(tmp_path: Path) -> None:
    output = tmp_path / "oversized"
    output.mkdir()
    (output / "metadata.json").write_bytes(b" " * (8 * 1024 * 1024 + 1))

    with pytest.raises(ValidationError, match="metadata exceeds"):
        load_result(output)


def test_load_result_rejects_deeply_nested_metadata(tmp_path: Path) -> None:
    output = tmp_path / "deeply-nested"
    output.mkdir()
    (output / "metadata.json").write_text("[" * 65 + "0" + "]" * 65, encoding="utf-8")

    with pytest.raises(ValidationError, match="nesting levels"):
        load_result(output)


def test_load_result_rejects_unexpected_archive_members(tmp_path: Path) -> None:
    output = tmp_path / "unexpected-member"
    experiment = Experiment("standard_map", "iterate", parameters={"steps": 2}, seed=3)
    run_experiment(experiment, output=output)
    with zipfile.ZipFile(output / "arrays.npz", mode="a") as archive:
        archive.writestr("unexpected.npy", b"not an array")
    _refresh_archive_digest(output)

    with pytest.raises(ValidationError, match="members do not match"):
        load_result(output)


def test_load_result_rejects_archive_corruption(tmp_path: Path) -> None:
    output = tmp_path / "corrupt"
    experiment = Experiment("standard_map", "iterate", parameters={"steps": 2}, seed=3)
    run_experiment(experiment, output=output)
    with (output / "arrays.npz").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValidationError, match=r"archive failed.*integrity"):
        load_result(output)


def test_persistence_rejects_symbolic_link_output(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    experiment = Experiment("standard_map", "iterate", parameters={"steps": 2}, seed=3)

    with pytest.raises(ValidationError, match="symbolic-link"):
        run_experiment(experiment, output=link)


def test_sweep_rejects_symbolic_link_output_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    experiment = Experiment("standard_map", "iterate", parameters={"steps": 2}, seed=3)

    with pytest.raises(ValidationError, match="symbolic-link"):
        run_sweep(experiment, parameters=({},), output=link)


def _refresh_archive_digest(output: Path) -> None:
    archive = output / "arrays.npz"
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["arrays_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


# ---------------------------------------------------------------------------
# Archive preflight
#
# ``_preflight_archive`` inspects the ZIP directory and every NPY header before
# NumPy is allowed to allocate anything, so an attacker-supplied run directory
# cannot turn ``load_result`` into a decompression bomb or a pickle loader. Those
# refusals only run on input no honest writer produces, which is why they are
# reached here by forging the archive rather than by round-tripping a real one.
# ---------------------------------------------------------------------------


def _persisted_run(tmp_path: Path, name: str) -> Path:
    output = tmp_path / name
    experiment = Experiment("standard_map", "iterate", parameters={"steps": 2}, seed=3)
    run_experiment(experiment, output=output)
    return output


def _forge_result_descriptor(output: Path, array: str, field: str, value: object) -> None:
    path = output / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["result"]["arrays"][array][field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")


def _archive_members(output: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(output / "arrays.npz") as archive:
        return {member.filename: archive.read(member.filename) for member in archive.infolist()}


def _rewrite_archive(output: Path, members: Mapping[str, bytes], *, repeat: str = "") -> None:
    with zipfile.ZipFile(output / "arrays.npz", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, blob in members.items():
            archive.writestr(name, blob)
        if repeat:
            archive.writestr(repeat, members[repeat])
    _refresh_archive_digest(output)


def _npy_blob(array: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _retag_npy_version(blob: bytes, major: int, minor: int) -> bytes:
    """Rewrite the two version bytes that follow the six-byte NPY magic."""
    patched = bytearray(blob)
    patched[6], patched[7] = major, minor
    return bytes(patched)


def _promote_npy_to_version_2(blob: bytes) -> bytes:
    """Re-encode a 1.0 member as 2.0, which widens the header length to four bytes."""
    length = int.from_bytes(blob[8:10], "little")
    header, body = blob[10 : 10 + length], blob[10 + length :]
    return b"\x93NUMPY\x02\x00" + len(header).to_bytes(4, "little") + header + body


def _mark_members_encrypted(output: Path) -> None:
    """Set the ZIP general-purpose encryption bit on every central-directory entry.

    ``zipfile`` clears ``ZipInfo.flag_bits`` when it writes a member, so the bit
    has to be set in the finished file. ``infolist()`` reads it back from the
    central directory, which is where ``_preflight_archive`` looks.
    """
    path = output / "arrays.npz"
    data = bytearray(path.read_bytes())
    end_record = data.rfind(b"PK\x05\x06")
    position = int.from_bytes(data[end_record + 16 : end_record + 20], "little")
    while data[position : position + 4] == b"PK\x01\x02":
        data[position + 8] |= 0x01
        name, extra, comment = (
            int.from_bytes(data[position + offset : position + offset + 2], "little")
            for offset in (28, 30, 32)
        )
        position += 46 + name + extra + comment
    path.write_bytes(bytes(data))
    _refresh_archive_digest(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", "float32", r"array 'states' uses unsupported dtype 'float32'"),
        ("dtype", "O", "cannot use an object dtype"),
        ("dtype", "definitely-not-a-dtype", "invalid dtype descriptor"),
        ("shape", "13", "invalid shape descriptor"),
        ("shape", [-1, 2], "negative dimension"),
        ("shape", [1] * 33, "more than 32 dimensions"),
        ("shape", [1 << 26], "uncompressed bytes"),
    ],
)
def test_preflight_rejects_forged_array_descriptors(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Every descriptor is vetted before NumPy sees the archive.

    The dtype allowlist is what keeps ``allow_pickle=False`` from being the only
    thing standing between a forged run directory and arbitrary object
    deserialization, and the shape and byte budgets bound the allocation the load
    is about to request. The archive itself is untouched, so its digest still
    verifies and each case really is the descriptor check firing.
    """
    output = _persisted_run(tmp_path, "forged-descriptor")
    _forge_result_descriptor(output, "states", field, value)

    with pytest.raises(ValidationError, match=message):
        load_result(output)


def test_preflight_rejects_an_oversized_descriptor_count(tmp_path: Path) -> None:
    output = _persisted_run(tmp_path, "too-many-arrays")
    path = output / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    descriptors = metadata["result"]["arrays"]
    template = descriptors["states"]
    descriptors.update({f"filler{index}": template for index in range(129)})
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValidationError, match="more than 128 arrays"):
        load_result(output)


def test_preflight_rejects_encrypted_archive_members(tmp_path: Path) -> None:
    """An encrypted member cannot be validated, so it is refused rather than opened."""
    output = _persisted_run(tmp_path, "encrypted")
    _mark_members_encrypted(output)

    with pytest.raises(ValidationError, match="contains an encrypted member"):
        load_result(output)


def test_preflight_rejects_an_unsupported_npy_version(tmp_path: Path) -> None:
    """Only the two NPY header layouts this loader can parse are accepted."""
    output = _persisted_run(tmp_path, "npy-version")
    members = _archive_members(output)
    _rewrite_archive(
        output, {name: _retag_npy_version(blob, 3, 0) for name, blob in members.items()}
    )

    with pytest.raises(ValidationError, match=r"unsupported NPY version \(3, 0\)"):
        load_result(output)


def test_preflight_accepts_npy_version_2_0(tmp_path: Path) -> None:
    """The 2.0 layout is a supported header width, not an unknown version.

    NumPy only writes it when a header exceeds 64 KiB, which no result in this
    library reaches, so the accepting branch is unreachable from a real save and
    is pinned by re-encoding a real archive instead.
    """
    output = _persisted_run(tmp_path, "npy-version-2")
    members = _archive_members(output)
    _rewrite_archive(output, {n: _promote_npy_to_version_2(b) for n, b in members.items()})

    restored = load_result(output)
    assert restored.status == "success"
    assert isinstance(restored.result, Trajectory)


def test_preflight_rejects_duplicate_archive_members(tmp_path: Path) -> None:
    """Two members with one name make the descriptor set an ambiguous claim.

    ``zipfile`` reports the second write as a ``UserWarning`` rather than refusing
    it, which is exactly the shape of archive a hand-built ZIP can carry, so the
    duplicate count is checked before the member names are compared.
    """
    output = _persisted_run(tmp_path, "duplicate-member")
    members = _archive_members(output)
    with pytest.warns(UserWarning, match="Duplicate name"):
        _rewrite_archive(output, members, repeat="states.npy")

    with pytest.raises(ValidationError, match="duplicate member names"):
        load_result(output)


def test_preflight_rejects_a_member_header_contradicting_its_descriptor(tmp_path: Path) -> None:
    """The NPY header is compared against the descriptor before any data is read."""
    output = _persisted_run(tmp_path, "header-mismatch")
    members = _archive_members(output)
    members["states.npy"] = _npy_blob(np.zeros((2, 13), dtype=np.float64))
    _rewrite_archive(output, members)

    with pytest.raises(ValidationError, match=r"member 'states.npy' does not match its descriptor"):
        load_result(output)


def test_preflight_reports_a_truncated_member_as_a_validation_error(tmp_path: Path) -> None:
    """A header that runs off the end of the member is a refusal, not an ``EOFError``."""
    output = _persisted_run(tmp_path, "truncated-member")
    members = _archive_members(output)
    members["states.npy"] = b"\x93NUMPY\x01\x00"
    _rewrite_archive(output, members)

    with pytest.raises(ValidationError, match="could not inspect persisted arrays"):
        load_result(output)


def test_load_result_requires_readable_metadata_a_digest_and_a_known_status(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(ValidationError, match="could not read persisted metadata"):
        load_result(missing)

    output = _persisted_run(tmp_path, "no-digest")
    path = output / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["arrays_sha256"]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValidationError, match="missing a valid arrays SHA-256 digest"):
        load_result(output)

    metadata["arrays_sha256"] = "0" * 64
    metadata["status"] = "half-finished"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValidationError, match="unknown persisted run status 'half-finished'"):
        load_result(output)


def test_provenance_helpers_return_none_when_the_environment_cannot_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance is recorded when it can be, and omitted when it cannot.

    A machine with no ``git`` on ``PATH`` or a NumPy built without a config dump
    must still produce a loadable run, so every one of these probes degrades to
    ``None`` instead of raising.
    """

    def unavailable(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", unavailable)
    assert execution._git_commit() is None
    assert execution._git_is_dirty() is None

    monkeypatch.setattr(np.__config__, "show", None, raising=False)
    assert execution._blas_description() is None

    for value in ("not a mapping", {"Build Dependencies": "not a mapping"}, {}):
        monkeypatch.setattr(np.__config__, "show", lambda mode, result=value: result, raising=False)
        assert execution._blas_description() is None

    def refuses(mode: str) -> object:
        raise TypeError(mode)

    monkeypatch.setattr(np.__config__, "show", refuses, raising=False)
    assert execution._blas_description() is None


def test_sweep_argument_validation_and_refusals(tmp_path: Path) -> None:
    experiment = Experiment("baker_map", "iterate", parameters={"steps": 2}, seed=6)

    with pytest.raises(ValidationError, match="resume and continue_on_error must be bool"):
        run_sweep(experiment, parameters=({},), output=tmp_path / "flags", resume="yes")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="duplicate parameter combinations"):
        run_sweep(experiment, parameters=({"cut": 0.5}, {"cut": 0.5}), output=tmp_path / "dupes")
    with pytest.raises(ValidationError, match="parameters must be a mapping"):
        run_experiment(experiment, parameters=[("cut", 0.5)])  # type: ignore[arg-type]

    output = tmp_path / "exists"
    sweep = run_sweep(experiment, parameters=({"cut": 0.4},), output=output)
    assert sweep.succeeded == 1
    with pytest.raises(ValidationError, match="sweep output already exists"):
        run_sweep(experiment, parameters=({"cut": 0.4},), output=output, resume=False)


def test_seeded_sweep_is_reproducible_and_order_independent(tmp_path: Path) -> None:
    experiment = Experiment(
        "cat_map",
        "ulam_stationary_density",
        parameters={"partition_shape": [2, 2], "samples_per_cell": 24, "strict": False},
        seed=90210,
    )
    grid = ({"tolerance": 1e-8}, {"tolerance": 1e-9})
    first = run_sweep(experiment, parameters=grid, output=tmp_path / "first")
    second = run_sweep(experiment, parameters=reversed(grid), output=tmp_path / "second")
    first_by_id = {run.job_id: run for run in first.runs}
    second_by_id = {run.job_id: run for run in second.runs}

    assert first.succeeded == second.succeeded == 2
    assert first_by_id.keys() == second_by_id.keys()
    for job_id, left in first_by_id.items():
        right = second_by_id[job_id]
        assert left.seed == right.seed
        assert left.result == right.result


def test_resume_skips_success_and_reruns_incomplete_job(tmp_path: Path) -> None:
    output = tmp_path / "sweep"
    experiment = Experiment(
        "standard_map",
        "iterate",
        parameters={"initial_state": [0.1, 0.2], "steps": 5},
        seed=11,
    )
    grid = cartesian_grid(kick_strength=[1.0, 2.0])
    initial = run_sweep(experiment, parameters=grid, output=output)
    preserved_path = initial.runs[0].output_path
    incomplete_path = initial.runs[1].output_path
    assert preserved_path is not None and incomplete_path is not None
    preserved_mtime = (preserved_path / "metadata.json").stat().st_mtime_ns

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["jobs"][initial.runs[1].job_id]["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (incomplete_path / "metadata.json").unlink()
    time.sleep(0.001)

    resumed = run_sweep(experiment, parameters=grid, output=output, resume=True)
    assert resumed.succeeded == 2
    assert (preserved_path / "metadata.json").stat().st_mtime_ns == preserved_mtime
    assert (incomplete_path / "metadata.json").is_file()


def test_sweep_continues_after_failure_and_records_error(tmp_path: Path) -> None:
    experiment = Experiment("standard_map", "iterate", parameters={"steps": 2}, seed=5)
    sweep = run_sweep(
        experiment,
        parameters=({"kick_strength": 1.0}, {"kick_strength": "invalid"}),
        output=tmp_path / "failure",
    )
    assert sweep.succeeded == 1
    assert sweep.failed == 1
    failed = next(run for run in sweep.runs if run.status == "failed")
    assert failed.seed is not None
    assert failed.error is not None
    assert failed.error["type"] == "ValidationError"
    restored = load_result(failed.output_path or Path())
    assert restored.status == "failed"
    assert restored.result is None


def test_seed_policy_and_manifest_compatibility(tmp_path: Path) -> None:
    unseeded = Experiment("standard_map", "iterate", parameters={"steps": 1})
    with pytest.warns(ReproducibilityWarning):
        run_experiment(unseeded, output=tmp_path / "warning")
    strict = Experiment(
        "standard_map",
        "iterate",
        parameters={"steps": 1},
        strict_reproducibility=True,
    )
    with pytest.raises(ValidationError, match="declare a seed"):
        run_experiment(strict, output=tmp_path / "strict")

    experiment = Experiment("standard_map", "iterate", parameters={"steps": 1}, seed=4)
    output = tmp_path / "compatibility"
    run_sweep(experiment, parameters=({"kick_strength": 1.0},), output=output)
    with pytest.raises(ValidationError, match="different experiment or parameter grid") as mismatch:
        run_sweep(experiment, parameters=({"kick_strength": 2.0},), output=output)
    # A refusal that does not say how to proceed just moves the problem to the user.
    message = str(mismatch.value)
    assert "resume=False" in message
    assert str(output) in message


def test_experiment_and_run_survive_pickle_for_parallel_sweeps() -> None:
    """A frozen ``Experiment`` stores ``MappingProxyType`` and must still pickle."""
    experiment = Experiment(
        "cat_map",
        "iterate",
        parameters={"steps": 4, "matrix": ((2, 1), (1, 1))},
        seed=11,
    )

    restored = pickle.loads(pickle.dumps(experiment))
    copied = copy.deepcopy(experiment)

    assert restored == experiment
    assert copied == experiment
    assert restored.to_dict() == experiment.to_dict()

    run = run_experiment(experiment)
    restored_run = pickle.loads(pickle.dumps(run))
    assert restored_run.result == run.result
    assert dict(restored_run.environment) == dict(run.environment)


def test_environment_records_the_versions_the_standards_document_requires() -> None:
    """``numerical-standards.md`` section 10.1 requires SciPy, not just NumPy.

    SciPy drives ARPACK and the sparse eigensolvers, so a run recorded without
    it is not reproducible. BLAS identity and thread limits change summation
    order, and a dirty worktree must never be persisted as a clean commit.
    """
    run = run_experiment(Experiment("cat_map", "iterate", parameters={"steps": 3}))
    environment = dict(run.environment)

    for key in ("python", "implementation", "platform", "chaos_numerics", "numpy", "scipy"):
        assert isinstance(environment[key], str) and environment[key]
    assert environment["scipy"] == scipy.__version__
    assert environment["numpy"] == np.__version__
    if "git_dirty" in environment:
        assert isinstance(environment["git_dirty"], bool)
    if "thread_limits" in environment:
        limits = environment["thread_limits"]
        assert isinstance(limits, dict)
        assert all(isinstance(value, str) for value in limits.values())
    # Everything recorded must survive the JSON persistence path.
    assert json.loads(json.dumps(environment)) == environment


def test_unknown_model_and_analysis_messages_enumerate_valid_values() -> None:
    with pytest.raises(ValidationError, match=r"unknown experiment model 'nope_model'; expected"):
        run_experiment(Experiment("nope_model", "iterate", parameters={"steps": 2}))
    with pytest.raises(ValidationError, match=r"expected one of .*'standard_map'"):
        run_experiment(Experiment("nope_model", "iterate", parameters={"steps": 2}))
    with pytest.raises(ValidationError, match=r"unknown experiment analysis 'nope'; expected"):
        run_experiment(Experiment("cat_map", "nope", parameters={"steps": 2}))
    with pytest.raises(ValidationError, match=r"expected one of .*'ulam_stationary_density'"):
        run_experiment(Experiment("cat_map", "nope", parameters={"steps": 2}))


# ---------------------------------------------------------------------------
# Quantum models and spectral statistics in the sweep framework
#
# The README teaches that the ramp-plateau form factor needs a Bloch-phase
# ensemble average. Until the quantum models entered the registry that average
# could only be written as a hand-rolled loop, which is exactly the work
# ``run_sweep`` exists to do, so the tests below drive it through the framework.
# ---------------------------------------------------------------------------

#: Twelve generic Bloch phases. Both twists are irrational-looking and neither
#: is 0 or 1/2, so parity and the antiunitary symmetry are broken and each
#: spectrum is a single CUE-like block that needs no desymmetrization.
BLOCH_PHASES: tuple[list[float], ...] = tuple(
    [
        round(0.12 + 0.76 * ((0.11 + index * 0.6180339887) % 1.0), 6),
        round(0.12 + 0.76 * ((0.29 + index * 0.4142135624) % 1.0), 6),
    ]
    for index in range(12)
)
SFF_TIMES: tuple[float, ...] = tuple(round(value, 6) for value in np.linspace(0.05, 2.0, 79))
CUE_SECTOR = "single sector: parity and antiunitary symmetry both broken"


def _quantum_experiment(analysis: str, **parameters: object) -> Experiment:
    base: dict[str, object] = {"dimension": 128, "kick_strength": 10.0}
    base.update(parameters)
    return Experiment("kicked_rotor", analysis, parameters=base, seed=20240808)


def test_quantum_registry_names_appear_in_the_enumerated_error_messages() -> None:
    with pytest.raises(ValidationError, match=r"expected one of .*'quantum_baker_map'"):
        run_experiment(Experiment("nope_model", "eigenphases", parameters={"dimension": 8}))
    with pytest.raises(ValidationError, match=r"expected one of .*'spectral_form_factor'"):
        run_experiment(Experiment("kicked_rotor", "nope", parameters={"dimension": 8}))
    assert set(execution._QUANTUM_MODELS) <= set(execution._EXPERIMENT_MODELS)
    assert set(execution._QUANTUM_ANALYSES) <= set(execution._EXPERIMENT_ANALYSES)


def test_a_mismatched_model_and_analysis_say_they_cannot_be_combined() -> None:
    """A quantum model with a classical analysis is a pairing error, not an unknown name.

    Before the two families were separated this fell through to the classical
    Lyapunov routine and died on a missing ``jacobian``, which reads as a defect
    in ``KickedRotor`` rather than as a request that does not typecheck.
    """
    with pytest.raises(ValidationError) as quantum_model:
        run_experiment(
            Experiment("kicked_rotor", "lyapunov_spectrum", parameters={"dimension": 8, "steps": 4})
        )
    message = str(quantum_model.value)
    assert "cannot be combined" in message
    assert "'lyapunov_spectrum' applies to classical models" in message
    assert "'spectral_form_factor'" in message

    with pytest.raises(ValidationError) as classical_model:
        run_experiment(Experiment("standard_map", "spectral_form_factor", parameters={"steps": 4}))
    assert "applies to quantum models" in str(classical_model.value)
    assert "'ulam_stationary_density'" in str(classical_model.value)


def test_boundary_phases_accept_the_three_json_encodings() -> None:
    """``BoundaryPhases`` is not JSON, so three JSON spellings build the same twist.

    The mapping form matters most: it is what ``BoundaryPhases.to_dict()``
    returns and therefore what a persisted result carries, so a phase read back
    off disk has to be usable as a parameter without reshaping.
    """
    reference = quantum_eigenphases(KickedRotor(16, 3.0, BoundaryPhases(0.25, 0.25)))
    expected = np.asarray(reference.values)
    for encoding in (0.25, [0.25, 0.25], {"position": 0.25, "momentum": 0.25}):
        run = run_experiment(
            Experiment(
                "kicked_rotor",
                "eigenphases",
                parameters={"dimension": 16, "kick_strength": 3.0, "boundary_phases": encoding},
            )
        )
        assert isinstance(run.result, AnalysisResult)
        # Compared at rounding level, not with `==`. The three encodings build the
        # same operator, but on the oldest supported NumPy a complex128 elementwise
        # multiply is not reproducible call to call (see the reproducibility section
        # of docs/design/numerical-standards.md), so two eigenphase arrays computed
        # from equal-but-distinct inputs differ in the last bits.
        np.testing.assert_allclose(run.result.values, expected, rtol=0.0, atol=1e-13)

    asymmetric = run_experiment(
        Experiment(
            "kicked_rotor",
            "eigenphases",
            parameters={"dimension": 16, "kick_strength": 3.0, "boundary_phases": [0.25, 0.13]},
        )
    )
    assert isinstance(asymmetric.result, AnalysisResult)
    # A different twist is a different operator, so this separation is physical and
    # far above any rounding: measured 0.30 in the maximum eigenphase difference.
    assert float(np.abs(np.asarray(asymmetric.result.values) - expected).max()) > 1e-3

    for bad, message in (
        ([0.1, 0.2, 0.3], "must be .position, momentum."),
        ({"position": 0.1, "phase": 0.2}, "accepts only 'position' and 'momentum'"),
        ("0.25", "must be a real number"),
        (None, "must be a real number"),
    ):
        with pytest.raises(ValidationError, match=message):
            run_experiment(
                Experiment(
                    "kicked_rotor",
                    "eigenphases",
                    parameters={"dimension": 16, "boundary_phases": bad},
                )
            )


def test_quantum_eigenphases_and_eigenstates_reuse_the_public_results(tmp_path: Path) -> None:
    """The registry must not reimplement the analysis, only route parameters to it."""
    model = QuantumBakerMap(dimension=126)
    phases_output = tmp_path / "eigenphases"
    states_output = tmp_path / "eigenstates"
    declaration: dict[str, object] = {"dimension": 126}

    phases = run_experiment(
        Experiment("quantum_baker_map", "eigenphases", parameters=declaration, seed=1),
        output=phases_output,
    )
    states = run_experiment(
        Experiment("quantum_baker_map", "eigenstates", parameters=declaration, seed=1),
        output=states_output,
    )

    assert isinstance(phases.result, AnalysisResult)
    assert phases.result == quantum_eigenphases(model)
    assert isinstance(states.result, EigenstateResult)
    # Eigenvectors are O(N**2) on disk; the point of persisting them at all is
    # that a later desymmetrization needs the basis, not just the phases.
    assert states.result.eigenstates.shape == (126, 126)
    assert load_result(phases_output).result == phases.result
    assert load_result(states_output).result == states.result

    resolved = run_experiment(
        Experiment(
            "quantum_baker_map",
            "eigenphases",
            parameters={"dimension": 126, "parity_sector": "even"},
        )
    )
    assert isinstance(resolved.result, AnalysisResult)
    assert resolved.result.values.size == 63
    assert resolved.result.metadata.parameters["symmetry_sector"] == "even"


def test_parity_sector_is_refused_when_the_model_has_no_parity() -> None:
    with pytest.raises(ValidationError, match="exposes no parity symmetry"):
        run_experiment(
            Experiment(
                "kicked_rotor",
                "eigenphases",
                parameters={
                    "dimension": 16,
                    "boundary_phases": [0.25, 0.13],
                    "parity_sector": "even",
                },
            )
        )
    with pytest.raises(ValidationError, match="parity_sector must be 'even', 'odd', or null"):
        run_experiment(
            Experiment(
                "quantum_baker_map",
                "eigenphases",
                parameters={"dimension": 8, "parity_sector": "both"},
            )
        )


def test_spectral_curve_round_trips_with_and_without_an_error_estimate(tmp_path: Path) -> None:
    """``result_type='spectral_curve'`` needs its own restore branch, absent arrays included.

    ``uncertainty`` and ``variance`` are omitted from the archive rather than
    written as zeros when no bootstrap ran, so ``None`` has to survive the round
    trip as ``None`` and not come back as an all-zero curve.
    """
    plain_output = tmp_path / "plain"
    plain = run_experiment(
        _quantum_experiment(
            "spectral_form_factor",
            dimension=64,
            boundary_phases=list(BLOCH_PHASES[0]),
            symmetry_sector=CUE_SECTOR,
            times=[0.1, 0.5, 1.0, 1.5],
        ),
        output=plain_output,
    )
    assert isinstance(plain.result, SpectralCurve)
    assert plain.result.uncertainty is None and plain.result.variance is None
    restored = load_result(plain_output)
    assert isinstance(restored.result, SpectralCurve)
    assert restored.result == plain.result
    assert restored.result.uncertainty is None and restored.result.variance is None

    metadata = json.loads((plain_output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["result"]["result_type"] == "spectral_curve"
    assert set(metadata["result"]["arrays"]) == {"x", "values"}

    bootstrapped_output = tmp_path / "bootstrapped"
    # The bootstrap path warns that its spread is a resampling diagnostic and not an
    # error bar on K(tau); what is under test here is that the arrays persist.
    with pytest.warns(NumericalWarning, match="resampling diagnostic"):
        bootstrapped = run_experiment(
            _quantum_experiment(
                "spectral_form_factor",
                dimension=64,
                boundary_phases=list(BLOCH_PHASES[0]),
                symmetry_sector=CUE_SECTOR,
                times=[0.1, 0.5, 1.0, 1.5],
                window="hann",
                connected=False,
                bootstrap=16,
            ),
            output=bootstrapped_output,
        )
    assert isinstance(bootstrapped.result, SpectralCurve)
    assert bootstrapped.result.uncertainty is not None
    assert bootstrapped.result.variance is not None
    assert load_result(bootstrapped_output).result == bootstrapped.result
    encoded = json.loads((bootstrapped_output / "metadata.json").read_text(encoding="utf-8"))
    assert set(encoded["result"]["arrays"]) == {"x", "values", "uncertainty", "variance"}


def test_restore_rejects_a_result_type_it_does_not_know(tmp_path: Path) -> None:
    output = _persisted_run(tmp_path, "unknown-type")
    path = output / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["result"]["result_type"] = "spectral_curves"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValidationError, match="unknown persisted result type 'spectral_curves'"):
        load_result(output)


def test_sweeping_bloch_phases_averages_the_form_factor_onto_the_cue_ramp_plateau(
    tmp_path: Path,
) -> None:
    """The headline figure, produced by the headline feature.

    ``K(tau)`` does not self-average: one spectrum fluctuates by order one at
    every ``tau`` however large ``N`` is, and only an ensemble average converges
    onto the CUE ``min(tau, 1)`` ramp and plateau. That average is a parameter
    sweep over Bloch phases, so it must be expressible as one ``run_sweep``
    call rather than as a loop in a notebook.

    Measured at ``N = 128`` over the twelve phases of ``BLOCH_PHASES``: the mean
    of the ensemble-averaged curve over the ramp ``0.1 <= tau <= 0.9`` is 1.080
    times the CUE reference mean, the plateau mean over ``tau >= 1.2`` is 0.982
    against 1, and the RMS distance to CUE falls from 0.770 for a single member
    to 0.205 for the average -- a factor of 3.8. The bands below are wide enough
    to absorb a different BLAS but narrow enough that losing the average, or
    losing the unfolding, fails them.
    """
    output = tmp_path / "bloch-ensemble"
    experiment = Experiment(
        "kicked_rotor",
        "spectral_form_factor",
        parameters={
            "dimension": 128,
            "kick_strength": 10.0,
            "symmetry_sector": CUE_SECTOR,
            "times": list(SFF_TIMES),
        },
        seed=20240808,
    )
    sweep = run_sweep(
        experiment,
        parameters=cartesian_grid(boundary_phases=[list(pair) for pair in BLOCH_PHASES]),
        output=output,
    )

    assert sweep.succeeded == len(BLOCH_PHASES)
    curves = []
    for run in sweep.runs:
        assert isinstance(run.result, SpectralCurve)
        assert np.allclose(run.result.x, SFF_TIMES)
        curves.append(np.asarray(run.result.values))
    average = np.mean(curves, axis=0)
    times = np.asarray(SFF_TIMES)
    reference = np.asarray(rmt_reference("spectral_form_factor", "cue", times).values)

    ramp = (times >= 0.1) & (times <= 0.9)
    plateau = times >= 1.2
    ramp_ratio = float(average[ramp].mean() / reference[ramp].mean())
    plateau_height = float(average[plateau].mean())
    average_distance = float(np.sqrt(np.mean((average - reference) ** 2)))
    member_distance = float(
        np.mean([np.sqrt(np.mean((curve - reference) ** 2)) for curve in curves])
    )

    assert 0.80 < ramp_ratio < 1.30, ramp_ratio
    assert 0.80 < plateau_height < 1.25, plateau_height
    # The whole reason the ensemble is needed: the average is much closer to CUE
    # than a typical member of it.
    assert average_distance < 0.5 * member_distance, (average_distance, member_distance)

    # The persisted sweep is the deliverable, not just the in-memory curves.
    reloaded = [load_result(run.output_path or Path()) for run in sweep.runs]
    for run, again in zip(sweep.runs, reloaded, strict=True):
        assert again.result == run.result


def test_number_variance_sweep_child_seeds_are_bit_reproducible(tmp_path: Path) -> None:
    """``sha256-v1`` child seeds must reach the bootstrap RNG unchanged.

    The quantum models are deterministic, so ``number_variance`` with a positive
    ``bootstrap`` is where a lost seed would actually show up: the values would
    agree and only the error bars would drift.
    """
    experiment = Experiment(
        "quantum_baker_map",
        "number_variance",
        parameters={
            "dimension": 126,
            "parity_sector": "even",
            "lengths": [1.0, 2.0, 4.0, 8.0],
            "samples": 512,
            "bootstrap": 32,
        },
        seed=4242,
    )
    grid = cartesian_grid(unfold_method=["mean", "polynomial"])
    # L=8 on 63 levels leaves seven arcs, below the eight the batch-means error bar
    # needs, so each run warns; the seeds reaching the RNG are what is under test.
    with pytest.warns(NumericalWarning, match="leave fewer than 8 independent windows"):
        first = run_sweep(experiment, parameters=grid, output=tmp_path / "first")
        second = run_sweep(experiment, parameters=reversed(grid), output=tmp_path / "second")

    assert first.succeeded == second.succeeded == 2
    first_by_id = {run.job_id: run for run in first.runs}
    second_by_id = {run.job_id: run for run in second.runs}
    assert first_by_id.keys() == second_by_id.keys()
    for job_id, left in first_by_id.items():
        right = second_by_id[job_id]
        assert left.seed == right.seed
        assert isinstance(left.result, SpectralCurve)
        assert isinstance(right.result, SpectralCurve)
        assert left.result == right.result
        assert left.result.uncertainty is not None
        assert right.result.uncertainty is not None
        # Bit equality, not allclose: a reseeded bootstrap would still be close.
        assert np.array_equal(left.result.uncertainty, right.result.uncertainty)
        assert left.result.metadata.parameters["seed"] == left.seed
        assert left.result.metadata.parameters["symmetry_sector"] == "parity-even"


def test_a_sweep_without_a_symmetry_sector_lets_the_warning_through(tmp_path: Path) -> None:
    """Omitting ``symmetry_sector`` warns, and the sweep neither suppresses nor requires it.

    Comparing an unresolved spectrum with RMT is a real mistake, so the
    framework must not swallow the diagnostic; it must also not refuse the run,
    because a caller may legitimately be looking at the raw spectrum. The
    warning is both raised and recorded in the persisted metadata.
    """
    experiment = _quantum_experiment(
        "spectral_form_factor",
        dimension=64,
        times=[0.25, 0.75, 1.5],
    )
    with pytest.warns(NumericalWarning, match="symmetry sector is unknown"):
        sweep = run_sweep(
            experiment,
            parameters=cartesian_grid(boundary_phases=[list(BLOCH_PHASES[0])]),
            output=tmp_path / "unlabelled",
        )
    assert sweep.succeeded == 1
    result = sweep.runs[0].result
    assert isinstance(result, SpectralCurve)
    assert result.metadata.parameters["symmetry_sector"] is None

    labelled = run_experiment(
        _quantum_experiment(
            "spectral_form_factor",
            dimension=64,
            symmetry_sector=CUE_SECTOR,
            times=[0.25, 0.75, 1.5],
        )
    )
    assert isinstance(labelled.result, SpectralCurve)
    assert labelled.result.metadata.parameters["symmetry_sector"] == CUE_SECTOR


def test_quantum_parameters_are_validated_and_unused_ones_refused() -> None:
    declarations: tuple[tuple[dict[str, object], str], ...] = (
        ({}, "dimension must be an integer"),
        ({"dimension": 16, "kick_strength": 3.0}, "unsupported experiment parameters"),
        ({"dimension": 8, "symmetry_sector": " padded "}, "non-empty trimmed string"),
    )
    for parameters, message in declarations:
        with pytest.raises(ValidationError, match=message):
            run_experiment(Experiment("quantum_baker_map", "eigenphases", parameters=parameters))

    # Every curve parameter is validated before the O(N**3) diagonalization runs,
    # so these refusals never reach the eigensolver at all.
    overrides: tuple[tuple[dict[str, object], str], ...] = (
        ({"unfold_method": "cubic"}, "unfold_method must be 'mean' or 'polynomial'"),
        ({"window": "hamming"}, "window must be 'none' or 'hann'"),
        ({"times": None}, "times must be array-like"),
    )
    for parameters, message in overrides:
        merged: dict[str, object] = {
            "dimension": 16,
            "symmetry_sector": CUE_SECTOR,
            "times": [0.5],
        }
        merged.update(parameters)
        with pytest.raises(ValidationError, match=message):
            run_experiment(Experiment("quantum_cat_map", "spectral_form_factor", parameters=merged))

    with pytest.raises(ValidationError, match="lengths must be array-like"):
        run_experiment(
            Experiment(
                "quantum_cat_map",
                "number_variance",
                parameters={"dimension": 16, "symmetry_sector": CUE_SECTOR},
            )
        )


def test_the_dense_limit_guard_is_not_silently_raised_by_the_sweep() -> None:
    """A sweep that densifies a huge model by accident is what the guard prevents."""
    with pytest.raises(ValidationError, match="exceeds dense_limit"):
        run_experiment(Experiment("kicked_rotor", "eigenphases", parameters={"dimension": 1024}))
    run = run_experiment(
        Experiment(
            "kicked_rotor",
            "eigenphases",
            parameters={"dimension": 640, "dense_limit": 640},
        )
    )
    assert isinstance(run.result, AnalysisResult)
    assert run.result.values.size == 640


def test_cylinder_rotor_and_cat_map_are_reachable_from_the_registry(tmp_path: Path) -> None:
    cylinder = run_experiment(
        Experiment(
            "cylinder_kicked_rotor",
            "eigenphases",
            parameters={"dimension": 24, "kick_strength": 5.0, "effective_hbar": 1.5},
            seed=2,
        ),
        output=tmp_path / "cylinder",
    )
    assert isinstance(cylinder.result, AnalysisResult)
    assert cylinder.result.values.size == 24
    assert load_result(tmp_path / "cylinder").result == cylinder.result

    cat = run_experiment(
        Experiment(
            "quantum_cat_map",
            "eigenphases",
            parameters={"dimension": 16, "matrix": [[3, 1], [2, 1]]},
        )
    )
    assert isinstance(cat.result, AnalysisResult)
    assert cat.result.values.size == 16


def test_cylinder_rotor_spectral_statistics_warn_that_the_lattice_is_a_truncation() -> None:
    """The cylinder rotor's spectrum belongs to the truncation, not only to the rotor.

    ``CylinderKickedRotor`` documents this, but the registry is where the pairing
    of that model with a level statistic actually gets written down, so the
    warning has to fire there. ``eigenphases`` alone is fine -- computing the
    spectrum is not the mistake, comparing it with RMT is.
    """
    curve = Experiment(
        "cylinder_kicked_rotor",
        "number_variance",
        parameters={
            "dimension": 32,
            "kick_strength": 5.0,
            "symmetry_sector": "momentum-lattice truncation",
            "lengths": [1.0, 2.0],
            "samples": 128,
        },
    )
    with pytest.warns(NumericalWarning, match="truncation of an infinite momentum lattice"):
        run = run_experiment(curve)
    assert isinstance(run.result, SpectralCurve)

    phases = run_experiment(
        Experiment(
            "cylinder_kicked_rotor",
            "eigenphases",
            parameters={"dimension": 32, "kick_strength": 5.0},
        )
    )
    assert isinstance(phases.result, AnalysisResult)

"""Integration tests for experiment execution, persistence, and resume."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from chaos_numerics.core import ReproducibilityWarning, Trajectory, ValidationError
from chaos_numerics.experiment import (
    Experiment,
    cartesian_grid,
    load_result,
    run_experiment,
    run_sweep,
    zip_grid,
)


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
    with pytest.raises(ValidationError, match="does not match"):
        run_sweep(experiment, parameters=({"kick_strength": 2.0},), output=output)

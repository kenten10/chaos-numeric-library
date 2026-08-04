"""Quickstart: run, persist, and reload one classical experiment."""

from pathlib import Path
from tempfile import TemporaryDirectory

from chaos_numerics import Experiment, run_experiment
from chaos_numerics.experiment import load_result

experiment = Experiment(
    "standard_map",
    "iterate",
    parameters={"kick_strength": 3.0, "initial_state": [0.1, 0.2], "steps": 20},
    seed=7,
)
with TemporaryDirectory() as directory:
    output = Path(directory) / "run"
    run_experiment(experiment, output=output)
    restored = load_result(output)
    assert restored.status == "success"
    assert restored.result is not None
    print(restored.result)

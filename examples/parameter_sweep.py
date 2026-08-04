"""Run and resume a small standard-map Lyapunov sweep."""

from __future__ import annotations

import argparse

from chaos_numerics.experiment import Experiment, cartesian_grid, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/standard-map")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    experiment = Experiment(
        model="standard_map",
        analysis="lyapunov_spectrum",
        parameters={"initial_state": [0.1, 0.2], "transient": 50, "strict": False},
        seed=args.seed,
    )
    summary = run_sweep(
        experiment,
        parameters=cartesian_grid(kick_strength=[2.0, 4.0], steps=[200, 400]),
        output=args.output,
        resume=True,
    )
    print(f"completed={summary.succeeded} failed={summary.failed} output={summary.output_path}")


if __name__ == "__main__":
    main()

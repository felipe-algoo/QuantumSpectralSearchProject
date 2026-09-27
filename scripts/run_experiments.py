from __future__ import annotations
from pathlib import Path
import sys
import json
from qss.config import load_config
from qss.experiments import run_experiment_suite, ExperimentConfig
from qss.utils import ensure_directory, save_json


def main(config_path: str = "config/parameters.yaml") -> int:
    raw = load_config(config_path)
    config = ExperimentConfig(
        base_seed=raw.base_seed,
        graph_types=raw.graph_types,
        graph_sizes=raw.graph_sizes,
        n_instances=raw.n_instances,
        noise_levels=raw.noise_levels,
        gamma_strategy=raw.gamma_strategy,
        t_max_factor=raw.t_max_factor,
        n_time_steps=raw.n_time_steps,
        use_laplacian=raw.use_laplacian,
        graph_kwargs=raw.graph_kwargs,
        data_dir=raw.data_dir,
        results_dir=raw.results_dir,
        figures_dir=raw.figures_dir,
    )
    output_dir = ensure_directory(config.results_dir)
    print(f"Running experiment suite with base_seed={config.base_seed}")
    df = run_experiment_suite(config, output_dir)
    print(f"Completed {df.shape[0]} experiments.")
    print(df.groupby("graph_type")["success_probability"].describe())
    return 0


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/parameters.yaml"
    raise SystemExit(main(cfg_path))
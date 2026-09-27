from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    base_seed: int
    graph_types: list[str]
    graph_sizes: list[int]
    n_instances: int
    noise_levels: list[float]
    gamma_strategy: str
    t_max_factor: float
    n_time_steps: int
    use_laplacian: bool
    graph_kwargs: dict[str, dict[str, Any]]
    data_dir: str
    results_dir: str
    figures_dir: str


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    exp = raw["experiment"]
    output = raw["output"]
    return ExperimentConfig(
        base_seed=exp["base_seed"],
        graph_types=list(exp["graph_types"]),
        graph_sizes=list(exp["graph_sizes"]),
        n_instances=exp["n_instances"],
        noise_levels=list(exp["noise_levels"]),
        gamma_strategy=exp["gamma_strategy"],
        t_max_factor=exp["t_max_factor"],
        n_time_steps=exp["n_time_steps"],
        use_laplacian=exp["use_laplacian"],
        graph_kwargs=exp.get("graph_kwargs", {}),
        data_dir=output["data_dir"],
        results_dir=output["results_dir"],
        figures_dir=output["figures_dir"],
    )
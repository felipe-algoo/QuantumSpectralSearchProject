from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .graphs import make_generator, adjacency_matrix, laplacian_matrix
from .spectral import (
    compute_adjacency_spectrum,
    compute_laplacian_spectrum,
    spectral_gap,
    laplacian_algebraic_connectivity,
    spectral_distribution_statistics,
)
from .quantum_walk import (
    run_continuous_time_quantum_search,
    compute_optimal_gamma,
)


def _deterministic_seed(base: int, *identifiers) -> int:
    payload = f"{base}-" + "-".join(str(x) for x in identifiers)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return (int(digest, 16) % (2**31))


@dataclass
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
    graph_kwargs: dict
    data_dir: str
    results_dir: str
    figures_dir: str


def _resolve_gamma(strategy: str, adjacency: sp.csr_matrix, n_nodes: int, gap_value: float) -> float:
    if strategy == "average_degree":
        return compute_optimal_gamma(adjacency)
    if strategy == "spectral_gap":
        return 1.0 / max(gap_value, 1e-6)
    if strategy == "uniform":
        return 1.0 / n_nodes
    raise ValueError(f"Unknown gamma strategy: {strategy}")


def run_single_experiment(
    config: ExperimentConfig,
    graph_type: str,
    n_nodes: int,
    instance_idx: int,
    noise_level: float,
) -> dict:
    graph_seed = _deterministic_seed(config.base_seed, graph_type, n_nodes, instance_idx)
    generator = make_generator(graph_type, **config.graph_kwargs.get(graph_type, {}))
    graph = generator.generate(n_nodes, seed=graph_seed)
    adjacency = adjacency_matrix(graph)
    laplacian = laplacian_matrix(graph)
    adj_spectrum = compute_adjacency_spectrum(adjacency)
    lap_spectrum = compute_laplacian_spectrum(laplacian)
    adj_gap = spectral_gap(adj_spectrum)
    algebraic_conn = laplacian_algebraic_connectivity(lap_spectrum)
    spectral_stats = spectral_distribution_statistics(adj_spectrum)
    gamma = _resolve_gamma(
        config.gamma_strategy, adjacency, n_nodes, adj_gap.gap
    )
    marked_seed = _deterministic_seed(config.base_seed + 7, graph_type, n_nodes, instance_idx)
    marked_vertex = int(marked_seed % n_nodes)
    t_max = float(config.t_max_factor * np.sqrt(n_nodes))
    noise_seed = _deterministic_seed(config.base_seed + 99, graph_type, n_nodes, instance_idx, int(noise_level * 1e6))
    result = run_continuous_time_quantum_search(
        adjacency=adjacency,
        gamma=gamma,
        marked_vertex=marked_vertex,
        t_max=t_max,
        n_time_steps=config.n_time_steps,
        noise_level=noise_level,
        noise_seed=noise_seed,
        use_laplacian=config.use_laplacian,
    )
    return {
        "graph_type": graph_type,
        "n_nodes": n_nodes,
        "instance_idx": instance_idx,
        "graph_seed": graph_seed,
        "noise_seed": noise_seed,
        "marked_vertex": marked_vertex,
        "gamma": gamma,
        "noise_level": noise_level,
        "spectral_gap_adjacency": adj_gap.gap,
        "spectral_gap_largest": adj_gap.largest,
        "spectral_gap_second": adj_gap.second_largest,
        "algebraic_connectivity": algebraic_conn,
        "success_probability": result.success_probability,
        "optimal_time": result.optimal_time,
        "integrated_success": result.integrated_success,
        "t_max": t_max,
        "use_laplacian": config.use_laplacian,
        **spectral_stats,
        "full_eigenvalues": adj_spectrum.eigenvalues.tolist(),
        "times": result.times.tolist(),
        "marked_probabilities": result.marked_probabilities.tolist(),
    }


def run_experiment_suite(config: ExperimentConfig, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total = (
        len(config.graph_types)
        * len(config.graph_sizes)
        * config.n_instances
        * len(config.noise_levels)
    )
    counter = 0
    for graph_type in config.graph_types:
        for n_nodes in config.graph_sizes:
            for instance_idx in range(config.n_instances):
                for noise_level in config.noise_levels:
                    record = run_single_experiment(
                        config, graph_type, n_nodes, instance_idx, noise_level
                    )
                    records.append(record)
                    counter += 1
                    print(f"[{counter}/{total}] {graph_type} n={n_nodes} inst={instance_idx} noise={noise_level}")
    summary_columns = [
        c for c in records[0].keys()
        if c not in ("full_eigenvalues", "times", "marked_probabilities")
    ]
    df = pd.DataFrame([{c: r[c] for c in summary_columns} for r in records])
    df.to_csv(output_dir / "experiment_summary.csv", index=False)
    with open(output_dir / "experiment_full_records.json", "w", encoding="utf-8") as fh:
        json.dump(records, fh)
    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2, default=str)
    return df
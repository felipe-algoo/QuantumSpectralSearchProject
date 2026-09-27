from .config import ExperimentConfig, load_config
from .graphs import make_generator, adjacency_matrix, laplacian_matrix
from .spectral import (
    SpectralDecomposition,
    SpectralGap,
    compute_adjacency_spectrum,
    compute_laplacian_spectrum,
    spectral_gap,
    laplacian_algebraic_connectivity,
)
from .quantum_walk import (
    QuantumWalkResult,
    build_search_hamiltonian,
    run_continuous_time_quantum_search,
    compute_optimal_gamma,
)
from .experiments import run_experiment_suite, run_single_experiment
from .statistics import (
    bootstrap_confidence_interval,
    spectral_efficiency_regression,
    spectral_efficiency_hypothesis_test,
    effect_size_cohens_d,
)
from .visualization import (
    plot_spectral_distribution,
    plot_search_probability_over_time,
    plot_spectral_gap_vs_efficiency,
    plot_statistical_summary,
)

__version__ = "0.1.0"
__all__ = [
    "ExperimentConfig", "load_config",
    "make_generator", "adjacency_matrix", "laplacian_matrix",
    "SpectralDecomposition", "SpectralGap",
    "compute_adjacency_spectrum", "compute_laplacian_spectrum",
    "spectral_gap", "laplacian_algebraic_connectivity",
    "QuantumWalkResult", "build_search_hamiltonian",
    "run_continuous_time_quantum_search", "compute_optimal_gamma",
    "run_experiment_suite", "run_single_experiment",
    "bootstrap_confidence_interval", "spectral_efficiency_regression",
    "spectral_efficiency_hypothesis_test", "effect_size_cohens_d",
    "plot_spectral_distribution", "plot_search_probability_over_time",
    "plot_spectral_gap_vs_efficiency", "plot_statistical_summary",
]
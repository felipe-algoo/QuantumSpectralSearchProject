from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class QuantumWalkResult:
    times: np.ndarray
    state_trajectory: np.ndarray
    marked_probabilities: np.ndarray
    success_probability: float
    optimal_time: float
    optimal_time_index: int
    integrated_success: float
    hamiltonian: sp.csr_matrix
    gamma: float
    marked_vertex: int
    noise_level: float


def build_search_hamiltonian(
    adjacency: sp.csr_matrix,
    gamma: float,
    marked_vertex: int,
    use_laplacian: bool = False,
) -> sp.csr_matrix:
    n = adjacency.shape[0]
    if not 0 <= marked_vertex < n:
        raise IndexError(f"marked_vertex {marked_vertex} out of range [0,{n})")
    if use_laplacian:
        degree = np.asarray(adjacency.sum(axis=1)).flatten()
        laplacian = sp.diags(degree, format="csr") - adjacency
        operator = -gamma * laplacian
    else:
        operator = -gamma * adjacency
    oracle_entries = np.array([-1.0], dtype=np.float64)
    oracle = sp.csr_matrix(
        (oracle_entries, (np.array([marked_vertex]), np.array([marked_vertex]))),
        shape=(n, n),
    )
    return (operator + oracle).tocsr()


def add_gaussian_hermitian_noise(
    hamiltonian: sp.csr_matrix,
    noise_level: float,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    if noise_level <= 0.0:
        return hamiltonian
    n = hamiltonian.shape[0]
    real_part = rng.standard_normal((n, n))
    symmetric = 0.5 * (real_part + real_part.T)
    spectral_norm = float(np.linalg.norm(symmetric, ord=2))
    if spectral_norm < 1e-12:
        return hamiltonian
    scaled = noise_level * symmetric / spectral_norm
    return (hamiltonian + sp.csr_matrix(scaled)).tocsr()


def uniform_initial_state(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.complex128) / np.sqrt(n)


def evolve_state(
    hamiltonian: sp.csr_matrix,
    initial_state: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    states = np.zeros((times.size, initial_state.size), dtype=np.complex128)
    complex_hamiltonian = (-1j * hamiltonian).tocsr()
    for idx, t in enumerate(times):
        states[idx] = spla.expm_multiply(complex_hamiltonian, initial_state, start=0.0, stop=float(t), num=2, endpoint=True)[-1]
    return states


def run_continuous_time_quantum_search(
    adjacency: sp.csr_matrix,
    gamma: float,
    marked_vertex: int,
    t_max: float,
    n_time_steps: int,
    noise_level: float = 0.0,
    noise_seed: int = 0,
    use_laplacian: bool = False,
) -> QuantumWalkResult:
    if t_max <= 0:
        raise ValueError(f"t_max must be positive; got {t_max}")
    if n_time_steps < 2:
        raise ValueError(f"n_time_steps must be >=2; got {n_time_steps}")
    hamiltonian = build_search_hamiltonian(
        adjacency, gamma, marked_vertex, use_laplacian
    )
    if noise_level > 0.0:
        rng = np.random.default_rng(noise_seed)
        hamiltonian = add_gaussian_hermitian_noise(hamiltonian, noise_level, rng)
    initial_state = uniform_initial_state(adjacency.shape[0])
    times = np.linspace(0.0, t_max, n_time_steps)
    trajectory = evolve_state(hamiltonian, initial_state, times)
    marked_probs = np.abs(trajectory[:, marked_vertex]) ** 2
    optimal_idx = int(np.argmax(marked_probs))
    integrated = float(np.trapz(marked_probs, times))
    return QuantumWalkResult(
        times=times,
        state_trajectory=trajectory,
        marked_probabilities=marked_probs,
        success_probability=float(marked_probs[optimal_idx]),
        optimal_time=float(times[optimal_idx]),
        optimal_time_index=optimal_idx,
        integrated_success=integrated,
        hamiltonian=hamiltonian,
        gamma=gamma,
        marked_vertex=marked_vertex,
        noise_level=noise_level,
    )


def compute_optimal_gamma(adjacency: sp.csr_matrix) -> float:
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    avg_degree = float(degrees.mean())
    return 1.0 / avg_degree if avg_degree > 0 else 1.0
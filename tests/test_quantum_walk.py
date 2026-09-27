import numpy as np
import networkx as nx
import pytest
from qss.graphs import adjacency_matrix
from qss.quantum_walk import (
    build_search_hamiltonian,
    uniform_initial_state,
    run_continuous_time_quantum_search,
    compute_optimal_gamma,
    add_gaussian_hermitian_noise,
)


def test_hamiltonian_hermitian():
    g = nx.complete_graph(8)
    A = adjacency_matrix(g)
    H = build_search_hamiltonian(A, gamma=0.125, marked_vertex=0)
    H_dense = H.toarray()
    assert np.allclose(H_dense, H_dense.T)


def test_uniform_initial_state_normalized():
    psi = uniform_initial_state(16)
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_complete_graph_search_success():
    n = 64
    g = nx.complete_graph(n)
    A = adjacency_matrix(g)
    gamma = 1.0 / n
    result = run_continuous_time_quantum_search(
        adjacency=A,
        gamma=gamma,
        marked_vertex=0,
        t_max=2.0 * np.pi * np.sqrt(n) / 2.0 + 1.0,
        n_time_steps=600,
        noise_level=0.0,
    )
    assert result.success_probability > 0.85


def test_unitary_evolution_norm_preserved():
    n = 16
    g = nx.cycle_graph(n)
    A = adjacency_matrix(g)
    result = run_continuous_time_quantum_search(
        adjacency=A, gamma=0.1, marked_vertex=0, t_max=5.0, n_time_steps=50
    )
    norms = np.linalg.norm(result.state_trajectory, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_noise_preserves_hermiticity():
    g = nx.complete_graph(10)
    A = adjacency_matrix(g)
    H = build_search_hamiltonian(A, gamma=0.1, marked_vertex=0)
    rng = np.random.default_rng(42)
    H_noisy = add_gaussian_hermitian_noise(H, noise_level=0.05, rng=rng)
    Hn = H_noisy.toarray()
    assert np.allclose(Hn, Hn.T, atol=1e-10)


def test_compute_optimal_gamma_complete():
    g = nx.complete_graph(8)
    A = adjacency_matrix(g)
    gamma = compute_optimal_gamma(A)
    assert np.isclose(gamma, 1.0 / 7.0)


def test_invalid_t_max_raises():
    with pytest.raises(ValueError):
        run_continuous_time_quantum_search(
            adjacency=adjacency_matrix(nx.path_graph(4)),
            gamma=0.1, marked_vertex=0, t_max=-1.0, n_time_steps=10,
        )
import numpy as np
import networkx as nx
from qss.graphs import adjacency_matrix, laplacian_matrix
from qss.spectral import (
    compute_adjacency_spectrum,
    compute_laplacian_spectrum,
    spectral_gap,
    laplacian_algebraic_connectivity,
    spectral_distribution_statistics,
)


def test_complete_graph_spectrum():
    g = nx.complete_graph(10)
    A = adjacency_matrix(g)
    spec = compute_adjacency_spectrum(A)
    assert np.isclose(spec.eigenvalues[0], 9.0, atol=1e-8)
    assert np.allclose(np.sort(spec.eigenvalues[1:]), -1.0)


def test_path_graph_largest_eigenvalue():
    g = nx.path_graph(20)
    A = adjacency_matrix(g)
    spec = compute_adjacency_spectrum(A)
    expected = 2.0 * np.cos(np.pi / (20 + 1))
    assert np.isclose(spec.eigenvalues[0], expected, atol=1e-6)


def test_spectral_gap_complete():
    g = nx.complete_graph(8)
    A = adjacency_matrix(g)
    spec = compute_adjacency_spectrum(A)
    gap = spectral_gap(spec)
    assert np.isclose(gap.largest, 7.0)
    assert np.isclose(gap.gap, 8.0)


def test_laplacian_algebraic_connectivity_path():
    g = nx.path_graph(10)
    L = laplacian_matrix(g)
    spec = compute_laplacian_spectrum(L)
    a_conn = laplacian_algebraic_connectivity(spec)
    expected = 2.0 * (1.0 - np.cos(np.pi / 10))
    assert np.isclose(a_conn, expected, atol=1e-6)


def test_spectral_distribution_statistics_finite():
    g = nx.cycle_graph(12)
    A = adjacency_matrix(g)
    spec = compute_adjacency_spectrum(A)
    stats = spectral_distribution_statistics(spec)
    for v in stats.values():
        assert np.isfinite(v)
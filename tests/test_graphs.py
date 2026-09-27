import pytest
import networkx as nx
from qss.graphs import (
    make_generator,
    PathGraphGenerator,
    CycleGraphGenerator,
    CompleteGraphGenerator,
    ErdosRenyiGenerator,
    BarabasiAlbertGenerator,
    adjacency_matrix,
    laplacian_matrix,
)


def test_path_graph_structure():
    g = PathGraphGenerator().generate(5, seed=0)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4


def test_cycle_graph_structure():
    g = CycleGraphGenerator().generate(6, seed=0)
    assert g.number_of_edges() == 6


def test_complete_graph_structure():
    g = CompleteGraphGenerator().generate(5, seed=0)
    assert g.number_of_edges() == 10


def test_erdos_renyi_reproducibility():
    gen = ErdosRenyiGenerator(p=0.3)
    g1 = gen.generate(20, seed=123)
    g2 = gen.generate(20, seed=123)
    assert nx.to_numpy_array(g1).tolist() == nx.to_numpy_array(g2).tolist()


def test_barabasi_albert_reproducibility():
    gen = BarabasiAlbertGenerator(m=2)
    g1 = gen.generate(20, seed=7)
    g2 = gen.generate(20, seed=7)
    assert nx.to_numpy_array(g1).tolist() == nx.to_numpy_array(g2).tolist()


def test_barabasi_albert_invalid_m():
    with pytest.raises(ValueError):
        BarabasiAlbertGenerator(m=0)


def test_adjacency_matrix_symmetric():
    g = ErdosRenyiGenerator(p=0.2).generate(15, seed=1)
    A = adjacency_matrix(g).toarray()
    assert np.allclose(A, A.T)


def test_laplacian_row_sums_zero():
    g = CompleteGraphGenerator().generate(6, seed=0)
    L = laplacian_matrix(g).toarray()
    assert np.allclose(L.sum(axis=1), 0.0)


def test_registry_unknown_type():
    with pytest.raises(KeyError):
        make_generator("nonexistent")
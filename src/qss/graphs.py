from __future__ import annotations
from typing import Protocol, Any
import numpy as np
import networkx as nx
import scipy.sparse as sp


class GraphGenerator(Protocol):
    def generate(self, n_nodes: int, seed: int) -> nx.Graph: ...


class PathGraphGenerator:
    def generate(self, n_nodes: int, seed: int = 0) -> nx.Graph:
        return nx.path_graph(n_nodes)


class CycleGraphGenerator:
    def generate(self, n_nodes: int, seed: int = 0) -> nx.Graph:
        return nx.cycle_graph(n_nodes)


class CompleteGraphGenerator:
    def generate(self, n_nodes: int, seed: int = 0) -> nx.Graph:
        return nx.complete_graph(n_nodes)


class ErdosRenyiGenerator:
    def __init__(self, p: float = 0.1):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Edge probability p must be in [0,1]; got {p}")
        self.p = float(p)

    def generate(self, n_nodes: int, seed: int = 0) -> nx.Graph:
        return nx.gnp_random_graph(n_nodes, self.p, seed=seed)


class BarabasiAlbertGenerator:
    def __init__(self, m: int = 2):
        if m < 1:
            raise ValueError(f"Preferential attachment m must be >=1; got {m}")
        self.m = int(m)

    def generate(self, n_nodes: int, seed: int = 0) -> nx.Graph:
        if n_nodes <= self.m:
            raise ValueError(
                f"n_nodes ({n_nodes}) must exceed m ({self.m}) for Barabasi-Albert"
            )
        return nx.barabasi_albert_graph(n_nodes, self.m, seed=seed)


GRAPH_REGISTRY: dict[str, type] = {
    "path": PathGraphGenerator,
    "cycle": CycleGraphGenerator,
    "complete": CompleteGraphGenerator,
    "erdos_renyi": ErdosRenyiGenerator,
    "barabasi_albert": BarabasiAlbertGenerator,
}


def make_generator(graph_type: str, **kwargs: Any) -> GraphGenerator:
    if graph_type not in GRAPH_REGISTRY:
        raise KeyError(
            f"Unknown graph type '{graph_type}'. "
            f"Available: {sorted(GRAPH_REGISTRY)}"
        )
    return GRAPH_REGISTRY[graph_type](**kwargs)


def adjacency_matrix(graph: nx.Graph, dtype: type = np.float64) -> sp.csr_matrix:
    nodelist = sorted(graph.nodes())
    return nx.adjacency_matrix(graph, nodelist=nodelist, dtype=dtype).tocsr()


def laplacian_matrix(graph: nx.Graph, dtype: type = np.float64) -> sp.csr_matrix:
    nodelist = sorted(graph.nodes())
    return nx.laplacian_matrix(graph, nodelist=nodelist, dtype=dtype).tocsr()


def graph_summary(graph: nx.Graph) -> dict[str, float | int]:
    degrees = [d for _, d in graph.degree()]
    return {
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(np.max(degrees)) if degrees else 0,
        "is_connected": bool(nx.is_connected(graph)) if graph.number_of_nodes() > 0 else False,
    }
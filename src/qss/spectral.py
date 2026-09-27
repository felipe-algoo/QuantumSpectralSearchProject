from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class SpectralDecomposition:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    matrix_type: str


@dataclass
class SpectralGap:
    largest: float
    second_largest: float
    gap: float


def _dense_eigh(matrix: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    dense = matrix.toarray().astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    order = np.argsort(eigenvalues)[::-1][:k]
    return eigenvalues[order], eigenvectors[:, order]


def _sparse_top_eigs(matrix: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    if k >= n - 1:
        return _dense_eigh(matrix, k)
    eigenvalues, eigenvectors = spla.eigsh(matrix.asfptype(), k=k, which="LA")
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def compute_adjacency_spectrum(
    adjacency: sp.csr_matrix, k: Optional[int] = None
) -> SpectralDecomposition:
    n = adjacency.shape[0]
    k_eff = n if k is None else min(k, n)
    eigenvalues, eigenvectors = (
        _dense_eigh(adjacency, k_eff) if k_eff >= n else _sparse_top_eigs(adjacency, k_eff)
    )
    return SpectralDecomposition(
        eigenvalues=eigenvalues.astype(np.float64),
        eigenvectors=eigenvectors.astype(np.float64),
        matrix_type="adjacency",
    )


def compute_laplacian_spectrum(
    laplacian: sp.csr_matrix, k: Optional[int] = None
) -> SpectralDecomposition:
    n = laplacian.shape[0]
    k_eff = n if k is None else min(k, n)
    eigenvalues, eigenvectors = (
        _dense_eigh(laplacian, k_eff) if k_eff >= n else _sparse_top_eigs(laplacian, k_eff)
    )
    return SpectralDecomposition(
        eigenvalues=eigenvalues.astype(np.float64),
        eigenvectors=eigenvectors.astype(np.float64),
        matrix_type="laplacian",
    )


def spectral_gap(spectrum: SpectralDecomposition) -> SpectralGap:
    if spectrum.eigenvalues.size < 2:
        raise ValueError("Spectral gap requires at least two eigenvalues.")
    largest = float(spectrum.eigenvalues[0])
    second_largest = float(spectrum.eigenvalues[1])
    return SpectralGap(
        largest=largest,
        second_largest=second_largest,
        gap=largest - second_largest,
    )


def laplacian_algebraic_connectivity(lap_spectrum: SpectralDecomposition) -> float:
    sorted_eigs = np.sort(lap_spectrum.eigenvalues)
    threshold = 1e-10
    non_zero = sorted_eigs[sorted_eigs > threshold]
    return float(non_zero[0]) if non_zero.size > 0 else 0.0


def spectral_distribution_statistics(spectrum: SpectralDecomposition) -> dict[str, float]:
    ev = spectrum.eigenvalues
    return {
        "eigenvalue_min": float(np.min(ev)),
        "eigenvalue_max": float(np.max(ev)),
        "eigenvalue_mean": float(np.mean(ev)),
        "eigenvalue_std": float(np.std(ev)),
        "eigenvalue_median": float(np.median(ev)),
        "spectral_radius": float(np.max(np.abs(ev))),
    }
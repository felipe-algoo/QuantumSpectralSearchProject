from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def configure_publication_style(font_size: int = 11, font_family: str = "serif") -> None:
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titlesize": font_size + 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_spectral_distribution(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> None:
    configure_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("deep", n_colors=df["graph_type"].nunique())
    for color, (graph_type, sub) in zip(palette, df.groupby("graph_type")):
        all_eigs = np.concatenate([
            np.array(ev, dtype=np.float64) for ev in sub["full_eigenvalues"].head(20)
        ])
        sns.kdeplot(all_eigs, ax=ax, label=graph_type.replace("_", " ").title(), color=color, fill=True, alpha=0.25, linewidth=1.4)
    ax.set_xlabel("Eigenvalue $\\lambda$")
    ax.set_ylabel("Spectral density (KDE)")
    ax.set_title("Adjacency Spectral Distributions by Graph Family")
    ax.legend(title="Graph family", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_search_probability_over_time(
    df: pd.DataFrame,
    output_path: Path,
    graph_type: Optional[str] = None,
    noise_level: float = 0.0,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> None:
    configure_publication_style()
    sub = df[(df["noise_level"] == noise_level)]
    if graph_type is not None:
        sub = sub[sub["graph_type"] == graph_type]
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("deep", n_colors=sub["n_nodes"].nunique())
    for color, (n_nodes, group) in zip(palette, sub.groupby("n_nodes")):
        instance = group.iloc[0]
        times = np.array(instance["times"])
        probs = np.array(instance["marked_probabilities"])
        ax.plot(times, probs, label=f"$N={n_nodes}$", color=color, linewidth=1.5)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Success probability $p_w(t)$")
    ax.set_title(f"Continuous-time quantum search dynamics (noise $\\sigma={noise_level}$)")
    ax.legend(title="Graph size", frameon=False, ncol=2)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_spectral_gap_vs_efficiency(
    df: pd.DataFrame,
    output_path: Path,
    x_column: str = "spectral_gap_adjacency",
    y_column: str = "success_probability",
    figsize: tuple[float, float] = (7.0, 4.5),
) -> None:
    configure_publication_style()
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column,
        hue="graph_type",
        style="graph_type",
        s=55,
        alpha=0.85,
        ax=ax,
        palette="deep",
    )
    x = df[x_column].to_numpy(dtype=np.float64)
    y = df[y_column].to_numpy(dtype=np.float64)
    if x.size > 1 and np.std(x) > 0:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, np.polyval(coeffs, xs), color="black", linestyle="--", linewidth=1.2, label="OLS fit")
    ax.set_xlabel("Spectral gap $\\Delta = \\lambda_1 - \\lambda_2$")
    ax.set_ylabel("Peak success probability $\\max_t p_w(t)$")
    ax.set_title("Spectral Gap vs. Quantum Search Efficiency")
    ax.legend(title="Graph family", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_statistical_summary(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple[float, float] = (8.0, 4.5),
) -> None:
    configure_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.boxplot(
        data=df, x="graph_type", y="success_probability",
        hue="noise_level", ax=axes[0], palette="deep", linewidth=0.9, fliersize=3,
    )
    axes[0].set_title("Success probability distribution")
    axes[0].set_xlabel("Graph family")
    axes[0].set_ylabel("Peak success probability")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="Noise $\\sigma$", frameon=False, fontsize=8)

    sns.boxplot(
        data=df, x="graph_type", y="optimal_time",
        hue="noise_level", ax=axes[1], palette="deep", linewidth=0.9, fliersize=3,
    )
    axes[1].set_title("Optimal hitting time")
    axes[1].set_xlabel("Graph family")
    axes[1].set_ylabel("Optimal time $t^*$")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(title="Noise $\\sigma$", frameon=False, fontsize=8)

    fig.suptitle("Statistical summary across graph families and noise levels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
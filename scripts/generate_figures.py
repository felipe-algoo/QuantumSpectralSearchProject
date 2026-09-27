from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from qss.config import load_config
from qss.visualization import (
    plot_spectral_distribution,
    plot_search_probability_over_time,
    plot_spectral_gap_vs_efficiency,
    plot_statistical_summary,
)
from qss.statistics import (
    spectral_efficiency_regression,
    spectral_efficiency_hypothesis_test,
    per_group_confidence_intervals,
    effect_size_cohens_d,
)
from qss.utils import ensure_directory, save_json


def main(results_dir: str = "results") -> int:
    results_path = Path(results_dir)
    summary_csv = results_path / "experiment_summary.csv"
    full_json = results_path / "experiment_full_records.json"

    df = pd.read_csv(summary_csv)
    with open(full_json, "r", encoding="utf-8") as fh:
        full_records = json.load(fh)
    df["full_eigenvalues"] = [r["full_eigenvalues"] for r in full_records]
    df["times"] = [r["times"] for r in full_records]
    df["marked_probabilities"] = [r["marked_probabilities"] for r in full_records]

    figures_dir = ensure_directory(results_path / "figures")

    plot_spectral_distribution(df, figures_dir / "fig1_spectral_distributions.png")
    plot_search_probability_over_time(
        df, figures_dir / "fig2_search_dynamics.png",
        graph_type="complete", noise_level=0.0,
    )
    plot_spectral_gap_vs_efficiency(
        df, figures_dir / "fig3_gap_vs_efficiency.png",
    )
    plot_statistical_summary(df, figures_dir / "fig4_statistical_summary.png")

    regression = spectral_efficiency_regression(df)
    hypothesis = spectral_efficiency_hypothesis_test(df, alternative="greater")
    ci_table = per_group_confidence_intervals(
        df, group_column="graph_type", value_column="success_probability"
    )
    ci_table.to_csv(results_path / "per_group_confidence_intervals.csv", index=False)

    save_json({
        "regression": {
            "intercept": regression.intercept,
            "slope": regression.slope,
            "r_squared": regression.r_squared,
            "std_error": regression.std_error,
            "p_value": regression.p_value,
            "n_observations": regression.n_observations,
        },
        "hypothesis_test": {
            "statistic": hypothesis.statistic,
            "p_value": hypothesis.p_value,
            "alternative": hypothesis.alternative,
            "significant": hypothesis.significant,
            "alpha": hypothesis.alpha,
        },
    }, results_path / "statistical_analysis.json")

    print(f"Figures written to {figures_dir}")
    print(f"Regression R^2 = {regression.r_squared:.4f}, slope p-value = {regression.p_value:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
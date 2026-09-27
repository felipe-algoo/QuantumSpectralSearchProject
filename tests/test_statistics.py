import numpy as np
import pandas as pd
from qss.statistics import (
    bootstrap_confidence_interval,
    spectral_efficiency_regression,
    spectral_efficiency_hypothesis_test,
    effect_size_cohens_d,
    per_group_confidence_intervals,
)


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=2.5, scale=1.0, size=500)
    ci = bootstrap_confidence_interval(data, n_resamples=2000, seed=0)
    assert ci.lower < 2.5 < ci.upper
    assert ci.lower < ci.estimate < ci.upper


def test_regression_recovers_linear_relationship():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, size=200)
    y = 1.5 * x + 2.0 + rng.normal(scale=0.1, size=200)
    df = pd.DataFrame({"gap": x, "success": y})
    res = spectral_efficiency_regression(df, predictor="gap", response="success")
    assert abs(res.slope - 1.5) < 0.1
    assert res.r_squared > 0.95
    assert res.p_value < 1e-6


def test_hypothesis_test_significant_positive_correlation():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 5, size=300)
    y = 0.4 * x + rng.normal(scale=0.05, size=300)
    df = pd.DataFrame({"gap": x, "success": y})
    ht = spectral_efficiency_hypothesis_test(df, predictor="gap", response="success", alternative="greater")
    assert ht.significant
    assert ht.statistic > 0.9


def test_cohens_d_zero_when_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert effect_size_cohens_d(a, a.copy()) == 0.0


def test_per_group_ci_returns_dataframe():
    df = pd.DataFrame({
        "graph_type": ["complete", "complete", "path", "path"],
        "success_probability": [0.9, 0.92, 0.3, 0.35],
    })
    out = per_group_confidence_intervals(df, "graph_type", "success_probability", n_resamples=500)
    assert set(out["graph_type"]) == {"complete", "path"}
    assert (out["ci_lower"] <= out["estimate"]).all()
    assert (out["estimate"] <= out["ci_upper"]).all()
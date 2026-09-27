from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


@dataclass
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    n_resamples: int
    confidence_level: float


@dataclass
class RegressionResult:
    intercept: float
    slope: float
    r_squared: float
    std_error: float
    p_value: float
    n_observations: int


@dataclass
class HypothesisTest:
    statistic: float
    p_value: float
    alternative: str
    significant: bool
    alpha: float


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        raise ValueError("Cannot bootstrap empty data.")
    estimates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(data, size=data.size, replace=True)
        estimates[i] = statistic_fn(sample)
    point_estimate = float(statistic_fn(data))
    alpha = 1.0 - confidence_level
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return BootstrapCI(
        estimate=point_estimate,
        lower=lower,
        upper=upper,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )


def spectral_efficiency_regression(
    df: pd.DataFrame,
    predictor: str = "spectral_gap_adjacency",
    response: str = "success_probability",
) -> RegressionResult:
    x = df[predictor].to_numpy(dtype=np.float64)
    y = df[response].to_numpy(dtype=np.float64)
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    return RegressionResult(
        intercept=float(model.params[0]),
        slope=float(model.params[1]),
        r_squared=float(model.rsquared),
        std_error=float(model.bse[1]),
        p_value=float(model.pvalues[1]),
        n_observations=int(model.nobs),
    )


def spectral_efficiency_hypothesis_test(
    df: pd.DataFrame,
    predictor: str = "spectral_gap_adjacency",
    response: str = "success_probability",
    alternative: str = "greater",
    alpha: float = 0.05,
) -> HypothesisTest:
    x = df[predictor].to_numpy(dtype=np.float64)
    y = df[response].to_numpy(dtype=np.float64)
    corr, p_two_sided = stats.pearsonr(x, y)
    if alternative == "greater":
        p_value = p_two_sided / 2.0 if corr > 0 else 1.0 - p_two_sided / 2.0
    elif alternative == "less":
        p_value = p_two_sided / 2.0 if corr < 0 else 1.0 - p_two_sided / 2.0
    else:
        p_value = p_two_sided
    return HypothesisTest(
        statistic=float(corr),
        p_value=float(p_value),
        alternative=alternative,
        significant=bool(p_value < alpha),
        alpha=alpha,
    )


def effect_size_cohens_d(
    group_a: np.ndarray, group_b: np.ndarray
) -> float:
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    pooled_std = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def per_group_confidence_intervals(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for group_value, sub in df.groupby(group_column):
        ci = bootstrap_confidence_interval(
            sub[value_column].to_numpy(),
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed + hash(str(group_value)) % 1000,
        )
        rows.append({
            group_column: group_value,
            "estimate": ci.estimate,
            "ci_lower": ci.lower,
            "ci_upper": ci.upper,
            "n": sub.shape[0],
        })
    return pd.DataFrame(rows)
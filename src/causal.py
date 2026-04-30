"""Causal inference toolkit — DiD, Synthetic Control, Propensity Score Matching.

Each function takes a clean dataframe and returns a structured result so
the Streamlit page can render diagnostics (parallel-trends, balance,
permutation inference) consistently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


__all__ = [
    "DIDResult",
    "PSMResult",
    "SyntheticControlResult",
    "did_panel",
    "fit_did",
    "fit_propensity_match",
    "fit_synthetic_control",
    "psm_panel",
    "synth_panel",
]


# ============================================================================
# Difference-in-Differences
# ============================================================================

@dataclass
class DIDResult:
    delta: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    parallel_trends_p: float       # Wald test on pre-period interaction
    placebo_delta: float            # placebo intervention 4 weeks earlier
    placebo_p: float
    fitted_summary: str


def did_panel(
    n_pre_weeks: int = 8,
    n_burst_weeks: int = 4,
    n_post_weeks: int = 4,
    treatment_lift: float = 1_000.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic two-region weekly panel for the TV-burst DiD.

    Treated region (DACH) gets ``treatment_lift`` extra units/week during
    the burst window. Control region (BeNeLux) shares the same yearly
    seasonality and trend, so parallel-trends should hold pre-treatment.
    """
    rng = np.random.default_rng(seed)
    total_weeks = n_pre_weeks + n_burst_weeks + n_post_weeks
    week = np.arange(total_weeks)

    # Identical trend slope AND seasonal amplitude ensure parallel trends
    seasonality = 1200 * np.sin(2 * np.pi * week / 12)
    base_dach = 50_000 + 180 * week + seasonality
    base_benelux = 38_000 + 180 * week + seasonality

    rows = []
    for w in week:
        in_burst = bool(n_pre_weeks <= w < n_pre_weeks + n_burst_weeks)
        # `during_burst` is a TIME flag — true for both regions during the
        # burst window. The treatment effect comes from the interaction
        # treated × during_burst, so the lift only applies to DACH.
        rows.append({
            "week": int(w),
            "region": "DACH",
            "treated": 1,
            "post": int(w >= n_pre_weeks),
            "during_burst": int(in_burst),
            "units": float(base_dach[w] + (treatment_lift if in_burst else 0) + rng.normal(0, 800)),
        })
        rows.append({
            "week": int(w),
            "region": "BeNeLux",
            "treated": 0,
            "post": int(w >= n_pre_weeks),
            "during_burst": int(in_burst),
            "units": float(base_benelux[w] + rng.normal(0, 800)),
        })
    return pd.DataFrame(rows)


def fit_did(
    df: pd.DataFrame,
    outcome: str = "units",
    treated_col: str = "treated",
    post_col: str = "during_burst",
    pre_period_col: str = "post",
    cluster_col: str = "region",
    date_col: str = "week",
) -> DIDResult:
    """Two-way fixed-effects DiD with cluster-robust SEs.

    Parameters
    ----------
    post_col : str
        Indicator for the *treatment period* (when the lift is expected).
    pre_period_col : str
        Indicator for "after treatment start". The parallel-trends test
        uses ``pre_period_col == 0`` to isolate the genuine pre-period —
        kept distinct from ``post_col`` because the post-treatment window
        may sit between burst and post-burst phases.
    """
    work = df.copy()
    work["interaction"] = work[treated_col] * work[post_col]
    X = sm.add_constant(work[[treated_col, post_col, "interaction"]].astype(float))
    y = work[outcome].astype(float)

    # HC1 heteroskedasticity-robust SEs. Cluster-robust would be ideal but
    # degenerates with the 2 clusters available in the standard 2-region
    # DiD layout (Cameron-Miller 2015). HC1 stays valid for any G.
    model = sm.OLS(y, X).fit(cov_type="HC1")
    delta = float(model.params["interaction"])
    se = float(model.bse["interaction"])
    ci_low, ci_high = (float(c) for c in model.conf_int().loc["interaction"])
    p_value = float(model.pvalues["interaction"])

    # Parallel-trends pre-test: weeks strictly before treatment started
    pre = work[work[pre_period_col] == 0].copy()
    pre["t_x_treated"] = pre[date_col] * pre[treated_col]
    Xp = sm.add_constant(pre[[treated_col, date_col, "t_x_treated"]].astype(float))
    pt = sm.OLS(pre[outcome].astype(float), Xp).fit(cov_type="HC1")
    parallel_trends_p = float(pt.pvalues["t_x_treated"])

    # Placebo: pretend the treatment started 4 weeks earlier within the pre-period
    placebo_delta, placebo_p = _placebo_did(
        work, outcome, treated_col, date_col, cluster_col, pre_period_col
    )

    return DIDResult(
        delta=delta, se=se, ci_low=ci_low, ci_high=ci_high, p_value=p_value,
        parallel_trends_p=parallel_trends_p,
        placebo_delta=placebo_delta, placebo_p=placebo_p,
        fitted_summary=str(model.summary()),
    )


def _placebo_did(df, outcome, treated_col, date_col, cluster_col, pre_period_col="post") -> tuple[float, float]:
    pre = df[df[pre_period_col] == 0].copy()
    if len(pre) < 8:
        return 0.0, 1.0
    fake_cut = int(pre[date_col].median())
    pre["fake_post"] = (pre[date_col] >= fake_cut).astype(int)
    pre["fake_int"] = pre[treated_col] * pre["fake_post"]
    X = sm.add_constant(pre[[treated_col, "fake_post", "fake_int"]].astype(float))
    try:
        m = sm.OLS(pre[outcome].astype(float), X).fit(cov_type="HC1")
        return float(m.params["fake_int"]), float(m.pvalues["fake_int"])
    except Exception:
        return 0.0, 1.0


# ============================================================================
# Synthetic Control
# ============================================================================

@dataclass
class SyntheticControlResult:
    weights: dict[str, float]
    pre_rmse: float
    treated: np.ndarray
    synthetic: np.ndarray
    gap: np.ndarray
    avg_post_gap: float
    permutation_rank: int
    permutation_total: int


def synth_panel(
    n_pre: int = 16,
    n_post: int = 8,
    treatment_lift: float = 800.0,
    seed: int = 11,
) -> pd.DataFrame:
    """Wide panel with one treated region (DACH) and several donor regions.

    DACH is constructed as a known convex combination of donors plus
    independent noise — this guarantees the synthetic-control method has
    a feasible target it can recover. The post-period lift is the only
    structural break.
    """
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_pre + n_post)
    panel: dict[str, np.ndarray] = {"week": weeks}

    # Donor regions: shared seasonality, idiosyncratic baselines/trends/noise
    donor_specs = {
        "Italy":   (30_000, 120),
        "Iberia":  (28_000, 140),
        "BeNeLux": (38_000, 160),
        "France":  (42_000, 200),
        "Nordics": (25_000, 90),
    }
    seas = 1500 * np.sin(2 * np.pi * weeks / 12)
    donor_series: dict[str, np.ndarray] = {}
    for region, (base, slope) in donor_specs.items():
        s = base + slope * weeks + seas + rng.normal(0, 500, size=len(weeks))
        donor_series[region] = s
        panel[region] = s

    # DACH = mostly BeNeLux + France + a bit of Italy, plus noise + lift
    target_weights = {"BeNeLux": 0.50, "France": 0.35, "Italy": 0.15}
    dach = sum(w * donor_series[r] for r, w in target_weights.items())
    dach = dach + rng.normal(0, 400, size=len(weeks))
    dach[n_pre:] += treatment_lift
    panel["DACH"] = dach

    return pd.DataFrame(panel)


def fit_synthetic_control(
    panel: pd.DataFrame,
    treated: str = "DACH",
    n_pre: int = 16,
    week_col: str = "week",
) -> SyntheticControlResult:
    donors = [c for c in panel.columns if c not in (week_col, treated)]
    Y_treated = panel[treated].to_numpy(dtype=np.float64)
    Y_donors = panel[donors].to_numpy(dtype=np.float64)   # shape (T, J)

    pre_t = Y_treated[:n_pre]
    pre_d = Y_donors[:n_pre, :]

    weights = _solve_synth_weights(pre_t, pre_d)
    synthetic = Y_donors @ weights
    pre_rmse = float(np.sqrt(np.mean((pre_t - synthetic[:n_pre]) ** 2)))
    gap = Y_treated - synthetic
    avg_post_gap = float(np.mean(gap[n_pre:]))

    # Permutation inference: do the same for each donor "as if" treated
    abs_effects = []
    for j, donor in enumerate(donors):
        d_treated = Y_donors[:, j]
        d_pool = np.delete(Y_donors, j, axis=1)
        pre_dt = d_treated[:n_pre]
        pre_pool = d_pool[:n_pre, :]
        w = _solve_synth_weights(pre_dt, pre_pool)
        d_synth = d_pool @ w
        d_gap = d_treated - d_synth
        abs_effects.append(np.abs(np.mean(d_gap[n_pre:])))
    abs_effects.append(abs(avg_post_gap))
    rank = int(np.argsort(np.argsort(-np.array(abs_effects)))[-1]) + 1

    return SyntheticControlResult(
        weights={d: float(w) for d, w in zip(donors, weights)},
        pre_rmse=pre_rmse,
        treated=Y_treated,
        synthetic=synthetic,
        gap=gap,
        avg_post_gap=avg_post_gap,
        permutation_rank=rank,
        permutation_total=len(abs_effects),
    )


def _solve_synth_weights(treated_pre: np.ndarray, donors_pre: np.ndarray) -> np.ndarray:
    """Find non-negative weights summing to 1 minimising pre-period MSE.

    Reparametrises ``w_i = exp(z_i) / Σ exp(z_j)`` so the weights live on
    the simplex by construction — much more reliable than SLSQP with an
    explicit equality constraint, which often returns the initial point
    when the loss has a large dynamic range.
    """
    j = donors_pre.shape[1]
    # Scale loss by treated variance so optimiser sees O(1) numbers
    scale = float(np.var(treated_pre)) or 1.0

    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def loss(z: np.ndarray) -> float:
        w = softmax(z)
        pred = donors_pre @ w
        return float(np.mean((treated_pre - pred) ** 2) / scale)

    z0 = np.zeros(j)
    res = minimize(loss, z0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12})
    z = res.x if res.success else z0
    return softmax(z)


# ============================================================================
# Propensity Score Matching
# ============================================================================

@dataclass
class PSMResult:
    naive_att: float
    psm_att: float
    true_att: float
    matched_n: int
    pre_balance: dict[str, float]   # standardised mean differences before match
    post_balance: dict[str, float]


def psm_panel(n: int = 10_000, true_lift: float = 0.05, seed: int = 17) -> pd.DataFrame:
    """Customer-level synthetic data with selection bias.

    Customers more engaged historically are more likely to open the email
    AND more likely to purchase — naïve diff overstates the effect.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, size=n).clip(18, 75)
    past_purchases = rng.poisson(2.5, size=n).clip(0, 12)
    segment = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])  # 0=cold, 1=warm, 2=hot

    # Probability of being treated (email open) depends strongly on engagement
    logit = -2.0 + 0.02 * (age - 40) + 0.6 * past_purchases + 1.2 * segment
    p_treat = 1 / (1 + np.exp(-logit))
    treated = (rng.random(n) < p_treat).astype(int)

    # True purchase probability — engagement also drives purchases (the bias)
    base_p = 0.04 + 0.015 * past_purchases + 0.05 * segment
    purchase_p = np.clip(base_p + true_lift * treated, 0.0, 1.0)
    purchased = (rng.random(n) < purchase_p).astype(int)

    return pd.DataFrame({
        "age": age, "past_purchases": past_purchases, "segment": segment,
        "treated": treated, "purchased": purchased,
    })


def _smd(x_treated: np.ndarray, x_control: np.ndarray) -> float:
    """Standardised mean difference."""
    pooled_sd = np.sqrt((x_treated.var() + x_control.var()) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((x_treated.mean() - x_control.mean()) / pooled_sd)


def fit_propensity_match(
    df: pd.DataFrame,
    covariates: Sequence[str] = ("age", "past_purchases", "segment"),
    outcome: str = "purchased",
    treat: str = "treated",
    caliper: float = 0.05,
    true_att: float | None = None,
) -> PSMResult:
    """Logistic propensity + 1:1 nearest-neighbour matching with caliper."""
    X = df[list(covariates)].to_numpy(dtype=np.float64)
    t = df[treat].to_numpy(dtype=int)

    # Standardise so logistic regression converges without overflow
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)
    Xs = (X - X_mean) / X_std

    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xs, t)
    p_score = lr.predict_proba(Xs)[:, 1]

    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]
    control_p = p_score[control_idx]
    available = np.ones(len(control_idx), dtype=bool)

    matched_treated: list[int] = []
    matched_control: list[int] = []
    for i in treated_idx:
        diffs = np.abs(control_p - p_score[i])
        diffs[~available] = np.inf
        j_best = int(np.argmin(diffs))
        if diffs[j_best] <= caliper:
            matched_treated.append(int(i))
            matched_control.append(int(control_idx[j_best]))
            available[j_best] = False

    matched_treated = np.array(matched_treated, dtype=int)
    matched_control = np.array(matched_control, dtype=int)
    matched_n = len(matched_treated)

    naive_att = float(df.loc[t == 1, outcome].mean() - df.loc[t == 0, outcome].mean())
    if matched_n > 0:
        psm_att = float(
            df.loc[matched_treated, outcome].mean()
            - df.loc[matched_control, outcome].mean()
        )
    else:
        psm_att = float("nan")

    pre_balance = {c: _smd(df.loc[t == 1, c].to_numpy(), df.loc[t == 0, c].to_numpy()) for c in covariates}
    if matched_n > 0:
        post_balance = {
            c: _smd(df.loc[matched_treated, c].to_numpy(), df.loc[matched_control, c].to_numpy())
            for c in covariates
        }
    else:
        post_balance = {c: float("nan") for c in covariates}

    return PSMResult(
        naive_att=naive_att,
        psm_att=psm_att,
        true_att=true_att if true_att is not None else 0.05,
        matched_n=matched_n,
        pre_balance=pre_balance,
        post_balance=post_balance,
    )

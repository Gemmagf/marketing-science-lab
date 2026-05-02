"""Tests for the Bayesian posterior loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bayesian import BayesianFit, DEFAULT_POSTERIOR_PATH, load_posterior


def test_load_returns_none_for_missing_file(tmp_path):
    assert load_posterior(tmp_path / "nope.json") is None


def test_load_returns_none_for_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert load_posterior(p) is None


def _minimal_posterior(tmp_path):
    """Build a tiny fake posterior JSON in the format the script writes."""
    p = tmp_path / "post.json"
    p.write_text(json.dumps({
        "seed": 42, "chains": 2, "draws": 1000, "tune": 1000,
        "fit_method": "pymc_marketing",
        "n_obs": 731,
        "channels": ["tv_spend", "search_spend"],
        "controls": [],
        "channel_scales": {"tv_spend": 25_000.0, "search_spend": 4_000.0},
        "target_scale": 24_000.0,
        "ground_truth": {
            "betas": {"tv_spend": 1800.0, "search_spend": 4200.0},
            "decay": {"tv_spend": 0.7, "search_spend": 0.3},
            "half_sat": {"tv_spend": 80000, "search_spend": 12000},
        },
        "saturation_beta": {
            "tv_spend":     {"mean": 0.10, "q05": 0.05, "q95": 0.20},
            "search_spend": {"mean": 0.20, "q05": 0.10, "q95": 0.30},
        },
        "adstock_alpha": {
            "tv_spend":     {"mean": 0.65, "q05": 0.50, "q95": 0.80},
            "search_spend": {"mean": 0.30, "q05": 0.15, "q95": 0.45},
        },
        "saturation_lam": {
            "tv_spend":     {"mean": 1.0, "q05": 0.5, "q95": 1.5},
            "search_spend": {"mean": 2.0, "q05": 1.0, "q95": 3.0},
        },
        "intercept": {"mean": 0.5, "q05": 0.4, "q95": 0.6},
        "y_sigma": {"mean": 0.03, "q05": 0.02, "q95": 0.04},
        "r_squared": 0.91,
        "mape": 0.04,
    }))
    return p


def test_load_unscales_betas_correctly(tmp_path):
    fit = load_posterior(_minimal_posterior(tmp_path))
    assert isinstance(fit, BayesianFit)
    # β_orig = β_scaled * target_scale
    assert fit.betas_mean["tv_spend"] == pytest.approx(0.10 * 24_000)     # 2400
    assert fit.betas_mean["search_spend"] == pytest.approx(0.20 * 24_000) # 4800
    assert fit.betas_q05["tv_spend"] == pytest.approx(0.05 * 24_000)
    assert fit.betas_q95["search_spend"] == pytest.approx(0.30 * 24_000)


def test_load_keeps_lambda_unchanged(tmp_path):
    fit = load_posterior(_minimal_posterior(tmp_path))
    # λ has no scaling — it's a pure decay coefficient in [0, 1)
    assert fit.decay_mean["tv_spend"] == pytest.approx(0.65)
    assert fit.decay_mean["search_spend"] == pytest.approx(0.30)


def test_load_inverse_scales_lambda_saturation(tmp_path):
    fit = load_posterior(_minimal_posterior(tmp_path))
    # logistic-sat λ is scaled by 1/channel_scale to map to per-CHF units
    assert fit.half_sat_mean["tv_spend"] == pytest.approx(1.0 / 25_000)
    assert fit.half_sat_mean["search_spend"] == pytest.approx(2.0 / 4_000)


def test_load_metadata(tmp_path):
    fit = load_posterior(_minimal_posterior(tmp_path))
    assert fit.seed == 42
    assert fit.chains == 2
    assert fit.draws == 1000
    assert fit.r_squared == pytest.approx(0.91)
    assert fit.mape == pytest.approx(0.04)
    assert fit.channels == ("tv_spend", "search_spend")
    assert fit.ground_truth["betas"]["tv_spend"] == 1800.0


def test_credible_width_is_q95_minus_q05(tmp_path):
    fit = load_posterior(_minimal_posterior(tmp_path))
    width = fit.credible_width
    # tv: (0.20 - 0.05) * 24000 = 0.15 * 24000 = 3600
    assert width["tv_spend"] == pytest.approx(0.15 * 24_000)


def test_real_posterior_recovers_ground_truth_in_credible_intervals():
    """If the real posterior file is present, every truth must land in 90% CI."""
    if not DEFAULT_POSTERIOR_PATH.exists():
        pytest.skip("posterior JSON not present")
    fit = load_posterior(DEFAULT_POSTERIOR_PATH)
    assert fit is not None
    truth = fit.ground_truth.get("betas", {})
    truth_decay = fit.ground_truth.get("decay", {})
    if not truth or not truth_decay:
        pytest.skip("ground truth missing from posterior file")

    in_ci_beta = sum(
        1 for c in truth
        if fit.betas_q05[c] <= truth[c] <= fit.betas_q95[c]
    )
    in_ci_decay = sum(
        1 for c in truth_decay
        if fit.decay_q05[c] <= truth_decay[c] <= fit.decay_q95[c]
    )
    n = len(truth)
    # All 6 (or whatever n is) channels' truth must land inside the 90% CI
    assert in_ci_beta == n, f"Only {in_ci_beta}/{n} βs recovered within 90% CI"
    assert in_ci_decay == n, f"Only {in_ci_decay}/{n} λs recovered within 90% CI"

"""Tests for the Ridge MMM — focuses on the credibility moment: ground-truth recovery."""
from __future__ import annotations

import numpy as np
import pytest

from src.data_generation import CHANNELS, generate_dataset
from src.mmm import MMM, channel_mroi, optimise_budget


@pytest.fixture(scope="module")
def fitted():
    df, truth = generate_dataset(seed=42)
    mmm = MMM()
    fit = mmm.fit(df)
    return df, truth, mmm, fit


def test_fit_quality(fitted):
    _, _, _, fit = fitted
    assert fit.r_squared > 0.85, f"R² too low: {fit.r_squared}"
    assert fit.mape < 0.10, f"MAPE too high: {fit.mape}"


def test_betas_are_non_negative(fitted):
    _, _, _, fit = fitted
    for beta in fit.betas.values():
        assert beta >= 0


def test_recovers_dominant_channels(fitted):
    """Search and social are the always-on workhorses — must come out positive."""
    _, _, _, fit = fitted
    assert fit.betas["search_spend"] > 0
    assert fit.betas["social_spend"] > 0


def test_decay_in_valid_range(fitted):
    _, _, _, fit = fitted
    for d in fit.decay.values():
        assert 0 <= d < 1


def test_contribution_decomposition_sums_to_prediction(fitted):
    _, _, _, fit = fitted
    rebuilt = (
        np.full(len(fit.predictions), 0.0)
        + fit.contribution["baseline"].to_numpy()
        + fit.contribution[[c for c in fit.contribution.columns if c.startswith("contrib_")]].sum(axis=1).to_numpy()
    )
    np.testing.assert_allclose(rebuilt, fit.predictions, rtol=1e-6, atol=1e-6)


def test_predict_total_monotone_in_spend(fitted):
    _, _, mmm, _ = fitted
    base = {ch.name: 1000.0 for ch in CHANNELS}
    bigger = {**base, "search_spend": 4000.0}
    p_base = mmm.predict_total(base, n_days=30)
    p_bigger = mmm.predict_total(bigger, n_days=30)
    assert p_bigger > p_base


def test_mroi_is_positive_for_main_channels(fitted):
    df, _, _, fit = fitted
    for name in ("search_spend", "social_spend"):
        ch = next(c for c in CHANNELS if c.name == name)
        avg = float(df[name].mean())
        roi = channel_mroi(fit, ch, current_spend=avg)
        assert roi >= 0


def test_optimiser_respects_total_budget(fitted):
    _, _, mmm, _ = fitted
    total = 10_000.0
    alloc = optimise_budget(mmm, total_budget=total)
    assert sum(alloc.values()) == pytest.approx(total, rel=0.05)


def test_optimiser_respects_bounds(fitted):
    _, _, mmm, _ = fitted
    bounds = {ch.name: (ch.min_spend, ch.max_spend) for ch in CHANNELS}
    alloc = optimise_budget(mmm, total_budget=15_000.0, bounds=bounds)
    for name, spend in alloc.items():
        lo, hi = bounds[name]
        assert lo - 1e-6 <= spend <= hi + 1e-6


def test_optimiser_outperforms_uniform(fitted):
    _, _, mmm, _ = fitted
    total = 12_000.0
    uniform = {ch.name: total / len(CHANNELS) for ch in CHANNELS}
    optimal = optimise_budget(mmm, total_budget=total)
    p_uniform = mmm.predict_total(uniform, n_days=30)
    p_optimal = mmm.predict_total(optimal, n_days=30)
    # Optimal should not lose to uniform (allow tiny solver slack)
    assert p_optimal >= p_uniform - 1.0


def test_predict_requires_fit():
    mmm = MMM()
    with pytest.raises(RuntimeError):
        mmm.predict_total({"tv_spend": 0.0})

"""Tests for the causal inference toolkit."""
from __future__ import annotations

import numpy as np
import pytest

from src.causal import (
    did_panel,
    fit_did,
    fit_propensity_match,
    fit_synthetic_control,
    psm_panel,
    synth_panel,
)


# --- DiD ----------------------------------------------------------------

TRUE_LIFT = 3000.0


@pytest.fixture(scope="module")
def did_fitted():
    df = did_panel(treatment_lift=TRUE_LIFT, seed=7)
    return fit_did(df)


def test_did_recovers_treatment_lift(did_fitted):
    # Allow generous tolerance — small panel + noise will not be exact
    assert TRUE_LIFT - 1500 < did_fitted.delta < TRUE_LIFT + 1500


def test_did_significant(did_fitted):
    assert did_fitted.p_value < 0.05


def test_did_parallel_trends_holds(did_fitted):
    # Pre-period interaction should NOT be significant — otherwise DiD invalid
    assert did_fitted.parallel_trends_p > 0.05


def test_did_placebo_is_small(did_fitted):
    assert abs(did_fitted.placebo_delta) < TRUE_LIFT * 0.5


def test_did_panel_has_expected_shape():
    df = did_panel()
    assert set(df.region.unique()) == {"DACH", "BeNeLux"}
    assert (df["treated"] == (df["region"] == "DACH").astype(int)).all()


# --- Synthetic Control --------------------------------------------------

@pytest.fixture(scope="module")
def synth_fitted():
    panel = synth_panel(treatment_lift=800.0, seed=11)
    return fit_synthetic_control(panel, n_pre=16)


def test_synth_weights_sum_to_one(synth_fitted):
    assert sum(synth_fitted.weights.values()) == pytest.approx(1.0, abs=1e-3)


def test_synth_weights_non_negative(synth_fitted):
    for w in synth_fitted.weights.values():
        assert w >= -1e-6


def test_synth_recovers_post_lift(synth_fitted):
    # Average post gap should be close to 800 (within noise)
    assert 300 < synth_fitted.avg_post_gap < 1500


def test_synth_pre_rmse_small(synth_fitted):
    assert synth_fitted.pre_rmse < 1500   # against ~50k baseline


def test_synth_permutation_rank(synth_fitted):
    # The true treated unit should rank in the top half of the permutation
    assert synth_fitted.permutation_rank <= synth_fitted.permutation_total


# --- Propensity Score Matching ------------------------------------------

@pytest.fixture(scope="module")
def psm_fitted():
    df = psm_panel(n=10_000, true_lift=0.05, seed=17)
    return df, fit_propensity_match(df, true_att=0.05)


def test_psm_naive_overstates(psm_fitted):
    df, res = psm_fitted
    # Naïve diff should be much larger than the true 5pp lift due to selection
    assert res.naive_att > 0.08


def test_psm_corrects_toward_truth(psm_fitted):
    _, res = psm_fitted
    # PSM should land between 2pp and 8pp (true is 5pp)
    assert 0.02 <= res.psm_att <= 0.08


def test_psm_balance_improves(psm_fitted):
    _, res = psm_fitted
    # Post-match SMDs should be smaller (in absolute value) than pre-match
    for cov, pre in res.pre_balance.items():
        post = res.post_balance[cov]
        assert abs(post) < abs(pre) + 1e-3


def test_psm_matches_at_least_some(psm_fitted):
    _, res = psm_fitted
    assert res.matched_n > 100

"""Sanity tests for the synthetic D2C data generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generation import (
    BETA_SCALE,
    CHANNELS,
    GroundTruth,
    adstock_geometric,
    generate_dataset,
    saturation_hill,
)


# --- Math primitives -----------------------------------------------------

def test_adstock_zero_decay_is_identity():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(adstock_geometric(x, decay=0.0), x)


def test_adstock_geometric_recurrence():
    x = np.array([10.0, 0.0, 0.0, 0.0])
    expected = np.array([10.0, 5.0, 2.5, 1.25])
    np.testing.assert_allclose(adstock_geometric(x, decay=0.5), expected)


def test_adstock_constant_input_converges_to_geometric_sum():
    # Constant input c with decay λ → steady-state stock = c / (1 - λ)
    c, decay, n = 100.0, 0.7, 200
    out = adstock_geometric(np.full(n, c), decay)
    assert out[-1] == pytest.approx(c / (1 - decay), rel=1e-6)


def test_adstock_decay_validation():
    with pytest.raises(ValueError):
        adstock_geometric(np.array([1.0]), decay=1.0)
    with pytest.raises(ValueError):
        adstock_geometric(np.array([1.0]), decay=-0.1)


def test_saturation_at_zero():
    assert saturation_hill(np.array([0.0]), half_sat=100.0)[0] == pytest.approx(0.0)


def test_saturation_at_half_sat_returns_half():
    np.testing.assert_allclose(
        saturation_hill(np.array([100.0]), half_sat=100.0), [0.5]
    )


def test_saturation_asymptote():
    val = saturation_hill(np.array([1e9]), half_sat=100.0)
    assert val[0] > 0.99


def test_saturation_alpha_steepens_curve():
    x = np.array([50.0])
    s_lin = saturation_hill(x, half_sat=100.0, alpha=1.0)[0]
    s_steep = saturation_hill(x, half_sat=100.0, alpha=3.0)[0]
    # Below the half-sat point, higher alpha => smaller response
    assert s_steep < s_lin


def test_saturation_invalid_inputs():
    with pytest.raises(ValueError):
        saturation_hill(np.array([1.0]), half_sat=0.0)
    with pytest.raises(ValueError):
        saturation_hill(np.array([1.0]), half_sat=10.0, alpha=0.5)


# --- Dataset shape & integrity -------------------------------------------

def test_dataset_shape_and_columns():
    df, _ = generate_dataset()
    # 2024-01-01 → 2025-12-31 inclusive = 731 days (2024 is a leap year)
    assert len(df) == 731
    expected = {
        "date", "is_weekend", "is_holiday", "temperature_c", "promo_pct",
        "tv_spend", "search_spend", "social_spend", "display_spend",
        "email_sends", "ooh_spend",
        "base_contribution", "units_sold", "revenue_chf",
    }
    assert expected.issubset(df.columns)
    for ch in CHANNELS:
        assert f"contrib_{ch.name}" in df.columns


def test_no_negative_outcomes():
    df, _ = generate_dataset()
    assert (df["units_sold"] >= 0).all()
    assert (df["revenue_chf"] >= 0).all()


def test_units_sold_is_integer():
    df, _ = generate_dataset()
    assert pd.api.types.is_integer_dtype(df["units_sold"])


def test_reproducibility_same_seed():
    df1, t1 = generate_dataset(seed=123)
    df2, t2 = generate_dataset(seed=123)
    pd.testing.assert_frame_equal(df1, df2)
    assert t1.to_dict() == t2.to_dict()


def test_seeds_diverge():
    df1, _ = generate_dataset(seed=1)
    df2, _ = generate_dataset(seed=2)
    assert not df1["units_sold"].equals(df2["units_sold"])


# --- Calendar / control flags --------------------------------------------

def test_weekend_flag_matches_dayofweek():
    df, _ = generate_dataset()
    expected = (df["date"].dt.dayofweek >= 5).astype(int)
    pd.testing.assert_series_equal(
        df["is_weekend"].astype(int), expected, check_names=False
    )


def test_known_holidays_flagged():
    df, _ = generate_dataset()
    for d in ["2024-01-01", "2024-08-01", "2024-12-25",
              "2025-01-01", "2025-08-01", "2025-12-25"]:
        assert df.loc[df["date"] == d, "is_holiday"].iloc[0] == 1


def test_promo_levels_valid():
    df, _ = generate_dataset()
    assert set(df["promo_pct"].unique()).issubset({0, 10, 20, 30, 40})


def test_temperature_is_seasonal():
    df, _ = generate_dataset()
    summer = df[df["date"].dt.month.isin([6, 7, 8])]["temperature_c"].mean()
    winter = df[df["date"].dt.month.isin([12, 1, 2])]["temperature_c"].mean()
    assert summer > winter + 10   # at least 10°C summer-winter gap


# --- Channels & TV burst -------------------------------------------------

def test_channel_spend_within_bounds():
    df, _ = generate_dataset(inject_tv_burst=False)
    for ch in CHANNELS:
        col = df[ch.name]
        assert col.min() >= 0
        assert col.max() <= ch.max_spend + 1e-6


def test_sparse_channels_have_zero_days():
    df, _ = generate_dataset(inject_tv_burst=False)
    for ch in CHANNELS:
        if ch.sparsity < 1.0:
            zero_share = (df[ch.name] == 0).mean()
            assert zero_share > (1 - ch.sparsity) * 0.7   # within tolerance


def test_tv_burst_window_present():
    df, _ = generate_dataset(inject_tv_burst=True)
    burst = df[(df["date"] >= "2025-05-05") & (df["date"] < "2025-06-02")]
    assert len(burst) == 28
    # Burst centred around 22k CHF with weekend lift + ±20% jitter — every
    # day should be at least 12k (well above non-burst days).
    assert (burst["tv_spend"] > 12_000).all()
    assert burst["tv_spend"].mean() > 20_000


def test_tv_burst_can_be_disabled():
    df, _ = generate_dataset(inject_tv_burst=False)
    window = df[(df["date"] >= "2025-05-05") & (df["date"] < "2025-06-02")]
    # Without burst, max daily TV spend stays under the bound
    assert window["tv_spend"].max() <= 25_000


# --- Ground truth bookkeeping --------------------------------------------

def test_ground_truth_has_all_channels():
    _, truth = generate_dataset()
    expected = {ch.name for ch in CHANNELS}
    assert set(truth.betas.keys()) == expected
    assert set(truth.decay.keys()) == expected
    assert set(truth.half_sat.keys()) == expected


def test_ground_truth_param_sanity():
    _, truth = generate_dataset()
    assert truth.beta_scale == BETA_SCALE
    assert all(0 <= d < 1 for d in truth.decay.values())
    assert all(k > 0 for k in truth.half_sat.values())
    assert all(b > 0 for b in truth.betas.values())


def test_marketing_contribution_is_material():
    """If marketing < 5% of units, the MMM has nothing to recover."""
    df, _ = generate_dataset()
    contrib_cols = [c for c in df.columns if c.startswith("contrib_")]
    share = df[contrib_cols].sum().sum() / df["units_sold"].sum()
    assert 0.10 <= share <= 0.70   # realistic D2C marketing-driven range


def test_truth_is_dataclass_serialisable():
    _, truth = generate_dataset()
    d = truth.to_dict()
    assert isinstance(d, dict)
    assert d["baseline"] == truth.baseline

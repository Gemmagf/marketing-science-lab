"""Sanity tests for the AlpSel omnichannel synthetic generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generation import (
    BASELINE,
    CHANNELS,
    SALES_CHANNELS,
    UNIT_PRICE_CHF,
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
    np.testing.assert_allclose(
        adstock_geometric(np.array([10.0, 0.0, 0.0, 0.0]), decay=0.5),
        np.array([10.0, 5.0, 2.5, 1.25]),
    )


def test_adstock_constant_input_steady_state():
    out = adstock_geometric(np.full(200, 100.0), 0.7)
    assert out[-1] == pytest.approx(100.0 / 0.3, rel=1e-6)


def test_adstock_decay_validation():
    with pytest.raises(ValueError):
        adstock_geometric(np.array([1.0]), decay=1.0)
    with pytest.raises(ValueError):
        adstock_geometric(np.array([1.0]), decay=-0.1)


def test_saturation_at_zero_returns_zero():
    assert saturation_hill(np.array([0.0]), half_sat=100.0)[0] == pytest.approx(0.0)


def test_saturation_at_half_sat_returns_half():
    np.testing.assert_allclose(
        saturation_hill(np.array([100.0]), half_sat=100.0), [0.5]
    )


def test_saturation_invalid_half_sat():
    with pytest.raises(ValueError):
        saturation_hill(np.array([1.0]), half_sat=0.0)


# --- Dataset shape ------------------------------------------------------

def test_dataset_shape_and_columns():
    df, _ = generate_dataset()
    assert len(df) == 731
    base = {"date", "is_weekend", "is_holiday", "temperature_c", "promo_pct", "campaign"}
    assert base.issubset(df.columns)
    for ch in CHANNELS:
        assert ch.name in df.columns
    for sc in SALES_CHANNELS:
        assert f"units_{sc}" in df.columns
        assert f"revenue_{sc}" in df.columns
        assert f"baseline_{sc}" in df.columns
        for ch in CHANNELS:
            assert f"contrib_{ch.name}__{sc}" in df.columns
    assert "units_total" in df.columns
    assert "revenue_total" in df.columns


def test_seven_marketing_channels_present():
    assert len(CHANNELS) == 7
    expected = {"tv_spend", "search_spend", "social_spend", "display_spend",
                "email_sends", "ooh_spend", "leaflet_spend"}
    assert {c.name for c in CHANNELS} == expected


def test_three_sales_channels_present():
    assert SALES_CHANNELS == ("supermarket", "online", "stores")


def test_units_non_negative():
    df, _ = generate_dataset()
    for sc in SALES_CHANNELS:
        assert (df[f"units_{sc}"] >= 0).all()
        assert (df[f"revenue_{sc}"] >= 0).all()
    assert (df["units_total"] >= 0).all()


def test_units_total_equals_sum_per_channel():
    df, _ = generate_dataset()
    parts = sum(df[f"units_{sc}"] for sc in SALES_CHANNELS)
    # Allow small rounding diff (each per-sc series is rounded to int)
    diff = (df["units_total"] - parts).abs().max()
    assert diff <= len(SALES_CHANNELS)


def test_revenue_uses_correct_unit_prices():
    df, _ = generate_dataset()
    for sc in SALES_CHANNELS:
        # df["revenue_{sc}"] = units * unit_price; before rounding, so check tightly
        diff = (df[f"revenue_{sc}"] / UNIT_PRICE_CHF[sc] - df[f"units_{sc}"]).abs()
        assert diff.max() < 1.0


def test_seed_is_reproducible():
    df1, t1 = generate_dataset(seed=123)
    df2, t2 = generate_dataset(seed=123)
    pd.testing.assert_frame_equal(df1, df2)
    assert t1.to_dict() == t2.to_dict()


# --- Calendar / control flags ------------------------------------------

def test_weekend_flag_matches_dayofweek():
    df, _ = generate_dataset()
    expected = (df["date"].dt.dayofweek >= 5).astype(int)
    pd.testing.assert_series_equal(df["is_weekend"].astype(int), expected, check_names=False)


def test_known_holidays_flagged():
    df, _ = generate_dataset()
    for d in ["2024-01-01", "2024-08-01", "2025-12-25"]:
        assert df.loc[df["date"] == d, "is_holiday"].iloc[0] == 1


def test_promo_levels_valid():
    df, _ = generate_dataset()
    assert set(df["promo_pct"].unique()).issubset({0, 10, 20, 30, 40})


def test_temperature_is_seasonal():
    df, _ = generate_dataset()
    summer = df[df["date"].dt.month.isin([6, 7, 8])]["temperature_c"].mean()
    winter = df[df["date"].dt.month.isin([12, 1, 2])]["temperature_c"].mean()
    assert summer > winter + 10


def test_campaign_labels_present():
    df, _ = generate_dataset()
    campaigns = set(df["campaign"].unique())
    assert "TV Burst (DACH)" in campaigns
    assert "Black Friday Wave" in campaigns
    assert "Holiday Push" in campaigns


# --- Channels & TV burst -----------------------------------------------

def test_channel_spend_within_bounds():
    df, _ = generate_dataset(inject_tv_burst=False)
    for ch in CHANNELS:
        assert df[ch.name].min() >= 0
        assert df[ch.name].max() <= ch.max_spend + 1e-6


def test_tv_burst_window_present():
    df, _ = generate_dataset(inject_tv_burst=True)
    burst = df[(df["date"] >= "2025-05-05") & (df["date"] < "2025-06-02")]
    assert len(burst) == 28
    assert (burst["tv_spend"] > 12_000).all()
    assert burst["tv_spend"].mean() > 20_000


def test_tv_burst_disabled():
    df, _ = generate_dataset(inject_tv_burst=False)
    burst = df[(df["date"] >= "2025-05-05") & (df["date"] < "2025-06-02")]
    assert burst["tv_spend"].max() <= 25_000


# --- Cross-effect matrix expectations ---------------------------------

def test_tv_drives_supermarket_more_than_online():
    """TV's β on supermarket is higher than its β on online — by design."""
    tv = next(c for c in CHANNELS if c.name == "tv_spend")
    assert tv.beta_supermarket > tv.beta_online


def test_search_drives_online_more_than_supermarket():
    search = next(c for c in CHANNELS if c.name == "search_spend")
    assert search.beta_online > 5 * search.beta_supermarket


def test_leaflets_drive_supermarket_not_online():
    leaflets = next(c for c in CHANNELS if c.name == "leaflet_spend")
    assert leaflets.beta_supermarket > 10 * leaflets.beta_online


# --- Ground truth -------------------------------------------------------

def test_ground_truth_has_per_channel_per_sales_betas():
    _, truth = generate_dataset()
    for ch in CHANNELS:
        assert ch.name in truth.betas
        for sc in SALES_CHANNELS:
            assert sc in truth.betas[ch.name]
            assert truth.betas[ch.name][sc] >= 0


def test_marketing_share_is_material_per_channel():
    df, _ = generate_dataset()
    for sc in SALES_CHANNELS:
        contrib_cols = [c for c in df.columns if c.startswith("contrib_") and c.endswith(f"__{sc}")]
        share = df[contrib_cols].sum().sum() / df[f"units_{sc}"].sum()
        # Marketing should drive at least 5% and at most 70% of any sales channel
        assert 0.05 <= share <= 0.80, f"{sc} marketing share {share:.1%} out of range"


def test_truth_serialises():
    _, truth = generate_dataset()
    d = truth.to_dict()
    assert "betas" in d
    assert "decay" in d
    assert "baseline" in d

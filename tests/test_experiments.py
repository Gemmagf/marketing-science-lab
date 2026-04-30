"""Tests for the experiment design calculator."""
from __future__ import annotations

import numpy as np
import pytest

from src.experiments import (
    obrien_fleming_bounds,
    power_curve,
    sample_size_mean,
    sample_size_proportion,
)


# --- proportion tests ---------------------------------------------------

def test_proportion_basic():
    plan = sample_size_proportion(
        baseline_rate=0.10, relative_mde=0.10, alpha=0.05, power=0.8,
        daily_traffic=2000,
    )
    # Detecting 1pp lift on 10% baseline ~ thousands per arm
    assert 10_000 < plan.sample_size_per_arm < 30_000
    assert plan.duration_days == pytest.approx(plan.total_sample / 2000)
    assert plan.absolute_mde == pytest.approx(0.01)


def test_proportion_absolute_and_relative_match():
    p_a = sample_size_proportion(baseline_rate=0.20, absolute_mde=0.02)
    p_b = sample_size_proportion(baseline_rate=0.20, relative_mde=0.10)
    assert p_a.sample_size_per_arm == p_b.sample_size_per_arm


def test_proportion_smaller_mde_needs_more_samples():
    big = sample_size_proportion(baseline_rate=0.10, relative_mde=0.20)
    small = sample_size_proportion(baseline_rate=0.10, relative_mde=0.05)
    assert small.sample_size_per_arm > big.sample_size_per_arm


def test_proportion_higher_power_needs_more_samples():
    p80 = sample_size_proportion(baseline_rate=0.10, relative_mde=0.10, power=0.80)
    p95 = sample_size_proportion(baseline_rate=0.10, relative_mde=0.10, power=0.95)
    assert p95.sample_size_per_arm > p80.sample_size_per_arm


def test_proportion_input_validation():
    with pytest.raises(ValueError):
        sample_size_proportion(baseline_rate=0.10)  # neither MDE
    with pytest.raises(ValueError):
        sample_size_proportion(baseline_rate=0.10, relative_mde=0.1, absolute_mde=0.01)
    with pytest.raises(ValueError):
        sample_size_proportion(baseline_rate=1.5, relative_mde=0.1)
    with pytest.raises(ValueError):
        sample_size_proportion(baseline_rate=0.10, absolute_mde=0.95)


def test_proportion_summary_has_required_fields():
    plan = sample_size_proportion(baseline_rate=0.10, relative_mde=0.10)
    md = plan.as_markdown()
    for token in ("Sample", "Power", "Daily traffic", "duration"):
        assert token in md


# --- mean tests ---------------------------------------------------------

def test_mean_basic():
    plan = sample_size_mean(
        baseline_mean=50.0, std=20.0, absolute_mde=2.0,
        alpha=0.05, power=0.8, daily_traffic=500,
    )
    assert plan.sample_size_per_arm > 0
    assert plan.effect_size == pytest.approx(0.1)


def test_mean_validation():
    with pytest.raises(ValueError):
        sample_size_mean(baseline_mean=50.0, std=-1.0, relative_mde=0.05)
    with pytest.raises(ValueError):
        sample_size_mean(baseline_mean=50.0, std=10.0)


# --- power curves -------------------------------------------------------

def test_power_curve_monotone():
    pc = power_curve({"d=0.2": 0.2, "d=0.5": 0.5}, n_max=2000, n_points=20)
    for label, p in pc.powers.items():
        assert np.all(np.diff(p) >= -1e-6), f"power not monotone for {label}"
    # Larger effect should reach higher power for same n
    assert (pc.powers["d=0.5"] >= pc.powers["d=0.2"] - 1e-6).all()


def test_power_curve_test_validation():
    with pytest.raises(ValueError):
        power_curve({"d=0.2": 0.2}, test="bayes")


# --- sequential ---------------------------------------------------------

def test_obrien_fleming_decreasing():
    bounds = obrien_fleming_bounds(n_looks=5, alpha=0.05)
    assert bounds.shape == (5,)
    assert np.all(np.diff(bounds) <= 0)   # strict at first looks, ~equal at end


def test_obrien_fleming_validation():
    with pytest.raises(ValueError):
        obrien_fleming_bounds(n_looks=0)
    with pytest.raises(ValueError):
        obrien_fleming_bounds(alpha=0)


def test_obrien_fleming_final_look_matches_classic():
    bounds = obrien_fleming_bounds(n_looks=5, alpha=0.05)
    from scipy.stats import norm
    assert bounds[-1] == pytest.approx(norm.ppf(1 - 0.025), rel=1e-4)

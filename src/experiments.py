"""Experiment design calculator — power, MDE, sample size, sequential primer.

Wraps :mod:`statsmodels.stats.power` with a consistent return shape so
the Streamlit page can render any test type without per-test branching.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats
from statsmodels.stats.power import (
    NormalIndPower,
    TTestIndPower,
)
from statsmodels.stats.proportion import proportion_effectsize


__all__ = [
    "ExperimentPlan",
    "PowerCurve",
    "obrien_fleming_bounds",
    "power_curve",
    "sample_size_proportion",
    "sample_size_mean",
]


@dataclass
class ExperimentPlan:
    """Result of a sample-size calculation."""
    sample_size_per_arm: int
    total_sample: int
    duration_days: float
    relative_mde: float
    absolute_mde: float
    alpha: float
    power: float
    test_type: str
    effect_size: float
    summary_md: str

    def as_markdown(self) -> str:
        return self.summary_md


# --- proportions (conversion rates) --------------------------------------

def sample_size_proportion(
    baseline_rate: float,
    relative_mde: float | None = None,
    absolute_mde: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    daily_traffic: float = 1000.0,
    two_sided: bool = True,
) -> ExperimentPlan:
    """Required N per arm to detect an MDE on a conversion rate.

    Provide exactly one of ``relative_mde`` or ``absolute_mde``.
    """
    if (relative_mde is None) == (absolute_mde is None):
        raise ValueError("provide exactly one of relative_mde or absolute_mde")
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if absolute_mde is None:
        absolute_mde = baseline_rate * relative_mde
    if relative_mde is None:
        relative_mde = absolute_mde / baseline_rate

    treated_rate = baseline_rate + absolute_mde
    if not 0 < treated_rate < 1:
        raise ValueError(f"baseline + MDE = {treated_rate:.3f} outside (0, 1)")

    es = float(proportion_effectsize(treated_rate, baseline_rate))
    alternative = "two-sided" if two_sided else "larger"
    n = NormalIndPower().solve_power(
        effect_size=abs(es), alpha=alpha, power=power, alternative=alternative
    )
    n_per_arm = int(np.ceil(n))
    total = 2 * n_per_arm
    duration = total / daily_traffic if daily_traffic > 0 else float("inf")

    summary = (
        f"## Experiment plan — proportion test\n\n"
        f"- **Baseline**: {baseline_rate:.2%}\n"
        f"- **Detect**: ±{absolute_mde:.2%} absolute "
        f"({relative_mde:+.0%} relative)\n"
        f"- **α**: {alpha} ({'two-sided' if two_sided else 'one-sided'})\n"
        f"- **Power**: {power:.0%}\n"
        f"- **Sample**: {n_per_arm:,} per arm "
        f"(total {total:,})\n"
        f"- **Daily traffic**: {daily_traffic:,.0f}\n"
        f"- **Estimated duration**: {duration:.1f} days\n"
    )

    return ExperimentPlan(
        sample_size_per_arm=n_per_arm,
        total_sample=total,
        duration_days=float(duration),
        relative_mde=float(relative_mde),
        absolute_mde=float(absolute_mde),
        alpha=alpha, power=power,
        test_type="proportion",
        effect_size=es,
        summary_md=summary,
    )


# --- means (ARPU, AOV) ---------------------------------------------------

def sample_size_mean(
    baseline_mean: float,
    std: float,
    relative_mde: float | None = None,
    absolute_mde: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    daily_traffic: float = 1000.0,
    two_sided: bool = True,
) -> ExperimentPlan:
    """Required N per arm to detect an MDE on a mean (e.g. AOV)."""
    if (relative_mde is None) == (absolute_mde is None):
        raise ValueError("provide exactly one of relative_mde or absolute_mde")
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    if absolute_mde is None:
        absolute_mde = baseline_mean * relative_mde
    if relative_mde is None and baseline_mean != 0:
        relative_mde = absolute_mde / baseline_mean

    cohen_d = absolute_mde / std
    alternative = "two-sided" if two_sided else "larger"
    n = TTestIndPower().solve_power(
        effect_size=abs(cohen_d), alpha=alpha, power=power, alternative=alternative
    )
    n_per_arm = int(np.ceil(n))
    total = 2 * n_per_arm
    duration = total / daily_traffic if daily_traffic > 0 else float("inf")

    summary = (
        f"## Experiment plan — mean test\n\n"
        f"- **Baseline mean**: {baseline_mean:.2f}\n"
        f"- **Std**: {std:.2f}\n"
        f"- **Detect**: ±{absolute_mde:.3f} (Cohen's d = {cohen_d:.3f})\n"
        f"- **α**: {alpha} ({'two-sided' if two_sided else 'one-sided'})\n"
        f"- **Power**: {power:.0%}\n"
        f"- **Sample**: {n_per_arm:,} per arm "
        f"(total {total:,})\n"
        f"- **Daily traffic**: {daily_traffic:,.0f}\n"
        f"- **Estimated duration**: {duration:.1f} days\n"
    )

    return ExperimentPlan(
        sample_size_per_arm=n_per_arm,
        total_sample=total,
        duration_days=float(duration),
        relative_mde=float(relative_mde) if relative_mde is not None else float("nan"),
        absolute_mde=float(absolute_mde),
        alpha=alpha, power=power,
        test_type="mean",
        effect_size=float(cohen_d),
        summary_md=summary,
    )


# --- power curves --------------------------------------------------------

@dataclass
class PowerCurve:
    sample_sizes: np.ndarray
    powers: dict[str, np.ndarray]    # label -> power array


def power_curve(
    effect_sizes: dict[str, float],
    n_max: int = 5000,
    n_points: int = 50,
    alpha: float = 0.05,
    test: str = "proportion",
) -> PowerCurve:
    """Power vs sample size for several effect sizes."""
    if test not in ("proportion", "mean"):
        raise ValueError(f"test must be 'proportion' or 'mean', got {test!r}")
    sizes = np.linspace(50, n_max, n_points).astype(int)
    powers: dict[str, np.ndarray] = {}
    engine = NormalIndPower() if test == "proportion" else TTestIndPower()
    for label, es in effect_sizes.items():
        ps = np.array([
            engine.solve_power(effect_size=abs(es), nobs1=int(n), alpha=alpha)
            for n in sizes
        ])
        powers[label] = ps
    return PowerCurve(sample_sizes=sizes, powers=powers)


# --- sequential testing primer ------------------------------------------

def obrien_fleming_bounds(
    n_looks: int = 5,
    alpha: float = 0.05,
) -> np.ndarray:
    """O'Brien-Fleming alpha-spending boundaries (z-scale).

    Returns the per-look critical z-values that maintain overall type-I
    error at ``alpha`` under sequential peeking. Useful as an explainer
    of why you cannot eyeball a t-test as data accumulates.
    """
    if n_looks < 1:
        raise ValueError("n_looks must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    looks = np.arange(1, n_looks + 1)
    info_fraction = looks / n_looks
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    # Standard O'Brien-Fleming form
    return z_alpha / np.sqrt(info_fraction)

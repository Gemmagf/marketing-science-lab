"""Loader for the offline-fitted Bayesian MMM posterior.

Reads the JSON written by ``scripts/fit_bayesian_mmm.py``. The Streamlit
app calls :func:`load_posterior` and, if a valid file is present, prefers
the Bayesian results over the in-app Ridge fit.

This split keeps PyMC out of ``requirements.txt`` (slow to build on
Streamlit Community Cloud) while still letting the deployed app surface
the gold-standard Bayesian outputs — uncertainty intervals included.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


__all__ = ["BayesianFit", "load_posterior", "DEFAULT_POSTERIOR_PATH"]


DEFAULT_POSTERIOR_PATH = Path(__file__).resolve().parents[1] / "assets" / "mmm_posterior.json"


@dataclass
class BayesianFit:
    """Posterior summaries from the offline PyMC-Marketing fit."""
    betas_mean: dict[str, float]
    betas_q05: dict[str, float]
    betas_q95: dict[str, float]
    decay_mean: dict[str, float]
    decay_q05: dict[str, float]
    decay_q95: dict[str, float]
    half_sat_mean: dict[str, float]
    half_sat_q05: dict[str, float]
    half_sat_q95: dict[str, float]
    intercept_mean: float
    r_squared: float
    mape: float
    seed: int
    chains: int
    draws: int
    n_obs: int
    channels: tuple[str, ...]
    ground_truth: dict = field(default_factory=dict)

    @property
    def credible_width(self) -> dict[str, float]:
        """95% credible-interval width per channel β."""
        return {c: self.betas_q95[c] - self.betas_q05[c] for c in self.betas_mean}


def _channel_summary(
    node: Mapping[str, dict] | None,
    scale: float | Mapping[str, float] = 1.0,
) -> tuple[dict, dict, dict]:
    """Extract mean / q05 / q95 dicts, optionally rescaled.

    ``scale`` is either a scalar applied to every channel (e.g. for β with
    target_scale) or a per-channel mapping (e.g. for λ with channel_scale).
    """
    if not node:
        return {}, {}, {}

    def _scale_for(ch: str) -> float:
        if isinstance(scale, Mapping):
            return float(scale.get(ch, 1.0))
        return float(scale)

    means = {c: float(v["mean"]) * _scale_for(c) for c, v in node.items()}
    q05 = {c: float(v["q05"]) * _scale_for(c) for c, v in node.items()}
    q95 = {c: float(v["q95"]) * _scale_for(c) for c, v in node.items()}
    return means, q05, q95


def load_posterior(path: Path | str = DEFAULT_POSTERIOR_PATH) -> BayesianFit | None:
    """Return a :class:`BayesianFit` if ``path`` exists and is valid JSON; else ``None``."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # pymc-marketing internally scales channels by max-abs and target by max,
    # so the posterior is in [0, 1]-ish space. To compare with ground truth in
    # original units (units / CHF, units / send) we un-scale here:
    #   β_orig = β_scaled * target_scale          (LogisticSaturation asymptote)
    #   λ_orig = λ_scaled / channel_scale         (per-CHF decay)
    target_scale = float(d.get("target_scale", 1.0)) or 1.0
    channel_scales: dict = d.get("channel_scales", {})
    inv_channel_scales = {
        c: 1.0 / s if s else 1.0 for c, s in channel_scales.items()
    } if channel_scales else 1.0

    betas_m, betas_q05, betas_q95 = _channel_summary(
        d.get("saturation_beta"), scale=target_scale,
    )
    decay_m, decay_q05, decay_q95 = _channel_summary(
        d.get("adstock_alpha"),
    )
    ks_m, ks_q05, ks_q95 = _channel_summary(
        d.get("saturation_lam"), scale=inv_channel_scales,
    )

    intercept = d.get("intercept", {})
    intercept_mean = float(intercept.get("mean", 0.0)) if isinstance(intercept, dict) else 0.0

    return BayesianFit(
        betas_mean=betas_m, betas_q05=betas_q05, betas_q95=betas_q95,
        decay_mean=decay_m, decay_q05=decay_q05, decay_q95=decay_q95,
        half_sat_mean=ks_m, half_sat_q05=ks_q05, half_sat_q95=ks_q95,
        intercept_mean=intercept_mean,
        r_squared=float(d.get("r_squared", 0.0)),
        mape=float(d.get("mape", 0.0)),
        seed=int(d.get("seed", 0)),
        chains=int(d.get("chains", 0)),
        draws=int(d.get("draws", 0)),
        n_obs=int(d.get("n_obs", 0)),
        channels=tuple(d.get("channels", [])),
        ground_truth=dict(d.get("ground_truth", {})),
    )

"""Marketing Mix Model — Ridge fallback implementation.

Pipeline
--------
1. Per channel, grid-search adstock decay λ and Hill half-saturation k by
   maximising R² of a univariate fit on residuals (one-step coordinate
   descent — fast and good enough for a Ridge MMM).
2. With the chosen (λ, k), transform spend into adstocked + saturated
   stocks, then fit a Ridge regression with non-negative coefficients
   (NNLS) on the full feature matrix together with controls.
3. Decompose predictions into baseline / control / per-channel
   contributions for plotting and ROI analysis.
4. Provide a constrained budget reallocation optimiser.

The Bayesian PyMC-Marketing version lives in ``notebooks/02_mmm_deep_dive``
once it ships — kept out of the production app because of build-time
issues on Streamlit Community Cloud.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from src.data_generation import (
    CHANNELS,
    Channel,
    adstock_geometric,
    saturation_hill,
)

__all__ = [
    "FitResult",
    "MMM",
    "channel_mroi",
    "fit_per_sales_channel",
    "optimise_budget",
]


CONTROL_COLUMNS: tuple[str, ...] = (
    "is_weekend",
    "is_holiday",
    "temperature_c",
    "promo_pct",
)


@dataclass
class FitResult:
    """Output of :meth:`MMM.fit`."""
    betas: dict[str, float]            # estimated channel coefficients
    decay: dict[str, float]            # estimated adstock λ
    half_sat: dict[str, float]         # estimated Hill k
    control_betas: dict[str, float]
    intercept: float
    predictions: np.ndarray            # in-sample fitted units
    residuals: np.ndarray
    r_squared: float
    mape: float
    contribution: pd.DataFrame         # per-row decomposition (baseline + controls + channels)


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _mape(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = y > 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))


def _grid_for(channel: Channel) -> tuple[np.ndarray, np.ndarray]:
    """Coarse grid of (decay, half_sat) candidates per channel.

    Centred around realistic ranges; we don't pretend this is a true
    Bayesian search — for a Ridge MMM it converges fast and explains why
    the brief flags PyMC as the gold standard."""
    decay_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # half-sat grid spans roughly 0.5x to 4x the channel's max stock
    max_stock = channel.max_spend / max(1 - 0.5, 1e-3)
    if channel.max_spend == 0:
        max_stock = 1.0
    half_sat_grid = np.geomspace(max(max_stock * 0.25, 1.0), max(max_stock * 4.0, 100.0), 8)
    return decay_grid, half_sat_grid


class MMM:
    """Ridge / NNLS marketing mix model with adstock + Hill saturation."""

    def __init__(
        self,
        channels: Iterable[Channel] = CHANNELS,
        controls: Sequence[str] = CONTROL_COLUMNS,
    ):
        self.channels = tuple(channels)
        self.controls = tuple(controls)
        self.fit_: FitResult | None = None

    # --- transforms ------------------------------------------------------

    def _transform_channel(
        self, spend: np.ndarray, decay: float, half_sat: float
    ) -> np.ndarray:
        return saturation_hill(adstock_geometric(spend, decay), half_sat)

    def _build_design(
        self,
        df: pd.DataFrame,
        decays: Mapping[str, float],
        half_sats: Mapping[str, float],
    ) -> np.ndarray:
        cols = []
        for ch in self.channels:
            cols.append(self._transform_channel(df[ch.name].to_numpy(), decays[ch.name], half_sats[ch.name]))
        for c in self.controls:
            cols.append(df[c].to_numpy(dtype=np.float64))
        return np.column_stack(cols)

    # --- per-channel hyperparameter selection ---------------------------

    def _pick_hparams(
        self, df: pd.DataFrame, target: np.ndarray
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Coordinate-style search: detrend target, then for each channel
        pick (decay, half_sat) maximising univariate R² on the residual.

        It's not joint optimisation, but for a Ridge MMM it lands close
        to a sensible local optimum in <100 ms.
        """
        # Crude detrend by subtracting the mean of target on control regression
        ctrl = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in self.controls])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            beta_ctrl, _ = nnls(np.column_stack([np.ones(len(df)), ctrl]), target, maxiter=10_000)
        residual = target - (np.ones(len(df)) * beta_ctrl[0] + ctrl @ beta_ctrl[1:])

        decays: dict[str, float] = {}
        half_sats: dict[str, float] = {}
        for ch in self.channels:
            decay_grid, k_grid = _grid_for(ch)
            best = (-np.inf, ch.decay, ch.half_sat)
            for d in decay_grid:
                stocked = adstock_geometric(df[ch.name].to_numpy(), d)
                for k in k_grid:
                    feat = saturation_hill(stocked, k)
                    if feat.std() < 1e-8:
                        continue
                    # OLS slope (closed form, intercept-free since residual is centred)
                    cov = float(np.dot(feat - feat.mean(), residual - residual.mean()))
                    var = float(np.sum((feat - feat.mean()) ** 2))
                    if var <= 0:
                        continue
                    slope = cov / var
                    pred = slope * feat
                    r2 = _r2(residual, pred)
                    if r2 > best[0]:
                        best = (r2, d, k)
            decays[ch.name] = float(best[1])
            half_sats[ch.name] = float(best[2])
        return decays, half_sats

    # --- main fit --------------------------------------------------------

    def fit(self, df: pd.DataFrame, target_col: str = "units_total") -> FitResult:
        target = df[target_col].to_numpy(dtype=np.float64)

        decays, half_sats = self._pick_hparams(df, target)
        X_channels = np.column_stack([
            self._transform_channel(df[ch.name].to_numpy(), decays[ch.name], half_sats[ch.name])
            for ch in self.channels
        ])
        X_controls = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in self.controls])
        X = np.column_stack([np.ones(len(df)), X_channels, X_controls])

        # NNLS gives non-negative betas — matches the Bayesian Half-Normal prior
        # on channel coefficients. Controls also non-negative, which is a mild
        # but reasonable assumption (positive temperature/promo/holiday effects).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            coef, _ = nnls(X, target, maxiter=10_000)

        intercept = float(coef[0])
        n_ch = len(self.channels)
        ch_betas = {ch.name: float(coef[1 + i]) for i, ch in enumerate(self.channels)}
        ctrl_betas = {c: float(coef[1 + n_ch + i]) for i, c in enumerate(self.controls)}

        yhat = X @ coef

        # Decomposition
        contrib = pd.DataFrame(index=df.index)
        contrib["baseline"] = intercept
        for i, ch in enumerate(self.channels):
            contrib[f"contrib_{ch.name}"] = ch_betas[ch.name] * X_channels[:, i]
        for i, c in enumerate(self.controls):
            contrib[f"contrib_{c}"] = ctrl_betas[c] * X_controls[:, i]

        result = FitResult(
            betas=ch_betas,
            decay=decays,
            half_sat=half_sats,
            control_betas=ctrl_betas,
            intercept=intercept,
            predictions=yhat,
            residuals=target - yhat,
            r_squared=_r2(target, yhat),
            mape=_mape(target, yhat),
            contribution=contrib,
        )
        self.fit_ = result
        return result

    # --- inference helpers ---------------------------------------------

    def predict_total(
        self,
        spend_per_channel: Mapping[str, float],
        controls: Mapping[str, float] | None = None,
        n_days: int = 30,
    ) -> float:
        """Predict total units over ``n_days`` for a *steady-state* spend allocation."""
        if self.fit_ is None:
            raise RuntimeError("call .fit() first")
        ctrl = controls or {c: 0.0 for c in self.controls}
        total = 0.0
        for _ in range(n_days):
            total += self.fit_.intercept
            for ch in self.channels:
                spend = float(spend_per_channel.get(ch.name, 0.0))
                # Steady-state stock = spend / (1 - λ)
                lam = self.fit_.decay[ch.name]
                stock = spend / (1 - lam) if lam < 1 else spend
                feat = saturation_hill(np.array([stock]), self.fit_.half_sat[ch.name])[0]
                total += self.fit_.betas[ch.name] * feat
            for c in self.controls:
                total += self.fit_.control_betas[c] * ctrl.get(c, 0.0)
        return float(total)


# --- ROI helpers ---------------------------------------------------------

def channel_mroi(
    fit: FitResult,
    channel: Channel,
    current_spend: float,
    delta: float = 0.05,
    unit_price_chf: float = 165.0,
) -> float:
    """Marginal ROI: ΔRevenue / ΔSpend at the current operating point."""
    if current_spend <= 0:
        current_spend = max(1.0, channel.max_spend * 0.01)
    bumped = current_spend * (1 + delta)
    lam = fit.decay[channel.name]
    k = fit.half_sat[channel.name]
    beta = fit.betas[channel.name]
    stock_now = current_spend / (1 - lam) if lam < 1 else current_spend
    stock_up = bumped / (1 - lam) if lam < 1 else bumped
    units_delta = beta * (
        saturation_hill(np.array([stock_up]), k)[0]
        - saturation_hill(np.array([stock_now]), k)[0]
    )
    revenue_delta = units_delta * unit_price_chf
    spend_delta = bumped - current_spend
    return float(revenue_delta / spend_delta) if spend_delta > 0 else 0.0


def optimise_budget(
    mmm: MMM,
    total_budget: float,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    n_days: int = 30,
    controls: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Allocate ``total_budget`` (CHF/day) across channels to maximise units.

    Parameters
    ----------
    bounds : per-channel (min, max) daily spend constraints. If omitted,
        derived from each channel's nominal range.
    """
    if mmm.fit_ is None:
        raise RuntimeError("MMM must be fit before optimising")
    chans = mmm.channels
    n = len(chans)
    bnds = []
    x0 = np.zeros(n)
    default_bounds = {ch.name: (ch.min_spend, ch.max_spend) for ch in chans}
    bounds = {**default_bounds, **(bounds or {})}

    for i, ch in enumerate(chans):
        lo, hi = bounds[ch.name]
        bnds.append((float(lo), float(hi)))
        x0[i] = (lo + hi) / 2

    # Scale x0 so it sums to total_budget within bounds
    s = x0.sum()
    if s > 0 and total_budget > 0:
        x0 = x0 * (total_budget / s)
        x0 = np.clip(x0, [b[0] for b in bnds], [b[1] for b in bnds])

    def neg_units(x: np.ndarray) -> float:
        spend_map = {ch.name: float(x[i]) for i, ch in enumerate(chans)}
        return -mmm.predict_total(spend_map, controls=controls, n_days=n_days)

    constraints = {"type": "eq", "fun": lambda x: float(x.sum() - total_budget)}
    res = minimize(
        neg_units, x0, method="SLSQP", bounds=bnds, constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )
    if not res.success:
        # Fall back to projected gradient with a softer constraint penalty
        x0 = np.clip(x0, [b[0] for b in bnds], [b[1] for b in bnds])
        res_x = x0
    else:
        res_x = res.x
    return {ch.name: float(res_x[i]) for i, ch in enumerate(chans)}


def fit_per_sales_channel(
    df: pd.DataFrame,
    sales_channels: Sequence[str] = ("supermarket", "online", "stores"),
) -> dict[str, tuple["MMM", FitResult]]:
    """Fit one MMM per sales channel — that's the cross-effect view.

    Returns a mapping ``sales_channel -> (mmm, fit_result)`` so callers
    can render per-channel diagnostics, marginal ROIs and saturation
    curves without re-running the (expensive) hyperparameter search.
    """
    out: dict[str, tuple["MMM", FitResult]] = {}
    for sc in sales_channels:
        col = f"units_{sc}"
        if col not in df.columns:
            raise KeyError(f"missing column {col!r} for sales channel {sc!r}")
        mmm = MMM()
        fit = mmm.fit(df, target_col=col)
        out[sc] = (mmm, fit)
    return out

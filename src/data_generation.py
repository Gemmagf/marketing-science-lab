"""Synthetic D2C running-shoes brand data generator.

Builds a daily panel (default 2024-01-01 → 2025-12-31) with six marketing
channels, exogenous controls (weather, holidays, promotions) and a known
data-generating process so downstream MMM and causal models can be
validated against ground truth.

The full DGP parameters used to generate ``units_sold`` are returned in a
:class:`GroundTruth` object — store it next to the dataset so the MMM
diagnostics page can show beta/decay/half-saturation recovery side by
side with the truth.

Note on beta scaling
--------------------
PROJECT_BRIEF.md sketches per-channel coefficients (1.8, 4.2, 2.5, 0.8,
1.0, 1.2). With Michaelis-Menten saturation S(x) ∈ [0, 1) those would
yield total marketing contributions of ≤12 units against a 12,000-unit
baseline — effectively zero signal. We multiply by ``BETA_SCALE = 1000``
so contributions sit in the realistic 30-50% of baseline range, while
preserving the brief's relative channel ratios.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import lfilter

__all__ = [
    "BETA_SCALE",
    "CHANNELS",
    "Channel",
    "GroundTruth",
    "SWISS_HOLIDAYS",
    "adstock_geometric",
    "generate_dataset",
    "saturation_hill",
]


# Fixed-date Swiss federal holidays + Easter-derived dates + retail peaks
# (Black Friday week, Cyber Monday) for the default 2024-2025 window.
SWISS_HOLIDAYS: Tuple[str, ...] = (
    # 2024 — federal/cantonal
    "2024-01-01", "2024-01-02", "2024-03-29", "2024-04-01",
    "2024-05-01", "2024-05-09", "2024-05-20", "2024-08-01",
    "2024-12-25", "2024-12-26",
    # 2024 — retail peaks
    "2024-11-25", "2024-11-26", "2024-11-27", "2024-11-28",
    "2024-11-29", "2024-12-02",
    # 2025 — federal/cantonal
    "2025-01-01", "2025-01-02", "2025-04-18", "2025-04-21",
    "2025-05-01", "2025-05-29", "2025-06-09", "2025-08-01",
    "2025-12-25", "2025-12-26",
    # 2025 — retail peaks
    "2025-11-24", "2025-11-25", "2025-11-26", "2025-11-27",
    "2025-11-28", "2025-12-01",
)


BETA_SCALE: float = 1000.0


@dataclass(frozen=True)
class Channel:
    """One marketing channel and its true DGP parameters."""
    name: str
    min_spend: float
    max_spend: float
    decay: float        # adstock λ ∈ [0, 1)
    half_sat: float     # saturation k > 0 (Michaelis-Menten half-saturation)
    beta_raw: float     # brief's raw coefficient — multiplied by BETA_SCALE at runtime
    sparsity: float = 1.0   # P(spend > 0) per day; 1.0 = always-on

    @property
    def beta(self) -> float:
        return self.beta_raw * BETA_SCALE


CHANNELS: Tuple[Channel, ...] = (
    Channel("tv_spend",       0,    25_000, 0.7,  80_000, 1.8, sparsity=0.10),
    Channel("search_spend",   500,   4_000, 0.3,  12_000, 4.2),
    Channel("social_spend",   1_000, 8_000, 0.5,  20_000, 2.5),
    Channel("display_spend",  200,   2_500, 0.4,   6_000, 0.8),
    Channel("email_sends",    0,   500_000, 0.1, 800_000, 1.0, sparsity=0.30),
    Channel("ooh_spend",      0,     8_000, 0.6,  25_000, 1.2, sparsity=0.20),
)


# --- Math primitives ------------------------------------------------------

def adstock_geometric(x: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock: ``y[t] = x[t] + λ * y[t-1]``.

    Un-normalised on purpose — the half-saturation constants in the DGP
    are calibrated to the un-normalised stock level. Implemented with a
    single-pole IIR filter so it runs in C inside scipy.
    """
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay!r}")
    x = np.asarray(x, dtype=np.float64)
    return lfilter([1.0], [1.0, -decay], x)


def saturation_hill(x: np.ndarray, half_sat: float, alpha: float = 1.0) -> np.ndarray:
    """Hill / Michaelis-Menten saturation: ``S(x) = x^α / (k^α + x^α)``.

    With ``alpha=1`` reduces to the Michaelis-Menten form used in the brief.
    Bounded in [0, 1).
    """
    if half_sat <= 0:
        raise ValueError(f"half_sat must be > 0, got {half_sat!r}")
    if alpha < 1.0:
        raise ValueError(f"alpha must be >= 1, got {alpha!r}")
    x = np.asarray(x, dtype=np.float64)
    xa = np.power(x, alpha)
    return xa / (half_sat ** alpha + xa)


# --- Ground-truth bookkeeping --------------------------------------------

# Baseline DGP constants (units of "shoes per day" unless noted)
BASELINE: float = 12_000.0
TREND_SLOPE: float = 0.05            # 5% annual growth
WEEKLY_AMP: float = 800.0
YEARLY_AMP: float = 2_000.0
WEATHER_COEF: float = 80.0           # units per °C above 15°C
PROMO_AMP: float = 4_000.0           # units per 100% promo
HOLIDAY_AMP: float = 5_000.0
NOISE_SIGMA: float = 600.0
UNIT_PRICE_CHF: float = 165.0


@dataclass
class GroundTruth:
    """Frozen record of every DGP parameter used to generate the panel."""
    betas: dict
    decay: dict
    half_sat: dict
    baseline: float
    trend_slope: float
    weekly_amp: float
    yearly_amp: float
    weather_coef: float
    promo_amp: float
    holiday_amp: float
    noise_sigma: float
    avg_unit_price_chf: float
    beta_scale: float

    def to_dict(self) -> dict:
        return asdict(self)


# --- Helpers --------------------------------------------------------------

def _generate_spend(
    rng: np.random.Generator,
    n_days: int,
    channel: Channel,
) -> np.ndarray:
    base = rng.uniform(channel.min_spend, channel.max_spend, size=n_days)
    if channel.sparsity < 1.0:
        active = rng.random(n_days) < channel.sparsity
        base = base * active
    return base


def _inject_tv_burst(
    spend: np.ndarray,
    dates: pd.DatetimeIndex,
    burst_start: str = "2025-05-05",
    burst_weeks: int = 4,
    burst_daily: float = 22_000.0,
) -> np.ndarray:
    start = pd.Timestamp(burst_start)
    end = start + pd.Timedelta(weeks=burst_weeks)
    mask = (dates >= start) & (dates < end)
    out = spend.copy()
    out[mask] = burst_daily
    return out


def _temperature(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(len(dates))
    yearly = 12.5 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
    noise = rng.normal(0, 2.5, size=len(dates))
    return 12.5 + yearly + noise   # roughly -5 to +30 °C


def _is_holiday(dates: pd.DatetimeIndex) -> np.ndarray:
    holidays = pd.to_datetime(list(SWISS_HOLIDAYS)).normalize()
    return np.asarray(dates.normalize().isin(holidays), dtype=int)


def _promo_pct(
    dates: pd.DatetimeIndex,
    is_holiday: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    levels = np.array([0, 10, 20, 30, 40], dtype=np.float64)
    p = rng.choice(levels, size=len(dates), p=[0.55, 0.20, 0.15, 0.07, 0.03])
    # Always promote at least 20% during retail peaks / holidays
    p = np.where(is_holiday == 1, np.maximum(p, 20.0), p)
    return p


# --- Public API -----------------------------------------------------------

def generate_dataset(
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
    inject_tv_burst: bool = True,
) -> Tuple[pd.DataFrame, GroundTruth]:
    """Generate the full daily synthetic dataset plus ground-truth metadata.

    Parameters
    ----------
    start, end : str
        Inclusive date range, ISO format.
    seed : int
        RNG seed for reproducibility.
    inject_tv_burst : bool
        If True, replaces TV spend during weeks of 2025-05-05 with a heavy
        burst (used as the natural experiment for the DiD example).

    Returns
    -------
    df : pd.DataFrame
        Daily panel with columns: ``date``, channel spend columns,
        ``temperature_c``, ``is_weekend``, ``is_holiday``, ``promo_pct``,
        per-channel ``contrib_<channel>``, ``base_contribution``,
        ``units_sold``, ``revenue_chf``.
    truth : GroundTruth
        Every DGP parameter used to generate the panel.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    t = np.arange(n, dtype=np.float64)

    df = pd.DataFrame({"date": dates})
    df["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    df["is_holiday"] = _is_holiday(dates)
    df["temperature_c"] = _temperature(dates, rng)
    df["promo_pct"] = _promo_pct(dates, df["is_holiday"].to_numpy(), rng)

    # Channel spend & per-channel contribution to units
    contribs_total = np.zeros(n)
    for ch in CHANNELS:
        spend = _generate_spend(rng, n, ch)
        if inject_tv_burst and ch.name == "tv_spend":
            spend = _inject_tv_burst(spend, dates)
        df[ch.name] = spend
        stocked = adstock_geometric(spend, ch.decay)
        contribution = ch.beta * saturation_hill(stocked, ch.half_sat)
        df[f"contrib_{ch.name}"] = contribution
        contribs_total += contribution

    # Baseline + trend + seasonality + controls
    baseline_t = BASELINE * (1 + TREND_SLOPE * t / 365.25)
    weekly = WEEKLY_AMP * np.sin(2 * np.pi * t / 7)
    yearly = YEARLY_AMP * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
    weather = WEATHER_COEF * (df["temperature_c"].to_numpy() - 15.0)
    promo_jitter = rng.uniform(0.8, 1.2, size=n)
    promo = PROMO_AMP * (df["promo_pct"].to_numpy() / 100.0) * promo_jitter
    holiday = HOLIDAY_AMP * df["is_holiday"].to_numpy()

    df["base_contribution"] = baseline_t + weekly + yearly + weather + promo + holiday

    # Outcome
    noise = rng.normal(0, NOISE_SIGMA, size=n)
    units = (df["base_contribution"] + contribs_total + noise).clip(lower=0)
    df["revenue_chf"] = units * UNIT_PRICE_CHF
    df["units_sold"] = units.round().astype(int)

    truth = GroundTruth(
        betas={ch.name: ch.beta for ch in CHANNELS},
        decay={ch.name: ch.decay for ch in CHANNELS},
        half_sat={ch.name: ch.half_sat for ch in CHANNELS},
        baseline=BASELINE,
        trend_slope=TREND_SLOPE,
        weekly_amp=WEEKLY_AMP,
        yearly_amp=YEARLY_AMP,
        weather_coef=WEATHER_COEF,
        promo_amp=PROMO_AMP,
        holiday_amp=HOLIDAY_AMP,
        noise_sigma=NOISE_SIGMA,
        avg_unit_price_chf=UNIT_PRICE_CHF,
        beta_scale=BETA_SCALE,
    )
    return df, truth

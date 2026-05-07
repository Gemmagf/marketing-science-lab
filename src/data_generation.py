"""Synthetic AlpSel omnichannel retail dataset.

Builds a 731-day daily panel for a (fictional) Swiss omnichannel retailer
with three sales channels (supermarket, online, specialty stores) and
seven marketing channels. Each marketing channel affects each sales
channel differently — that's the "cross-effect matrix" hiring managers
want to see.

The data-generating process is fully known so downstream models can be
validated against ground truth.

Sales channels
--------------
- ``supermarket``: high volume, low basket value (CHF 85 / unit).
  Most receptive to TV, OOH and leaflets.
- ``online``: medium volume, mid basket (CHF 165). Most receptive to
  paid search, social and display retargeting.
- ``stores``: lower volume, premium basket (CHF 220) — specialty
  formats. Receptive to TV + OOH + leaflets but less to paid digital.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import lfilter

__all__ = [
    "BASELINE",
    "CAMPAIGNS",
    "CHANNELS",
    "Channel",
    "GroundTruth",
    "SALES_CHANNELS",
    "SWISS_HOLIDAYS",
    "UNIT_PRICE_CHF",
    "adstock_geometric",
    "generate_dataset",
    "saturation_hill",
]


# Sales channels — the three places AlpSel actually sells
SALES_CHANNELS: tuple[str, ...] = ("supermarket", "online", "stores")

SALES_CHANNEL_LABELS: dict[str, str] = {
    "supermarket": "Supermarket",
    "online":      "Online store",
    "stores":      "Specialty stores",
}

# Per-sales-channel daily baseline (units organic) and basket value (CHF)
BASELINE: dict[str, float] = {
    "supermarket": 6_000.0,   # ~50% of total
    "online":      4_200.0,   # ~35%
    "stores":      1_800.0,   # ~15%
}
UNIT_PRICE_CHF: dict[str, float] = {
    "supermarket": 85.0,
    "online":      165.0,
    "stores":      220.0,
}


# Swiss federal holidays + retail peaks for 2024-2025
SWISS_HOLIDAYS: Tuple[str, ...] = (
    "2024-01-01", "2024-01-02", "2024-03-29", "2024-04-01",
    "2024-05-01", "2024-05-09", "2024-05-20", "2024-08-01",
    "2024-12-25", "2024-12-26",
    "2024-11-25", "2024-11-26", "2024-11-27", "2024-11-28",
    "2024-11-29", "2024-12-02",
    "2025-01-01", "2025-01-02", "2025-04-18", "2025-04-21",
    "2025-05-01", "2025-05-29", "2025-06-09", "2025-08-01",
    "2025-12-25", "2025-12-26",
    "2025-11-24", "2025-11-25", "2025-11-26", "2025-11-27",
    "2025-11-28", "2025-12-01",
)


# Named retail campaigns
CAMPAIGNS: tuple[tuple[str, str, str], ...] = (
    ("2024-03-01", "2024-03-21", "Spring Launch"),
    ("2024-06-15", "2024-07-15", "Summer Peak"),
    ("2024-11-20", "2024-12-02", "Black Friday Wave"),
    ("2024-12-12", "2024-12-25", "Holiday Push"),
    ("2025-01-15", "2025-02-05", "Winter Clearance"),
    ("2025-05-05", "2025-06-02", "TV Burst (DACH)"),
    ("2025-09-15", "2025-10-15", "Autumn Activation"),
    ("2025-11-19", "2025-12-01", "Black Friday Wave"),
    ("2025-12-10", "2025-12-25", "Holiday Push"),
)


@dataclass(frozen=True)
class Channel:
    """One marketing channel + the per-sales-channel betas (cross-effect row)."""
    name: str
    label: str
    min_spend: float
    max_spend: float
    decay: float                            # adstock λ
    half_sat: float                         # Hill k
    beta_supermarket: float                 # incremental units / saturated stock unit, supermarket
    beta_online: float                      # incremental units / saturated stock unit, online
    beta_stores: float                      # incremental units / saturated stock unit, stores
    sparsity: float = 1.0                   # P(spend > 0) per day
    kind: str = "online"                    # "online" | "offline" | "instore"

    def beta_for(self, sales_channel: str) -> float:
        return {
            "supermarket": self.beta_supermarket,
            "online":      self.beta_online,
            "stores":      self.beta_stores,
        }[sales_channel]


# Cross-effect matrix encoded in the betas. Reads:
#   "TV is the strongest driver of supermarket; paid search the strongest of online;
#    leaflets are the strongest in-store driver and barely touch online."
CHANNELS: Tuple[Channel, ...] = (
    Channel(
        name="tv_spend", label="TV (linear & CTV)",
        min_spend=0, max_spend=25_000, decay=0.7, half_sat=80_000,
        beta_supermarket=2_200, beta_online=1_500, beta_stores=900,
        sparsity=0.10, kind="offline",
    ),
    Channel(
        name="search_spend", label="Paid search",
        min_spend=500, max_spend=4_000, decay=0.3, half_sat=12_000,
        beta_supermarket=200, beta_online=4_500, beta_stores=300,
        kind="online",
    ),
    Channel(
        name="social_spend", label="Social (Meta + TikTok)",
        min_spend=1_000, max_spend=8_000, decay=0.5, half_sat=20_000,
        beta_supermarket=400, beta_online=2_800, beta_stores=600,
        kind="online",
    ),
    Channel(
        name="display_spend", label="Display & retargeting",
        min_spend=200, max_spend=2_500, decay=0.4, half_sat=6_000,
        beta_supermarket=200, beta_online=1_200, beta_stores=200,
        kind="online",
    ),
    Channel(
        name="email_sends", label="Email & CRM",
        min_spend=0, max_spend=500_000, decay=0.1, half_sat=800_000,
        beta_supermarket=300, beta_online=900, beta_stores=200,
        sparsity=0.30, kind="online",
    ),
    Channel(
        name="ooh_spend", label="OOH (DOOH + transit)",
        min_spend=0, max_spend=8_000, decay=0.6, half_sat=25_000,
        beta_supermarket=900, beta_online=300, beta_stores=600,
        sparsity=0.20, kind="offline",
    ),
    Channel(
        name="leaflet_spend", label="Leaflets & in-store promo",
        min_spend=0, max_spend=12_000, decay=0.3, half_sat=30_000,
        beta_supermarket=2_800, beta_online=200, beta_stores=1_200,
        sparsity=0.40, kind="instore",
    ),
)


# --- Math primitives ------------------------------------------------------

def adstock_geometric(x: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock: y[t] = x[t] + λ · y[t-1] via single-pole IIR filter."""
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay!r}")
    x = np.asarray(x, dtype=np.float64)
    return lfilter([1.0], [1.0, -decay], x)


def saturation_hill(x: np.ndarray, half_sat: float, alpha: float = 1.0) -> np.ndarray:
    """Hill / Michaelis-Menten saturation: S(x) = x^α / (k^α + x^α)."""
    if half_sat <= 0:
        raise ValueError(f"half_sat must be > 0, got {half_sat!r}")
    if alpha < 1.0:
        raise ValueError(f"alpha must be >= 1, got {alpha!r}")
    x = np.asarray(x, dtype=np.float64)
    xa = np.power(x, alpha)
    return xa / (half_sat ** alpha + xa)


# --- DGP constants -------------------------------------------------------

TREND_SLOPE: float = 0.05
WEEKLY_AMP_SHARE: dict[str, float] = {"supermarket": 0.10, "online": 0.05, "stores": 0.07}
YEARLY_AMP_SHARE: dict[str, float] = {"supermarket": 0.15, "online": 0.20, "stores": 0.25}
WEATHER_COEF: dict[str, float] = {"supermarket": 50.0, "online": 30.0, "stores": 20.0}
PROMO_AMP: dict[str, float] = {"supermarket": 2_500.0, "online": 1_500.0, "stores": 600.0}
HOLIDAY_AMP: dict[str, float] = {"supermarket": 3_000.0, "online": 1_500.0, "stores": 800.0}
NOISE_SIGMA_SHARE: float = 0.07     # σ as 7% of channel baseline


@dataclass
class GroundTruth:
    """Frozen DGP record."""
    betas: dict                              # {channel: {sales_channel: β}}
    decay: dict                              # {channel: λ}
    half_sat: dict                           # {channel: k}
    baseline: dict                           # {sales_channel: baseline_units}
    unit_price_chf: dict                     # {sales_channel: CHF per unit}
    sales_channels: tuple = field(default_factory=lambda: SALES_CHANNELS)

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
    spend: np.ndarray, dates: pd.DatetimeIndex,
    burst_start: str = "2025-05-05", burst_weeks: int = 4,
    burst_daily: float = 22_000.0, rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    start = pd.Timestamp(burst_start)
    end = start + pd.Timedelta(weeks=burst_weeks)
    mask = (dates >= start) & (dates < end)
    out = spend.copy()
    burst_dates = dates[mask]
    weekend_lift = np.where(burst_dates.dayofweek >= 5, 1.15, 0.95)
    jitter = rng.normal(1.0, 0.08, size=len(burst_dates)).clip(0.7, 1.3)
    out[mask] = burst_daily * weekend_lift * jitter
    return out


def _temperature(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(len(dates))
    yearly = 12.5 * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
    noise = rng.normal(0, 2.5, size=len(dates))
    return 12.5 + yearly + noise


def _is_holiday(dates: pd.DatetimeIndex) -> np.ndarray:
    holidays = pd.to_datetime(list(SWISS_HOLIDAYS)).normalize()
    return np.asarray(dates.normalize().isin(holidays), dtype=int)


def _label_campaigns(dates: pd.DatetimeIndex) -> np.ndarray:
    out = np.array([""] * len(dates), dtype=object)
    for start, end, name in CAMPAIGNS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        out[(dates >= s) & (dates < e)] = name
    return out


def _promo_pct(
    dates: pd.DatetimeIndex, is_holiday: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    levels = np.array([0, 10, 20, 30, 40], dtype=np.float64)
    p = rng.choice(levels, size=len(dates), p=[0.55, 0.20, 0.15, 0.07, 0.03])
    p = np.where(is_holiday == 1, np.maximum(p, 20.0), p)
    return p


# --- Public API -----------------------------------------------------------

def generate_dataset(
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
    inject_tv_burst: bool = True,
) -> Tuple[pd.DataFrame, GroundTruth]:
    """Generate the AlpSel omnichannel daily panel + ground-truth metadata."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    t = np.arange(n, dtype=np.float64)

    df = pd.DataFrame({"date": dates})
    df["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    df["is_holiday"] = _is_holiday(dates)
    df["temperature_c"] = _temperature(dates, rng)
    df["promo_pct"] = _promo_pct(dates, df["is_holiday"].to_numpy(), rng)
    df["campaign"] = _label_campaigns(dates)

    # Adstocked + saturated stock per marketing channel (one column per pair feels noisy
    # at the dashboard level — store the saturated feature once and reuse for each sales channel)
    saturated: dict[str, np.ndarray] = {}
    for ch in CHANNELS:
        spend = _generate_spend(rng, n, ch)
        if inject_tv_burst and ch.name == "tv_spend":
            spend = _inject_tv_burst(spend, dates, rng=rng)
        df[ch.name] = spend
        saturated[ch.name] = saturation_hill(adstock_geometric(spend, ch.decay), ch.half_sat)

    # Per-sales-channel outcomes
    revenue_total = np.zeros(n)
    units_total = np.zeros(n)

    weekly_phase = 2 * np.pi * t / 7
    yearly_phase = 2 * np.pi * t / 365.25 - np.pi / 2
    weather_centred = df["temperature_c"].to_numpy() - 15.0
    promo_jitter = rng.uniform(0.8, 1.2, size=n)
    promo_pct = df["promo_pct"].to_numpy()
    holiday = df["is_holiday"].to_numpy()

    for sc in SALES_CHANNELS:
        baseline = BASELINE[sc] * (1 + TREND_SLOPE * t / 365.25)
        weekly = baseline * WEEKLY_AMP_SHARE[sc] * np.sin(weekly_phase)
        yearly = baseline * YEARLY_AMP_SHARE[sc] * np.sin(yearly_phase)
        weather = WEATHER_COEF[sc] * weather_centred
        promo = PROMO_AMP[sc] * (promo_pct / 100.0) * promo_jitter
        holiday_lift = HOLIDAY_AMP[sc] * holiday

        contrib_total = np.zeros(n)
        for ch in CHANNELS:
            beta = ch.beta_for(sc)
            ch_contrib = beta * saturated[ch.name]
            df[f"contrib_{ch.name}__{sc}"] = ch_contrib
            contrib_total += ch_contrib

        sigma = NOISE_SIGMA_SHARE * BASELINE[sc]
        noise = rng.normal(0, sigma, size=n)
        units = baseline + weekly + yearly + weather + promo + holiday_lift + contrib_total + noise
        units = np.clip(units, 0, None)

        df[f"units_{sc}"] = units.round().astype(int)
        df[f"revenue_{sc}"] = units * UNIT_PRICE_CHF[sc]
        df[f"baseline_{sc}"] = baseline
        units_total += units
        revenue_total += units * UNIT_PRICE_CHF[sc]

    df["units_total"] = units_total.round().astype(int)
    df["revenue_total"] = revenue_total

    truth = GroundTruth(
        betas={
            ch.name: {sc: ch.beta_for(sc) for sc in SALES_CHANNELS}
            for ch in CHANNELS
        },
        decay={ch.name: ch.decay for ch in CHANNELS},
        half_sat={ch.name: ch.half_sat for ch in CHANNELS},
        baseline=dict(BASELINE),
        unit_price_chf=dict(UNIT_PRICE_CHF),
    )
    return df, truth

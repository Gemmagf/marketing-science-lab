"""AlpSel · Decision Dashboard — ACTIVATE / WATCH / PAUSE.

Three columns, one row per recommendation. Each card has the action,
the CHF impact, the confidence and a deadline. The point is to walk out
of the page knowing exactly what to do this week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import brand
from src.data_generation import (
    BASELINE, CHANNELS, SALES_CHANNELS, UNIT_PRICE_CHF, generate_dataset,
    saturation_hill,
)
from src.mmm import fit_per_sales_channel, optimise_budget


st.set_page_config(
    page_title=f"{brand.BRAND_NAME} · Decision Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

brand.render_page_chrome("Decisions", "—",
                         "What to do this week — three columns, one row per action")
st.subheader(
    "🟢 ACTIVATE the underspent · 🟡 WATCH the marginal · 🔴 PAUSE the saturated."
)


@st.cache_data(show_spinner="Loading panel…")
def _load(seed: int = 42):
    return generate_dataset(seed=seed)


@st.cache_resource(show_spinner="Fitting MMM per sales channel…")
def _fit_all(seed: int = 42):
    df, truth = _load(seed)
    fits = fit_per_sales_channel(df)
    return df, truth, fits


df, truth, fits = _fit_all()


# ============================================================================
# Build per-(marketing, sales) recommendations
# ============================================================================

current_alloc = {ch.name: float(df[ch.name].mean()) for ch in CHANNELS}
total_current = sum(current_alloc.values())

records: list[dict] = []
for sc, (mmm, fit) in fits.items():
    optimal = optimise_budget(mmm, total_budget=total_current)
    for ch in CHANNELS:
        delta = optimal[ch.name] - current_alloc[ch.name]
        decay = fit.decay[ch.name]
        half_sat = fit.half_sat[ch.name]
        beta = fit.betas[ch.name]
        cur_stock = current_alloc[ch.name] / max(1 - decay, 1e-3)
        ds_dx = half_sat / ((half_sat + cur_stock) ** 2) if half_sat else 0
        marginal_per_chf_units = beta * ds_dx
        annual_chf = delta * marginal_per_chf_units * UNIT_PRICE_CHF[sc] * 365

        # Saturation level: how close is current_stock to half-saturation?
        # > 1.5 → past saturation (every CHF wasted)
        # 0.5..1.5 → on the curve (marginal returns)
        # < 0.5 → starved (steepest part of the curve)
        sat_ratio = cur_stock / half_sat if half_sat else 0

        records.append({
            "marketing_channel": ch.label,
            "sales_channel": sc.title(),
            "delta_chf_day": delta,
            "annual_chf_impact": annual_chf,
            "sat_ratio": sat_ratio,
            "current_chf_day": current_alloc[ch.name],
            "optimal_chf_day": optimal[ch.name],
            "decay": decay,
        })

records_df = pd.DataFrame(records)


# Bucket each row into ACTIVATE / WATCH / PAUSE by impact + saturation
def _bucket(row):
    if row["annual_chf_impact"] > 50_000 and row["sat_ratio"] < 0.6:
        return "activate"
    if row["annual_chf_impact"] < -50_000 or row["sat_ratio"] > 1.5:
        return "pause"
    return "watch"

records_df["bucket"] = records_df.apply(_bucket, axis=1)


# Rank within each bucket: ACTIVATE by upside desc, PAUSE by saving desc, WATCH by abs impact
activate = records_df[records_df["bucket"] == "activate"].sort_values(
    "annual_chf_impact", ascending=False).head(6)
pause = records_df[records_df["bucket"] == "pause"].sort_values(
    "annual_chf_impact", ascending=True).head(6)
watch = records_df[records_df["bucket"] == "watch"].sort_values(
    "annual_chf_impact", key=abs, ascending=False).head(6)


# ============================================================================
# 3-column layout
# ============================================================================

col_a, col_w, col_p = st.columns(3, gap="medium")


def _card(row, mode: str) -> None:
    color = {"activate": brand.COLOURS["good"],
             "watch":    brand.COLOURS["warn"],
             "pause":    brand.COLOURS["danger"]}[mode]
    icon = {"activate": "🟢", "watch": "🟡", "pause": "🔴"}[mode]
    arrow = "↑" if row["delta_chf_day"] > 0 else "↓"
    deadline = {"activate": "this week", "watch": "this month", "pause": "this week"}[mode]

    with st.container(border=True):
        st.caption(f"{icon} **{mode.upper()}** · deadline: {deadline}")
        st.markdown(f"#### {arrow} {row['marketing_channel']}")
        st.caption(f"impact on **{row['sales_channel']}**")
        st.markdown(
            f"Move from **CHF {row['current_chf_day']:,.0f}** → "
            f"**CHF {row['optimal_chf_day']:,.0f}** per day "
            f"(saturation ratio {row['sat_ratio']:.2f}× of half-sat)."
        )
        sign = "+" if row["annual_chf_impact"] >= 0 else ""
        st.metric("Annual CHF impact", f"{sign}CHF {row['annual_chf_impact']:,.0f}")


with col_a:
    st.markdown("### 🟢 ACTIVATE")
    st.caption("Channels still on the steep part of the curve — every CHF here pulls weight.")
    if len(activate) == 0:
        st.info("No clear scale-up moves at the current allocation.")
    for _, row in activate.iterrows():
        _card(row, "activate")
        st.write("")

with col_w:
    st.markdown("### 🟡 WATCH")
    st.caption("Marginal performers — keep on, but instrument heavily.")
    if len(watch) == 0:
        st.info("No marginal channels to watch.")
    for _, row in watch.iterrows():
        _card(row, "watch")
        st.write("")

with col_p:
    st.markdown("### 🔴 PAUSE")
    st.caption("Past saturation or actively underperforming — stop the bleeding.")
    if len(pause) == 0:
        st.info("Nothing meeting the pause threshold.")
    for _, row in pause.iterrows():
        _card(row, "pause")
        st.write("")


st.divider()


# ============================================================================
# Summary KPIs
# ============================================================================

total_activate = activate["annual_chf_impact"].sum()
total_pause = pause["annual_chf_impact"].sum()
net_impact = total_activate - total_pause   # pause impact is negative

c1, c2, c3, c4 = st.columns(4)
c1.metric("Activate impact", f"+CHF {total_activate:,.0f}/yr")
c2.metric("Pause savings", f"+CHF {abs(total_pause):,.0f}/yr",
          help="Stops the bleeding from saturated/underperforming channels")
c3.metric("Net annual impact", f"CHF {net_impact:,.0f}/yr",
          delta=f"{net_impact/df['revenue_total'].sum()*100:+.1f}% of current revenue")
c4.metric("Actions to take", f"{len(activate) + len(pause)}",
          help=f"{len(activate)} activate · {len(pause)} pause · {len(watch)} watch")


brand.render_synthetic_disclaimer()

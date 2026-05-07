"""AlpSel · Marketing Mix per channel.

Pick a sales channel (supermarket / online / stores) and see the MMM
fit for it: KPIs, decomposition, saturation curves, and the budget
reallocation that maximises units in *that* channel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import brand, viz
from src.bayesian import load_posterior
from src.data_generation import (
    BASELINE, CHANNELS, SALES_CHANNELS, UNIT_PRICE_CHF, generate_dataset,
    saturation_hill,
)
from src.mmm import MMM, channel_mroi, fit_per_sales_channel, optimise_budget


st.set_page_config(
    page_title=f"{brand.BRAND_NAME} · MMM per channel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


brand.render_page_chrome("MMM", "—",
                         "Marketing Mix Model — one fit per sales channel")
st.subheader(
    "Pick the sales channel you want to optimise for. Each fit isolates how "
    "marketing spend converts into units sold *for that channel*."
)


@st.cache_data(show_spinner="Loading panel…")
def _load(seed: int = 42):
    return generate_dataset(seed=seed)


@st.cache_resource(show_spinner="Fitting one MMM per sales channel…")
def _fit_all(seed: int = 42):
    df, truth = _load(seed)
    fits = fit_per_sales_channel(df)
    return df, truth, fits


df, truth, fits = _fit_all()
posterior = load_posterior()

# Sidebar / top selector for the sales channel
sc = st.radio(
    "Sales channel",
    SALES_CHANNELS,
    format_func=lambda s: {"supermarket": "🛒 Supermarket",
                           "online": "🛍 Online store",
                           "stores": "🏬 Specialty stores"}[s],
    horizontal=True,
)

mmm, fit = fits[sc]
unit_price = UNIT_PRICE_CHF[sc]


# --- Headline KPIs -------------------------------------------------------

st.divider()
st.markdown(f"### Last 24 months — {sc.title()}")

total_revenue = float(df[f"revenue_{sc}"].sum())
total_units = float(df[f"units_{sc}"].sum())
total_spend = float(df[["tv_spend", "search_spend", "social_spend",
                        "display_spend", "ooh_spend", "leaflet_spend"]].sum().sum())
roas = total_revenue / total_spend if total_spend else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Revenue ({sc})", f"CHF {total_revenue/1e6:,.1f}M")
k2.metric(f"Units sold ({sc})", f"{total_units:,.0f}")
k3.metric("Marketing spend", f"CHF {total_spend/1e6:,.1f}M")
k4.metric("ROAS attributable", f"{roas:,.1f}×",
          help=f"Total marketing CHF / {sc} revenue. "
               "Note: marketing also lifts other sales channels.")


# --- Channel decomposition -----------------------------------------------

st.markdown(f"### Decomposition — drivers of {sc.title()} units")
fit_view = fit.contribution.copy()
decomp_df = df[["date"]].copy()
decomp_df["baseline"] = fit_view["baseline"].values
for ch in CHANNELS:
    decomp_df[brand.CHANNEL_LABELS[ch.name]] = fit_view[f"contrib_{ch.name}"].values

palette = [
    brand.COLOURS["primary"], brand.COLOURS["accent"], brand.COLOURS["good"],
    brand.COLOURS["warn"], "#7A4DC8", brand.COLOURS["muted"], "#E91E63",
]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=decomp_df["date"], y=decomp_df["baseline"], mode="lines",
    name="Baseline (organic)", line=dict(color=brand.COLOURS["muted"], width=1),
    stackgroup="one",
))
for ch, col in zip(CHANNELS, palette):
    fig.add_trace(go.Scatter(
        x=decomp_df["date"], y=decomp_df[brand.CHANNEL_LABELS[ch.name]],
        mode="lines", name=brand.CHANNEL_LABELS[ch.name],
        line=dict(width=0), stackgroup="one", fillcolor=col,
    ))
fig.update_layout(
    title=f"Daily {sc} units — baseline + channel contributions",
    plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
    height=400, margin=dict(l=20, r=20, t=50, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor=brand.COLOURS["panel"], title="Units")
st.plotly_chart(fig, use_container_width=True)


# --- Saturation curves --------------------------------------------------

st.markdown(f"### Diminishing returns — where each channel sits for {sc.title()}")
st.caption(
    "Vertical bar = current avg daily spend. Channels on the flat part are "
    "wasting marginal CHF (candidates for PAUSE); channels still climbing are "
    "starved (candidates for ACTIVATE)."
)

cols = st.columns(3, gap="medium")
for i, ch in enumerate(CHANNELS):
    grid = np.linspace(1, ch.max_spend * 2, 80)
    lam = fit.decay[ch.name]
    stock = grid / max(1 - lam, 1e-3)
    response = fit.betas[ch.name] * saturation_hill(stock, fit.half_sat[ch.name])
    fig_s = viz.saturation_curve(
        spend_grid=grid, response=response,
        current_spend=float(df[ch.name].mean()),
        title=brand.CHANNEL_LABELS[ch.name],
        colour=palette[i],
    )
    fig_s.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=30))
    cols[i % 3].plotly_chart(fig_s, use_container_width=True)


# --- Optimiser ---------------------------------------------------------

st.divider()
st.markdown(f"### Optimal allocation for {sc.title()}")

current_alloc = {ch.name: float(df[ch.name].mean()) for ch in CHANNELS}
default_total = float(sum(current_alloc.values()))

target_total = st.slider(
    "Total daily marketing budget (CHF + email sends, summed)",
    min_value=int(default_total * 0.7),
    max_value=int(default_total * 1.3),
    value=int(default_total),
    step=1000,
)

optimal = optimise_budget(mmm, total_budget=float(target_total))

units_current = mmm.predict_total(current_alloc, n_days=90)
units_optimal = mmm.predict_total(optimal, n_days=90)
revenue_delta_year = (units_optimal - units_current) * unit_price * (365.0 / 90.0)

c_left, c_right = st.columns([1, 1.2], gap="large")
with c_left:
    if revenue_delta_year > 1000:
        st.success(
            f"#### Reallocate to capture **+CHF {revenue_delta_year:,.0f}** of "
            f"incremental annual revenue in {sc}."
        )
    elif revenue_delta_year > 0:
        st.info("#### Current allocation is close to optimal for this channel.")
    else:
        st.warning("#### Current allocation already at the local optimum.")

    st.markdown("**Top moves:**")
    moves = []
    for ch in CHANNELS:
        delta = optimal[ch.name] - current_alloc[ch.name]
        moves.append((ch.label, current_alloc[ch.name], optimal[ch.name], delta))
    for lbl, cur, opt, delta in sorted(moves, key=lambda x: abs(x[3]), reverse=True)[:4]:
        if abs(delta) < 50:
            continue
        arrow = "↑" if delta >= 0 else "↓"
        pct = (delta / cur) if cur > 0 else 0
        st.markdown(
            f"- **{lbl}** — {arrow} CHF {abs(delta):,.0f}/day "
            f"(`{cur:,.0f}` → `{opt:,.0f}`, **{pct:+.0%}**)"
        )

with c_right:
    labels = [brand.CHANNEL_LABELS[ch.name] for ch in CHANNELS]
    fig_opt = go.Figure()
    fig_opt.add_trace(go.Bar(
        x=labels, y=[current_alloc[ch.name] for ch in CHANNELS],
        name="Current", marker_color=brand.COLOURS["muted"],
    ))
    fig_opt.add_trace(go.Bar(
        x=labels, y=[optimal[ch.name] for ch in CHANNELS],
        name="Optimal", marker_color=brand.COLOURS["accent"],
    ))
    fig_opt.update_layout(
        barmode="group", title=f"Allocation — current vs optimal for {sc}",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=320, margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig_opt.update_xaxes(tickangle=-25)
    fig_opt.update_yaxes(gridcolor=brand.COLOURS["panel"], title="CHF / day")
    st.plotly_chart(fig_opt, use_container_width=True)


# --- Diagnostics expander ----------------------------------------------

with st.expander("📐 Diagnostics — for the analyst"):
    d1, d2 = st.columns(2)
    d1.metric("Ridge R²", f"{fit.r_squared:.3f}")
    d2.metric("Ridge MAPE", f"{fit.mape:.2%}")
    if posterior is not None:
        st.caption(
            f"A Bayesian posterior is also available (R² {posterior.r_squared:.0%}, "
            f"6/6 channels recovered within 90% CI on the previous single-output "
            f"model — multi-output Bayesian fit is on the roadmap)."
        )

    # Recovery vs ground truth (per-sales-channel βs from the new DGP)
    truth_betas_for_sc = {ch.name: truth.betas[ch.name][sc] for ch in CHANNELS}
    rec = pd.DataFrame({
        "channel":     [brand.CHANNEL_LABELS[ch.name] for ch in CHANNELS],
        "true β":      [truth_betas_for_sc[ch.name] for ch in CHANNELS],
        "Ridge β":     [fit.betas[ch.name] for ch in CHANNELS],
        "true λ":      [truth.decay[ch.name] for ch in CHANNELS],
        "Ridge λ":     [fit.decay[ch.name] for ch in CHANNELS],
    })
    st.dataframe(rec.style.format("{:,.2f}", subset=["true β", "Ridge β", "true λ", "Ridge λ"]),
                 use_container_width=True, hide_index=True)


brand.render_synthetic_disclaimer()

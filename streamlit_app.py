"""AlpSel · Where are sales coming from?

Single-screen omnichannel command centre: revenue mix by sales channel,
cross-effect matrix between marketing and sales, and a CTA to the deeper
dashboards (per-channel MMM, missed opportunities, decision dashboard).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import brand
from src.data_generation import (
    BASELINE, CHANNELS, SALES_CHANNELS, UNIT_PRICE_CHF, generate_dataset,
)


st.set_page_config(
    page_title=f"{brand.BRAND_NAME} · Marketing Science Lab",
    page_icon="🏔",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_data(show_spinner="Loading AlpSel panel…")
def _load(seed: int = 42):
    return generate_dataset(seed=seed)


df, truth = _load()


# ============================================================================
# HEADER
# ============================================================================

st.caption(f"**{brand.BRAND_NAME}** · Marketing Science Lab — omnichannel retail")
st.title("Where are sales coming from?")
st.subheader(
    "AlpSel sells through three channels: supermarket (50% revenue), online "
    "store (35%) and specialty stores (15%). This is the command centre."
)

st.divider()


# ============================================================================
# 1 · REVENUE MIX
# ============================================================================

st.header("1 · Revenue mix by sales channel")

revenue_by_sc = {sc: float(df[f"revenue_{sc}"].sum()) for sc in SALES_CHANNELS}
units_by_sc = {sc: float(df[f"units_{sc}"].sum()) for sc in SALES_CHANNELS}
total_revenue = sum(revenue_by_sc.values())
total_units = sum(units_by_sc.values())

# 4 KPI columns: total + 3 sales channels
k0, k1, k2, k3 = st.columns(4)
k0.metric(
    "Total revenue (24 months)",
    f"CHF {total_revenue/1e6:,.1f}M",
    help=f"{total_units:,.0f} units across all channels",
)
sc_palette = {
    "supermarket": brand.COLOURS["primary"],
    "online":      brand.COLOURS["accent"],
    "stores":      brand.COLOURS["good"],
}
sc_labels = {
    "supermarket": "🛒 Supermarket",
    "online":      "🛍 Online store",
    "stores":      "🏬 Specialty stores",
}
for col, sc in zip([k1, k2, k3], SALES_CHANNELS):
    share = revenue_by_sc[sc] / total_revenue if total_revenue else 0
    col.metric(
        sc_labels[sc],
        f"CHF {revenue_by_sc[sc]/1e6:,.1f}M",
        help=f"{share:.0%} of total · {units_by_sc[sc]:,.0f} units · "
             f"basket {UNIT_PRICE_CHF[sc]:.0f} CHF",
    )
    col.caption(f"{share:.0%} of revenue")

# Time-series stacked area + donut
c_left, c_right = st.columns([2, 1], gap="large")
with c_left:
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")[[f"revenue_{sc}" for sc in SALES_CHANNELS]].sum()
    fig = go.Figure()
    for sc in SALES_CHANNELS:
        fig.add_trace(go.Scatter(
            x=monthly.index, y=monthly[f"revenue_{sc}"], mode="lines",
            name=sc_labels[sc], line=dict(width=0),
            stackgroup="one", fillcolor=sc_palette[sc],
        ))
    fig.update_layout(
        title="Monthly revenue stack — where each franc lands",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=360, margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=brand.COLOURS["panel"], title="CHF")
    st.plotly_chart(fig, use_container_width=True)

with c_right:
    fig_donut = go.Figure(go.Pie(
        labels=[sc_labels[sc] for sc in SALES_CHANNELS],
        values=[revenue_by_sc[sc] for sc in SALES_CHANNELS],
        hole=0.55,
        marker=dict(colors=[sc_palette[sc] for sc in SALES_CHANNELS]),
        textinfo="label+percent",
    ))
    fig_donut.update_layout(
        title="Revenue mix",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=360, margin=dict(l=20, r=20, t=50, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

st.divider()


# ============================================================================
# 2 · CROSS-EFFECT MATRIX
# ============================================================================

st.header("2 · Cross-effect matrix — what drives what")
st.caption(
    "Each cell shows the marginal impact of the marketing channel (rows) "
    "on the sales channel (columns), as units per CHF spent. Higher = "
    "stronger driver. The diagonal-ish pattern shows online ads drive "
    "online sales, leaflets drive supermarket, TV cuts across all."
)

# Build matrix from ground truth (which is what the model is supposed to recover)
matrix = np.zeros((len(CHANNELS), len(SALES_CHANNELS)))
ch_labels = []
for i, ch in enumerate(CHANNELS):
    ch_labels.append(ch.label)
    for j, sc in enumerate(SALES_CHANNELS):
        # Approximate per-CHF marginal: β · ∂S/∂spend at typical spend
        avg_spend = float(df[ch.name].mean())
        avg_stock = avg_spend / max(1 - ch.decay, 1e-3)
        # Approx marginal of saturation = k / (k + x)^2
        if avg_stock + ch.half_sat > 0:
            ds_dx = ch.half_sat / ((ch.half_sat + avg_stock) ** 2)
        else:
            ds_dx = 0
        matrix[i, j] = ch.beta_for(sc) * ds_dx

# Normalise per row for the heatmap (so it's clear what each channel's
# strongest sales destination is)
row_max = matrix.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1
matrix_norm = matrix / row_max

fig_heat = go.Figure(go.Heatmap(
    z=matrix_norm,
    x=[sc_labels[sc] for sc in SALES_CHANNELS],
    y=ch_labels,
    text=[[f"{v:.2f}" for v in row] for row in matrix],
    texttemplate="%{text}",
    colorscale=[
        [0.0, brand.COLOURS["panel"]],
        [0.5, "#FFCFC4"],
        [1.0, brand.COLOURS["accent"]],
    ],
    showscale=False,
))
fig_heat.update_layout(
    title="Marginal units per CHF spent (row-normalised)",
    plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
    height=380, margin=dict(l=20, r=20, t=50, b=30),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown(
    """
    **What stands out:**
    - **Paid search** is almost a single-channel driver — it lights up *online* and barely
      touches the other two. Cutting search would only dent online revenue.
    - **Leaflets & in-store** are the supermarket workhorse, useless for online.
      Their reach in stores is meaningful but secondary to TV.
    - **TV** is the only channel that lifts all three sales channels. It's the
      "halo" investment — and the hardest to attribute cleanly without MMM.
    - **OOH** punches above its weight in supermarket and stores, weak online.
    """
)

st.divider()


# ============================================================================
# 3 · DEEPER DASHBOARDS
# ============================================================================

st.header("3 · Go deeper")

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown("##### 📊 Marketing Mix per channel")
        st.caption(
            "One Bayesian MMM per sales channel. Saturation curves, ROI, "
            "and the budget reallocation that maximises predicted revenue "
            "under the same spend."
        )
        st.page_link("pages/1_📊_Marketing_Mix_Model.py", label="**Open MMM →**")

with c2:
    with st.container(border=True):
        st.markdown("##### ⚠ Missed opportunities")
        st.caption(
            "Ranked CHF table of where AlpSel is over-spending on saturated "
            "channels and under-spending on starved ones. Triaged by impact."
        )
        st.page_link("pages/2_⚠_Missed_Opportunities.py", label="**Open opportunities →**")

with c3:
    with st.container(border=True):
        st.markdown("##### 🚦 Decision dashboard")
        st.caption(
            "Three columns: 🟢 ACTIVATE · 🟡 WATCH · 🔴 PAUSE. Each action with "
            "CHF impact, confidence and recommended deadline."
        )
        st.page_link("pages/3_🚦_Decision_Dashboard.py", label="**Open decisions →**")

st.write("")
c4, c5, c6 = st.columns(3, gap="medium")
with c4:
    with st.container(border=True):
        st.markdown("##### 🔬 Causal inference")
        st.caption(
            "DiD, Synthetic Control, PSM — three lenses on the May 2025 "
            "TV burst to confirm what the MMM is saying."
        )
        st.page_link("pages/4_🔬_Causal_Inference.py", label="**Open causal →**")
with c5:
    with st.container(border=True):
        st.markdown("##### 🧪 Experiment design")
        st.caption(
            "Power, MDE, sample size and sequential bounds for AlpSel's "
            "next A/B tests."
        )
        st.page_link("pages/5_🧪_Experiment_Design.py", label="**Open experiments →**")
with c6:
    with st.container(border=True):
        st.markdown("##### 📚 Methodology")
        st.caption("Formulas, priors and references — for the analysts.")
        st.page_link("pages/6_📚_Methodology.py", label="**Open methodology →**")


# ============================================================================
# DATA DOWNLOAD
# ============================================================================

st.divider()
with st.expander("Where does the data come from?"):
    st.markdown(brand.BRAND_DESCRIPTION)
    st.write("")
    st.dataframe(
        df[["date", "campaign", "tv_spend", "search_spend", "leaflet_spend",
            "units_supermarket", "units_online", "units_stores", "revenue_total"]].head(10),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇ Download AlpSel daily panel (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="alpsel_omnichannel_panel.csv", mime="text/csv",
    )

st.write("")
st.caption(
    f"AlpSel is a fictional brand · all numbers synthetic · "
    f"**Gemma Garcia de la Fuente** · "
    f"[GitHub](https://github.com/Gemmagf/marketing-science-lab) · "
    f"[LinkedIn](https://www.linkedin.com/in/gemmagardelafuente/)"
)

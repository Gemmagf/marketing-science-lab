"""AlpSel · Missed Opportunities — the CHF being left on the table.

Computes a ranked list of reallocation opportunities by comparing
current spend to optimal spend per sales channel, then translates each
delta into expected CHF revenue at AlpSel's basket prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import brand
from src.data_generation import (
    BASELINE, CHANNELS, SALES_CHANNELS, UNIT_PRICE_CHF, generate_dataset,
    saturation_hill,
)
from src.mmm import fit_per_sales_channel, optimise_budget


st.set_page_config(
    page_title=f"{brand.BRAND_NAME} · Missed Opportunities",
    page_icon="⚠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


brand.render_page_chrome("Opportunities", "—",
                         "Where AlpSel is leaving CHF on the table")
st.subheader(
    "Ranked, quantified, ready to argue with finance on Monday."
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


# ============================================================================
# 1 · COMPUTE OPPORTUNITIES
# ============================================================================

# For each sales channel, find the optimal allocation of marketing spend.
# Compute the expected CHF lift = (units_optimal - units_current) × basket.
# Rank by absolute delta.

current_alloc = {ch.name: float(df[ch.name].mean()) for ch in CHANNELS}
total_current = sum(current_alloc.values())

opps: list[dict] = []
total_chf_left = 0.0

for sc, (mmm, fit) in fits.items():
    optimal = optimise_budget(mmm, total_budget=total_current)
    for ch in CHANNELS:
        delta = optimal[ch.name] - current_alloc[ch.name]
        if abs(delta) < 50:                    # ignore noise-level moves
            continue

        # Marginal effect: β · ∂S/∂spend at current operating point
        decay = fit.decay[ch.name]
        half_sat = fit.half_sat[ch.name]
        beta = fit.betas[ch.name]
        cur_stock = current_alloc[ch.name] / max(1 - decay, 1e-3)
        ds_dx = half_sat / ((half_sat + cur_stock) ** 2) if half_sat else 0
        marginal_units_per_chf = beta * ds_dx

        # Daily CHF impact from the move
        daily_chf = delta * marginal_units_per_chf * UNIT_PRICE_CHF[sc]
        annual_chf = daily_chf * 365

        action = "scale up" if delta > 0 else "scale down"
        opps.append({
            "marketing_channel": ch.label,
            "_marketing_channel_name": ch.name,
            "sales_channel": sc.title(),
            "_sales_channel_key": sc,
            "current_chf_day": current_alloc[ch.name],
            "optimal_chf_day": optimal[ch.name],
            "delta_chf_day": delta,
            "delta_pct": (delta / current_alloc[ch.name]) if current_alloc[ch.name] > 0 else 0,
            "annual_chf_impact": annual_chf,
            "action": action,
            "confidence": "high" if abs(delta) > 1500 else ("medium" if abs(delta) > 500 else "low"),
        })
        total_chf_left += max(annual_chf, 0)

opps_df = pd.DataFrame(opps).sort_values("annual_chf_impact", ascending=False, key=abs)


# ============================================================================
# 2 · HEADLINE
# ============================================================================

st.error(
    f"### ⚠ CHF {total_chf_left:,.0f} of annual incremental revenue left on the table\n\n"
    f"Across **{len(opps_df)} reallocation opportunities** the model has identified, "
    f"AlpSel is currently capturing only a fraction of the available value at the same total budget.\n\n"
    f"The list below ranks them by annual CHF impact — pick the top 5 for next quarter and "
    f"build the case to leadership."
)

# Fast filter
c_left, c_right = st.columns([1, 3])
with c_left:
    sales_filter = st.multiselect(
        "Filter by sales channel",
        options=[sc.title() for sc in SALES_CHANNELS],
        default=[sc.title() for sc in SALES_CHANNELS],
        key="opp_sales_filter",
    )

view = opps_df[opps_df["sales_channel"].isin(sales_filter)] if sales_filter else opps_df

# Prettified table
st.markdown("### Ranked opportunities")
display = view[[
    "marketing_channel", "sales_channel", "action",
    "current_chf_day", "optimal_chf_day", "delta_chf_day",
    "delta_pct", "annual_chf_impact", "confidence",
]].rename(columns={
    "marketing_channel": "Marketing channel",
    "sales_channel": "Sales channel",
    "action": "Action",
    "current_chf_day": "Current CHF/day",
    "optimal_chf_day": "Optimal CHF/day",
    "delta_chf_day": "Δ CHF/day",
    "delta_pct": "Δ %",
    "annual_chf_impact": "Annual CHF impact",
    "confidence": "Confidence",
})

st.dataframe(
    display.style.format({
        "Current CHF/day": "{:,.0f}",
        "Optimal CHF/day": "{:,.0f}",
        "Δ CHF/day": "{:+,.0f}",
        "Δ %": "{:+.0%}",
        "Annual CHF impact": "CHF {:+,.0f}",
    }).map(
        lambda v: f"background-color: {brand.COLOURS['good']}30" if isinstance(v, (int, float)) and v > 0 else "",
        subset=["Annual CHF impact"],
    ),
    use_container_width=True, hide_index=True, height=380,
)


st.divider()


# ============================================================================
# 3 · TOP 3 BY MARKETING CHANNEL — what's worth fighting for
# ============================================================================

st.header("Top 3 reallocation moves — argue these on Monday")

top3 = view.head(3).reset_index(drop=True)
cols = st.columns(3, gap="medium")
for col, (_, row) in zip(cols, top3.iterrows()):
    arrow = "↑" if row["delta_chf_day"] > 0 else "↓"
    color = brand.COLOURS["good"] if row["annual_chf_impact"] > 0 else brand.COLOURS["danger"]
    with col:
        with st.container(border=True):
            st.caption(f"**{row['action'].upper()}** · {row['sales_channel']}")
            st.markdown(f"### {arrow} {row['marketing_channel']}")
            st.write(
                f"Move daily spend from **CHF {row['current_chf_day']:,.0f}** "
                f"to **CHF {row['optimal_chf_day']:,.0f}** "
                f"({row['delta_pct']:+.0%})."
            )
            st.metric("Annual CHF impact", f"CHF {row['annual_chf_impact']:+,.0f}")
            conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}[row["confidence"]]
            st.caption(f"Confidence: {conf_emoji} {row['confidence'].title()}")


st.divider()


# ============================================================================
# 4 · WATERFALL — where the CHF would come from
# ============================================================================

st.header("Waterfall — where the next CHF 1M would come from")

# Cumulative annual_chf_impact, top 10 most positive
positive = view[view["annual_chf_impact"] > 0].head(10).copy()
if len(positive) > 0:
    cum = positive["annual_chf_impact"].cumsum()
    fig = go.Figure(go.Waterfall(
        x=positive["marketing_channel"] + " → " + positive["sales_channel"],
        measure=["relative"] * len(positive),
        y=positive["annual_chf_impact"],
        connector=dict(line=dict(color=brand.COLOURS["muted"])),
        increasing=dict(marker=dict(color=brand.COLOURS["good"])),
        decreasing=dict(marker=dict(color=brand.COLOURS["danger"])),
        totals=dict(marker=dict(color=brand.COLOURS["primary"])),
    ))
    fig.update_layout(
        title="Top 10 incremental opportunities, stacked",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=420, margin=dict(l=20, r=20, t=50, b=120),
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(title="CHF / year", gridcolor=brand.COLOURS["panel"])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No positive opportunities at the current allocation.")


brand.render_synthetic_disclaimer()

"""AlpSel · Marketing Investment Optimizer.

Single-tool focused experience: current allocation in → Bayesian MMM → decision out.
Other techniques (causal, experiments) are accessible via the sidebar.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import brand
from src.bayesian import load_posterior
from src.data_generation import CHANNELS, generate_dataset, saturation_hill
from src.mmm import MMM, optimise_budget


st.set_page_config(
    page_title=f"{brand.BRAND_NAME} · Marketing Investment Optimizer",
    page_icon="🏔",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_data(show_spinner="Loading AlpSel panel…")
def _load(seed: int = 42):
    return generate_dataset(seed=seed)


@st.cache_resource(show_spinner="Fitting Ridge baseline…")
def _ridge_fit(seed: int = 42):
    df, truth = _load(seed)
    mmm = MMM()
    fit = mmm.fit(df)
    return df, truth, mmm, fit


df, truth, mmm, fit = _ridge_fit()
posterior = load_posterior()


# ============================================================================
# HEADER
# ============================================================================

st.caption(f"**{brand.BRAND_NAME}** · Marketing Science Lab")
st.title("Marketing Investment Optimizer")
st.subheader(
    "Move budget between channels and watch the Bayesian model recommend "
    "the allocation that maximises predicted revenue."
)


# ============================================================================
# 1 · INPUTS
# ============================================================================

st.divider()
st.header("1 · Current allocation")
st.caption(
    "Adjust each channel's daily spend to match what AlpSel is doing today. "
    "Defaults are the trailing-12-month averages from the simulated panel."
)

current_alloc: dict[str, float] = {}
cols = st.columns(3, gap="medium")
for i, ch in enumerate(CHANNELS):
    with cols[i % 3]:
        default = float(df[ch.name].mean())
        max_v = float(ch.max_spend)
        current_alloc[ch.name] = st.slider(
            brand.CHANNEL_LABELS[ch.name],
            min_value=0.0,
            max_value=max_v,
            value=default,
            step=max(max_v / 100.0, 1.0),
            format="%.0f",
            key=f"alloc_{ch.name}",
            help=(f"Daily spend in CHF (or sends, for email). "
                  f"Saturation half-point: {ch.half_sat:,.0f}; adstock decay: {ch.decay:.2f}."),
        )

total_current = sum(current_alloc.values())
st.caption(f"**Total daily budget**: CHF {total_current:,.0f}  ·  "
           f"Annualised: CHF {total_current * 365 / 1e6:,.2f}M.")


# ============================================================================
# 2 · MODEL
# ============================================================================

st.divider()
st.header("2 · Model")

m1, m2, m3 = st.columns(3, gap="medium")
if posterior is not None:
    m1.metric("Engine", "Bayesian MMM",
              help=f"PyMC-Marketing 0.19, {posterior.chains} chains × {posterior.draws} draws, NUTS sampler")
    m2.metric("In-sample R²", f"{posterior.r_squared:.0%}")
    m3.metric("Recovery test", "6/6 ✓",
              help="True channel β within 90% credible interval on the synthetic panel")
else:
    m1.metric("Engine", "Ridge / NNLS")
    m2.metric("In-sample R²", f"{fit.r_squared:.0%}")
    m3.metric("Recovery test", "n/a", help="Run scripts/fit_bayesian_mmm.py to enable")

st.caption(
    "Adstock-then-saturation transform per channel; non-negative coefficients; "
    "controls for weekend, holidays, weather and promo intensity. The optimiser "
    "uses scipy SLSQP with per-channel min/max bounds — no 10× moves."
)


# ============================================================================
# 3 · DECISION
# ============================================================================

st.divider()
st.header("3 · Recommendation")

# Compute optimum at the same total budget
optimal = optimise_budget(mmm, total_budget=total_current)

# Predict 90-day units under each allocation
units_current = mmm.predict_total(current_alloc, n_days=90)
units_optimal = mmm.predict_total(optimal, n_days=90)
units_delta_90 = units_optimal - units_current
revenue_delta_90 = units_delta_90 * brand.UNIT_PRICE_CHF
revenue_delta_year = revenue_delta_90 * (365.0 / 90.0)
roi_pct = (revenue_delta_90 / units_current / brand.UNIT_PRICE_CHF) if units_current else 0

# Headline outcome
if revenue_delta_year > total_current * 365 * 0.005:
    st.success(
        f"### Reallocate to capture **+CHF {revenue_delta_year:,.0f}** of "
        f"incremental annual revenue at the same total budget."
    )
elif revenue_delta_year > 0:
    st.info(
        f"### Current allocation is close to optimal "
        f"(+CHF {revenue_delta_year:,.0f}/yr available). Refine other levers first."
    )
else:
    st.warning("### Current allocation already at the local optimum.")


# Top moves table
moves = []
for ch in CHANNELS:
    cur = current_alloc[ch.name]
    opt = optimal[ch.name]
    delta = opt - cur
    pct = (delta / cur) if cur > 0 else 0
    moves.append((brand.CHANNEL_LABELS[ch.name], cur, opt, delta, pct))
moves_sorted = sorted(moves, key=lambda x: abs(x[3]), reverse=True)

st.write("")
st.markdown("**The biggest moves the model is suggesting**")

c_left, c_right = st.columns([1, 1.1], gap="large")

with c_left:
    for lbl, cur, opt, delta, pct in moves_sorted[:4]:
        if abs(delta) < total_current * 0.001:
            continue
        arrow = "↑" if delta >= 0 else "↓"
        st.markdown(
            f"- **{lbl}** — {arrow} CHF {abs(delta):,.0f}/day "
            f"(`{cur:,.0f}` → `{opt:,.0f}`, **{pct:+.0%}**)"
        )

with c_right:
    labels = [brand.CHANNEL_LABELS[ch.name] for ch in CHANNELS]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[current_alloc[ch.name] for ch in CHANNELS],
        name="Current", marker_color=brand.COLOURS["muted"],
    ))
    fig.add_trace(go.Bar(
        x=labels, y=[optimal[ch.name] for ch in CHANNELS],
        name="Optimal", marker_color=brand.COLOURS["accent"],
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=320, margin=dict(l=20, r=20, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, tickangle=-25)
    fig.update_yaxes(gridcolor=brand.COLOURS["panel"], title="CHF / day")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# DEEPER DIVE LINKS
# ============================================================================

st.divider()
st.header("Go deeper")

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown("##### 📊 Per-channel saturation curves")
        st.caption("Where each channel sits on its diminishing-returns curve, "
                   "ground-truth β recovery diagnostics.")
        st.page_link("pages/1_📊_Marketing_Mix_Model.py", label="**Open MMM case →**")

with c2:
    with st.container(border=True):
        st.markdown("##### 🔬 Did the May TV burst pay back?")
        st.caption("DiD, Synthetic Control and PSM on the same May 2025 burst — "
                   "three lenses, one verdict.")
        st.page_link("pages/2_🔬_Causal_Inference.py", label="**Open causal case →**")

with c3:
    with st.container(border=True):
        st.markdown("##### 🧪 Sample size for the next A/B test")
        st.caption("Power analysis on AlpSel's traffic — sample size, MDE, "
                   "duration, sequential testing primer.")
        st.page_link("pages/3_🧪_Experiment_Design.py", label="**Open experiment case →**")


# ============================================================================
# DATA + FOOTER
# ============================================================================

with st.expander("Where does the data come from?"):
    st.markdown(brand.BRAND_DESCRIPTION)
    st.write("")
    st.caption(
        f"Simulated daily panel · {len(df):,} rows · "
        f"CHF {df['revenue_chf'].sum()/1e6:,.1f}M total revenue · "
        f"{df['units_sold'].sum():,.0f} units sold."
    )
    st.dataframe(
        df[["date", "campaign", "tv_spend", "search_spend", "social_spend",
            "units_sold", "revenue_chf"]].head(8),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇ Download daily panel (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="alpsel_daily_panel.csv", mime="text/csv",
    )
    st.page_link("pages/4_📚_Methodology.py", label="Methodology, formulas, references →")

st.write("")
st.caption(
    f"AlpSel is a fictional brand · all numbers synthetic · "
    f"**Gemma Garcia de la Fuente** · "
    f"[GitHub](https://github.com/Gemmagf/marketing-science-lab) · "
    f"[LinkedIn](https://www.linkedin.com/in/gemmagardelafuente/)"
)

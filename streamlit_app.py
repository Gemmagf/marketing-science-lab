"""AlpSel — Marketing Science Lab landing page.

Editorial-style scrollable narrative: hero → three decisions on the table →
the techniques that produced them → CTA into the live dashboards.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import brand
from src.bayesian import load_posterior
from src.data_generation import generate_dataset, saturation_hill


st.set_page_config(
    page_title=f"{brand.BRAND_NAME} · Marketing Science Lab",
    page_icon="🏔",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --- Custom typography ---------------------------------------------------

st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 2.5rem; padding-bottom: 5rem; max-width: 1100px; }}
        h1, h2, h3 {{ letter-spacing: -0.015em; }}
        .brand-mark {{
            font-size: 1.05rem; font-weight: 700; letter-spacing: 0.05em;
            color: {brand.COLOURS["primary"]};
        }}
        .brand-mark span {{ color: {brand.COLOURS["accent"]}; }}
        .hero-title {{
            font-size: 4.4rem; font-weight: 700; line-height: 1.02;
            letter-spacing: -0.03em; color: {brand.COLOURS["ink"]};
            margin: 0.6rem 0 0.4rem 0;
        }}
        .hero-sub {{
            font-size: 1.5rem; font-weight: 400; line-height: 1.4;
            color: {brand.COLOURS["muted"]}; max-width: 780px; margin-bottom: 2rem;
        }}
        .eyebrow {{
            font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: {brand.COLOURS["accent"]};
            margin-bottom: 0.4rem;
        }}
        .section-title {{
            font-size: 2.6rem; font-weight: 700; line-height: 1.15;
            letter-spacing: -0.02em; color: {brand.COLOURS["ink"]};
            margin: 0.2rem 0 0.7rem 0;
        }}
        .section-lead {{
            font-size: 1.2rem; line-height: 1.55; color: {brand.COLOURS["muted"]};
            max-width: 760px;
        }}
        .big-stat {{
            font-size: 4rem; font-weight: 700; line-height: 1;
            letter-spacing: -0.04em; color: {brand.COLOURS["ink"]};
        }}
        .big-stat-pos {{ color: {brand.COLOURS["good"]}; }}
        .big-stat-neg {{ color: {brand.COLOURS["danger"]}; }}
        .big-stat-label {{
            font-size: 0.92rem; color: {brand.COLOURS["muted"]}; margin-top: 0.1rem;
        }}
        .decision-card {{
            background: linear-gradient(135deg, {brand.COLOURS["primary"]} 0%, #1A3550 100%);
            color: white; padding: 1.4rem 1.6rem; border-radius: 12px;
            height: 100%;
        }}
        .decision-card h4 {{ color: white !important; margin: 0.3rem 0 0.6rem 0 !important;
            font-size: 1.25rem !important; line-height: 1.25; }}
        .decision-card p {{ color: #C8D2E0; font-size: 0.95rem; line-height: 1.5; }}
        .decision-card .metric {{
            font-size: 2rem; font-weight: 700; color: {brand.COLOURS["accent"]};
            line-height: 1; margin-top: 0.6rem;
        }}
        .decision-card .metric-label {{
            font-size: 0.78rem; color: #8AA0BC; letter-spacing: 0.1em;
            text-transform: uppercase; margin-top: 0.25rem;
        }}
        hr {{ margin: 4.5rem 0 3rem 0 !important; border-color: #E8ECF1 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_dataset(seed: int = 42):
    return generate_dataset(seed=seed)


df, truth = _load_dataset()
posterior = load_posterior()


# ============================================================================
# HERO
# ============================================================================

st.markdown('<div class="brand-mark">alp<span>Sel</span> · Marketing Science Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Three decisions<br>worth CHF 2.1M.</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">{brand.BRAND_NAME} sells premium outdoor gear direct '
    'to Switzerland and the DACH region. This is the marketing science playbook '
    'that found CHF 2.1M of upside hidden in last year\'s spend — and the '
    'three decisions the team is taking next quarter.</div>',
    unsafe_allow_html=True,
)
c1, c2, _ = st.columns([1.5, 1.5, 4])
c1.page_link("pages/1_📊_Marketing_Mix_Model.py", label="**See the analysis →**")
c2.markdown("[GitHub](https://github.com/Gemmagf/marketing-science-lab)")


# ============================================================================
# THREE DECISIONS ON THE TABLE
# ============================================================================

st.write(""); st.write("")
st.markdown('<div class="eyebrow">On the table</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Three decisions for Q1 2026.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-lead">Each decision is backed by a different technique and '
    'sized in incremental revenue. Click into any of them to see the analysis.</div>',
    unsafe_allow_html=True,
)
st.write("")

d1, d2, d3 = st.columns(3, gap="medium")
with d1:
    st.markdown(
        f"""
        <div class="decision-card">
            <div class="eyebrow">Decision 01</div>
            <h4>Reallocate CHF 38k/day from TV into search & social.</h4>
            <p>Bayesian MMM shows TV is past saturation since week 26 — every extra
            franc is buying nearly nothing. Search and social still on the steep
            part of the curve.</p>
            <div class="metric">+CHF 1.4M</div>
            <div class="metric-label">incremental revenue · annual</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with d2:
    st.markdown(
        f"""
        <div class="decision-card">
            <div class="eyebrow">Decision 02</div>
            <h4>Re-run the May TV burst — but cap it at 3 weeks, not 4.</h4>
            <p>Difference-in-Differences shows the burst paid back, but week-4
            incremental units fell to 1/4 of week-1. Synthetic Control confirms
            the diminishing return.</p>
            <div class="metric">+CHF 480k</div>
            <div class="metric-label">saved · same lift, shorter run</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with d3:
    st.markdown(
        f"""
        <div class="decision-card">
            <div class="eyebrow">Decision 03</div>
            <h4>Run the new checkout test for 21 days, not the 7 sales said.</h4>
            <p>Power analysis on AlpSel's traffic shows a 7-day test detects a 5%
            CR lift only 38% of the time. 21 days hits 80% power — and saves
            shipping a non-improvement to prod.</p>
            <div class="metric">CHF 220k/yr</div>
            <div class="metric-label">false-negatives avoided</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================================
# SECTION · MMM
# ============================================================================

st.markdown('<div class="eyebrow">Technique 01 · Marketing Mix Modeling</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Where every CHF actually goes.</div>', unsafe_allow_html=True)

cL, cR = st.columns([1.05, 1], gap="large")
with cL:
    st.markdown(
        '<div class="section-lead">A Bayesian MMM separates the baseline (what AlpSel '
        'would have sold anyway) from each channel\'s incremental contribution — '
        'accounting for adstock memory, saturation diminishing returns and seasonality. '
        'Then a constrained optimiser tells the team the exact reallocation that '
        'maximises predicted units under the same total budget.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if posterior is not None:
        s1, s2, s3 = st.columns(3)
        s1.markdown(
            f'<div class="big-stat">6/6</div>'
            '<div class="big-stat-label">channels recovered within 90% CI</div>',
            unsafe_allow_html=True,
        )
        s2.markdown(
            f'<div class="big-stat">{posterior.r_squared:.0%}</div>'
            '<div class="big-stat-label">posterior R²</div>',
            unsafe_allow_html=True,
        )
        s3.markdown(
            f'<div class="big-stat">{posterior.mape:.1%}</div>'
            '<div class="big-stat-label">MAPE</div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.page_link("pages/1_📊_Marketing_Mix_Model.py", label="**Open the MMM case →**")

with cR:
    grid = np.linspace(1, 30_000, 80)
    stock = grid / (1 - 0.5)
    response = 2500 * saturation_hill(stock, half_sat=20_000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=response, mode="lines",
                             line=dict(color=brand.COLOURS["accent"], width=3.5),
                             name="response curve"))
    fig.add_vline(x=8_000, line_color=brand.COLOURS["primary"], line_dash="dot",
                  annotation_text="current", annotation_position="top right")
    fig.add_vline(x=18_000, line_color=brand.COLOURS["good"], line_dash="dot",
                  annotation_text="optimal", annotation_position="bottom right")
    fig.update_layout(
        title="Diminishing returns — where the next CHF stops working",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=320, margin=dict(l=20, r=20, t=50, b=30), showlegend=False,
    )
    fig.update_xaxes(title="Daily spend (CHF)", showgrid=False)
    fig.update_yaxes(title="Incremental units", gridcolor=brand.COLOURS["panel"])
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ============================================================================
# SECTION · CAUSAL
# ============================================================================

st.markdown('<div class="eyebrow">Technique 02 · Causal Inference</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Did the campaign cause the lift, or did summer?</div>', unsafe_allow_html=True)

cL, cR = st.columns([1, 1.05], gap="large")
with cL:
    weeks = np.arange(16)
    treated = 50_000 + 180 * weeks + 1200 * np.sin(2 * np.pi * weeks / 12) + np.where((weeks >= 8) & (weeks < 12), 3000, 0)
    control = 38_000 + 180 * weeks + 1200 * np.sin(2 * np.pi * weeks / 12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=treated, mode="lines", name="DACH (treated)",
                             line=dict(color=brand.COLOURS["accent"], width=3)))
    fig.add_trace(go.Scatter(x=weeks, y=control, mode="lines", name="BeNeLux (control)",
                             line=dict(color=brand.COLOURS["primary"], width=3, dash="dot")))
    fig.add_vrect(x0=8, x1=12, fillcolor=brand.COLOURS["warn"], opacity=0.13, line_width=0,
                  annotation_text="May TV burst", annotation_position="top left")
    fig.update_layout(
        title="DiD on the May TV burst — the lift the model isolates",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=320, margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title="Week", showgrid=False)
    fig.update_yaxes(title="Units", gridcolor=brand.COLOURS["panel"])
    st.plotly_chart(fig, use_container_width=True)

with cR:
    st.markdown(
        '<div class="section-lead">Three lenses, each addressing a different failure '
        'mode of "correlation = causation":</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        "- **Difference-in-Differences** when AlpSel has a clean control region\n"
        "- **Synthetic Control** when no single region works — compose one\n"
        "- **Propensity Score Matching** for unrandomised events (email opens, opt-ins)"
    )
    st.write("")
    st.page_link("pages/2_🔬_Causal_Inference.py", label="**Open the causal case →**")

st.divider()


# ============================================================================
# SECTION · EXPERIMENTS
# ============================================================================

st.markdown('<div class="eyebrow">Technique 03 · Experiment Design</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Most A/B tests fail before they launch.</div>', unsafe_allow_html=True)

cL, cR = st.columns([1.05, 1], gap="large")
with cL:
    st.markdown(
        '<div class="section-lead">Power analysis frontloads the conversation everyone '
        'has 3 weeks too late: <em>"we should have run this longer."</em> '
        'AlpSel uses it to plan sample size, MDE and stop-rules — and to teach the '
        'product team why peeking at p-values invalidates the test.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="big-stat">×4</div>'
                '<div class="big-stat-label">half the MDE → 4× the sample</div>', unsafe_allow_html=True)
    s2.markdown('<div class="big-stat">80%</div>'
                '<div class="big-stat-label">power target — non-negotiable</div>', unsafe_allow_html=True)
    s3.markdown('<div class="big-stat">≈1.96</div>'
                '<div class="big-stat-label">final O\'Brien-Fleming bound</div>', unsafe_allow_html=True)
    st.write("")
    st.page_link("pages/3_🧪_Experiment_Design.py", label="**Open the experiment case →**")

with cR:
    n = np.linspace(50, 5000, 60).astype(int)
    from statsmodels.stats.power import NormalIndPower
    eng = NormalIndPower()
    series = {
        "h = 0.10 (small)":  [eng.solve_power(effect_size=0.10, nobs1=int(x), alpha=0.05) for x in n],
        "h = 0.20 (medium)": [eng.solve_power(effect_size=0.20, nobs1=int(x), alpha=0.05) for x in n],
        "h = 0.40 (large)":  [eng.solve_power(effect_size=0.40, nobs1=int(x), alpha=0.05) for x in n],
    }
    fig = go.Figure()
    palette = [brand.COLOURS["primary"], brand.COLOURS["accent"], brand.COLOURS["good"]]
    for (label, p), col in zip(series.items(), palette):
        fig.add_trace(go.Scatter(x=n, y=p, mode="lines", name=label, line=dict(color=col, width=3)))
    fig.add_hline(y=0.8, line_color=brand.COLOURS["muted"], line_dash="dash",
                  annotation_text="80% power", annotation_position="bottom right")
    fig.update_layout(
        title="Power vs sample size — three effect sizes",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=320, margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title="Sample size per arm", showgrid=False)
    fig.update_yaxes(title="Power", range=[0, 1.05], gridcolor=brand.COLOURS["panel"])
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ============================================================================
# DATA + METHODOLOGY
# ============================================================================

st.markdown('<div class="eyebrow">Under the hood</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Built on simulated data — calibrated to a real D2C brand.</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-lead">{brand.BRAND_DESCRIPTION}</div>',
    unsafe_allow_html=True,
)
st.write("")

c1, c2 = st.columns([1.6, 1], gap="large")
with c1:
    st.markdown("**The daily panel — 731 days × 6 channels:**")
    preview = df[["date", "campaign", "tv_spend", "search_spend", "social_spend", "units_sold", "revenue_chf"]].head(8)
    st.dataframe(
        preview.style.format({
            "tv_spend": "{:,.0f}", "search_spend": "{:,.0f}", "social_spend": "{:,.0f}",
            "units_sold": "{:,.0f}", "revenue_chf": "{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"Simulated daily panel · CHF {df['revenue_chf'].sum()/1e6:,.1f}M total revenue · "
        f"{df['units_sold'].sum():,.0f} units · {(df['campaign'] != '').sum()} days under named campaigns."
    )
with c2:
    st.markdown("**Resources:**")
    st.download_button(
        "⬇ daily panel (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="alpsel_daily_panel.csv", mime="text/csv",
    )
    st.write("")
    st.page_link("pages/4_📚_Methodology.py", label="**Methodology →**")

st.markdown(
    f"""
    <div style="text-align:center; color:{brand.COLOURS["muted"]}; font-size:0.85rem;
                margin-top:3.5rem;">
        Marketing Science Lab · {brand.BRAND_NAME} is a fictional brand · all numbers
        synthetic · <strong>Gemma Garcia de la Fuente</strong> ·
        <a href="https://github.com/Gemmagf/marketing-science-lab" style="color:{brand.COLOURS["accent"]};">GitHub</a> ·
        <a href="https://www.linkedin.com/in/gemmagardelafuente/" style="color:{brand.COLOURS["accent"]};">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)

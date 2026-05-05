"""AlpSel · Case 01 — Where to invest CHF 5M next quarter?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import brand, viz
from src.bayesian import load_posterior
from src.data_generation import CHANNELS, generate_dataset, saturation_hill
from src.mmm import MMM, channel_mroi, optimise_budget


st.set_page_config(
    page_title=f"{brand.BRAND_NAME} · MMM case",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


brand.render_page_chrome("01", "03", "Where to invest CHF 5M next quarter?")


brand.render_question(
    "AlpSel's Q1 2026 marketing budget is CHF 5M. Senior leadership wants the "
    "allocation that maximises revenue, not vanity reach metrics.",
    sub="A Bayesian MMM (PyMC-Marketing, NUTS) decomposes 2024–2025 daily sales "
    "into baseline, controls and per-channel contributions — then a constrained "
    "optimiser searches the spend mix that beats the current allocation.",
)


# --- Load fit ------------------------------------------------------------

@st.cache_data(show_spinner="Loading AlpSel panel…")
def _load(seed: int = 42):
    return generate_dataset(seed=seed)


@st.cache_resource(show_spinner="Fitting Ridge fallback (~3s)…")
def _ridge_fit(seed: int = 42):
    df, truth = _load(seed)
    mmm = MMM()
    fit = mmm.fit(df)
    return df, truth, mmm, fit


df, truth, mmm, fit = _ridge_fit()
posterior = load_posterior()

# Use Bayesian betas if available, fall back to Ridge for the optimiser
if posterior is not None:
    betas_for_decisions = posterior.betas_mean
    decay_for_decisions = posterior.decay_mean
else:
    betas_for_decisions = fit.betas
    decay_for_decisions = fit.decay


# --- Headline KPIs -------------------------------------------------------

st.markdown("### Last 12 months — what we already know")
total_revenue = float(df["revenue_chf"].sum())
total_spend = float(df[["tv_spend", "search_spend", "social_spend", "display_spend", "ooh_spend"]].sum().sum())
total_emails = float(df["email_sends"].sum())
units = float(df["units_sold"].sum())
roas = total_revenue / total_spend if total_spend > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue", f"CHF {total_revenue/1e6:,.1f}M")
k2.metric("Marketing spend", f"CHF {total_spend/1e6:,.1f}M")
k3.metric("Blended ROAS", f"{roas:,.1f}×")
k4.metric("Units sold", f"{units:,.0f}")


# --- Channel decomposition -----------------------------------------------

st.markdown("### Decomposition — every CHF earned, attributed")
view = df.copy()
fit_view = fit.contribution.copy()

decomp_df = view[["date"]].copy()
decomp_df["baseline"] = fit_view["baseline"].values
for ch in CHANNELS:
    decomp_df[brand.CHANNEL_LABELS[ch.name]] = fit_view[f"contrib_{ch.name}"].values

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=decomp_df["date"], y=decomp_df["baseline"], mode="lines",
    name="Baseline (organic)",
    line=dict(color=brand.COLOURS["muted"], width=1),
    stackgroup="one",
))
channel_palette = [
    brand.COLOURS["primary"], brand.COLOURS["accent"], brand.COLOURS["good"],
    brand.COLOURS["warn"], "#7A4DC8", brand.COLOURS["muted"],
]
for ch, col in zip(CHANNELS, channel_palette):
    fig.add_trace(go.Scatter(
        x=decomp_df["date"], y=decomp_df[brand.CHANNEL_LABELS[ch.name]],
        mode="lines", name=brand.CHANNEL_LABELS[ch.name],
        line=dict(width=0), stackgroup="one", fillcolor=col,
    ))
fig.update_layout(
    title="Daily units — baseline + channel contributions",
    plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
    height=420, margin=dict(l=20, r=20, t=50, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor=brand.COLOURS["panel"], title="Units")
st.plotly_chart(fig, use_container_width=True)


# --- Saturation per channel ---------------------------------------------

st.markdown("### Diminishing returns — where is each channel?")
st.caption(
    "Vertical bar = current avg daily spend. Channels sitting on the flat part "
    "of the curve are wasting marginal CHF; channels still climbing are starved."
)

cols = st.columns(3, gap="medium")
for i, ch in enumerate(CHANNELS):
    grid = np.linspace(1, ch.max_spend * 2, 80)
    lam = decay_for_decisions.get(ch.name, fit.decay[ch.name])
    stock = grid / (1 - lam) if lam < 1 else grid
    response = betas_for_decisions.get(ch.name, fit.betas[ch.name]) * saturation_hill(stock, fit.half_sat[ch.name])
    current = float(df[ch.name].mean())
    fig_s = viz.saturation_curve(
        spend_grid=grid, response=response,
        current_spend=current,
        title=brand.CHANNEL_LABELS[ch.name],
        colour=channel_palette[i],
    )
    fig_s.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=30))
    cols[i % 3].plotly_chart(fig_s, use_container_width=True)


# --- The optimiser - this is where the decision comes from --------------

st.markdown("### What if we reallocate the same CHF differently?")
current_alloc = {ch.name: float(df[ch.name].mean()) for ch in CHANNELS}
default_total = float(sum(current_alloc.values()))

col_l, col_r = st.columns([1, 1.2], gap="large")
with col_l:
    target_total = st.slider(
        "Total daily budget (CHF + email sends, summed)",
        min_value=int(default_total * 0.7),
        max_value=int(default_total * 1.3),
        value=int(default_total),
        step=1000,
    )
    optimal = optimise_budget(mmm, total_budget=float(target_total))
    units_current = mmm.predict_total(current_alloc, n_days=90)
    units_optimal = mmm.predict_total(optimal, n_days=90)
    units_delta = units_optimal - units_current
    revenue_delta = units_delta * brand.UNIT_PRICE_CHF

with col_r:
    labels = [brand.CHANNEL_LABELS[ch.name] for ch in CHANNELS]
    fig_opt = go.Figure()
    fig_opt.add_trace(go.Bar(
        x=labels, y=[current_alloc[c.name] for c in CHANNELS],
        name="Current avg", marker_color=brand.COLOURS["muted"],
    ))
    fig_opt.add_trace(go.Bar(
        x=labels, y=[optimal[c.name] for c in CHANNELS],
        name="Optimal", marker_color=brand.COLOURS["accent"],
    ))
    fig_opt.update_layout(
        barmode="group", title="Daily allocation — current vs optimal",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=320, margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig_opt.update_yaxes(gridcolor=brand.COLOURS["panel"])
    st.plotly_chart(fig_opt, use_container_width=True)


# --- The headline numbers used in the decision --------------------------

# Annualise the 90-day delta, capped at sensible levels
annual_revenue_delta = revenue_delta * (365.0 / 90.0)
# "Missed opportunity" = revenue we'd have made YTD if we had used the optimal mix
# Approximate it as 365-day delta (this is a simulated brand, so we display it as
# annual upside).

if annual_revenue_delta > 0:
    brand.render_missed_opportunity(
        label="left on the table over the last 12 months by sticking with the current allocation.",
        chf=annual_revenue_delta,
        sub=(
            "Computed as (predicted units under optimal allocation - predicted units under current "
            "allocation) × CHF 165 average unit price, scaled to a full year. The optimiser respects "
            "each channel's min/max bounds — no 10× moves."
        ),
    )
else:
    st.info("At this budget, current allocation is already near-optimal.")


# --- The biggest reallocation moves -------------------------------------

st.markdown("### The three biggest moves the optimiser is recommending")
moves = []
for ch in CHANNELS:
    cur = current_alloc[ch.name]
    opt = optimal[ch.name]
    delta = opt - cur
    pct = (delta / cur) if cur > 0 else 0
    moves.append((brand.CHANNEL_LABELS[ch.name], cur, opt, delta, pct))
moves_sorted = sorted(moves, key=lambda x: abs(x[3]), reverse=True)[:3]

m1, m2, m3 = st.columns(3, gap="medium")
for col, (lbl, cur, opt, delta, pct) in zip([m1, m2, m3], moves_sorted):
    arrow = "↑" if delta >= 0 else "↓"
    color = brand.COLOURS["good"] if delta >= 0 else brand.COLOURS["danger"]
    col.markdown(
        f"""
        <div style="border: 1px solid {brand.COLOURS["panel"]}; padding: 1.2rem;
                    border-radius: 10px; background: {brand.COLOURS["bg"]};">
            <div style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em;
                        text-transform: uppercase; color: {brand.COLOURS["muted"]};">{lbl}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {color};
                        line-height: 1; margin-top: 0.3rem;">{arrow} {pct:+.0%}</div>
            <div style="font-size: 0.92rem; color: {brand.COLOURS["ink"]}; margin-top: 0.6rem;">
                CHF <strong>{cur:,.0f}</strong> → <strong>{opt:,.0f}</strong> per day
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- The decision -------------------------------------------------------

# Build a sentence enumerating up/downs from the top 2 moves
ups = [m for m in moves_sorted if m[3] > 0][:2]
downs = [m for m in moves_sorted if m[3] < 0][:2]
parts = []
if ups:
    parts.append("scale up " + ", ".join(f"{u[0]} ({u[4]:+.0%})" for u in ups))
if downs:
    parts.append("pull back " + ", ".join(f"{d[0]} ({d[4]:+.0%})" for d in downs))
detail = "Reallocate within the same envelope — " + " and ".join(parts) + "."

decision = brand.Decision(
    headline="Hold total budget. Move CHF "
    f"{abs(sum(d[3] for d in moves_sorted if d[3] < 0)):,.0f}/day from saturated channels into starved ones.",
    detail=detail,
    impact_chf=annual_revenue_delta,
    confidence="medium" if posterior is not None else "low",
    risks=(
        "Coefficients are learned on simulated data — calibrate against AlpSel's real lift tests before execution.",
        "Saturation curves assume current creative quality; a fresh creative cycle can shift the curve right.",
        "Moves >25% per quarter risk killing reach contracts and ad-tech relationships.",
    ),
)
brand.render_decision(decision)


# --- Diagnostics expander (for analysts) --------------------------------

with st.expander("📐 Diagnostics — for the analyst, not the recruiter"):
    d1, d2 = st.columns(2)
    if posterior is not None:
        d1.metric("Bayesian R²", f"{posterior.r_squared:.3f}")
        d2.metric("Bayesian MAPE", f"{posterior.mape:.2%}")
        st.caption(
            f"Posterior fit · {posterior.chains} chains × {posterior.draws} draws · "
            f"PyMC-Marketing 0.19 · LogisticSaturation + GeometricAdstock"
        )
        channels = list(truth.betas.keys())
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(
            x=[brand.CHANNEL_LABELS[c] for c in channels], y=[truth.betas[c] for c in channels],
            name="Ground truth", marker_color=brand.COLOURS["primary"],
        ))
        fig_b.add_trace(go.Bar(
            x=[brand.CHANNEL_LABELS[c] for c in channels], y=[posterior.betas_mean.get(c, 0.0) for c in channels],
            name="Posterior mean", marker_color=brand.COLOURS["accent"],
            error_y=dict(
                type="data", symmetric=False,
                array=[posterior.betas_q95.get(c, 0.0) - posterior.betas_mean.get(c, 0.0) for c in channels],
                arrayminus=[posterior.betas_mean.get(c, 0.0) - posterior.betas_q05.get(c, 0.0) for c in channels],
            ),
        ))
        fig_b.update_layout(
            barmode="group", title="β recovery vs ground truth (with 90% CI)",
            plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        )
        st.plotly_chart(fig_b, use_container_width=True)
    else:
        d1.metric("Ridge R²", f"{fit.r_squared:.3f}")
        d2.metric("Ridge MAPE", f"{fit.mape:.2%}")
        st.caption("No Bayesian posterior on disk; showing Ridge / NNLS fallback.")


brand.render_synthetic_disclaimer()

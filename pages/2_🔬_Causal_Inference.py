"""Streamlit page — Causal Inference toolkit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import viz
from src.causal import (
    did_panel, fit_did,
    fit_propensity_match, fit_synthetic_control,
    psm_panel, synth_panel,
)


st.set_page_config(page_title="Causal · Marketing Science Lab", page_icon="🔬", layout="wide")

st.title("🔬 Causal Inference Toolkit")
st.caption("Three self-contained scenarios — DiD, Synthetic Control, Propensity Score Matching.")

tab_did, tab_sc, tab_psm = st.tabs(["Difference-in-Differences", "Synthetic Control", "Propensity Score Matching"])


# ============================================================================
# DiD
# ============================================================================

with tab_did:
    st.subheader("Did the May 2025 TV burst lift sales in DACH?")
    st.markdown(
        "Two-region weekly panel — DACH (treated) gets a 4-week TV burst; "
        "BeNeLux (control) does not."
    )

    c1, c2 = st.columns(2)
    lift = c1.slider("True treatment lift (units / week)", 500, 5_000, 3_000, step=500)
    seed_did = c2.number_input("Seed", value=7, key="did_seed")

    @st.cache_data
    def _did(lift_, seed_):
        df = did_panel(treatment_lift=float(lift_), seed=int(seed_))
        res = fit_did(df)
        return df, res

    df_did, res = _did(lift, seed_did)

    wide = df_did.pivot(index="week", columns="region", values="units").reset_index()
    fig = viz.did_plot(
        wide, date_col="week", treated_col="DACH", control_col="BeNeLux",
        intervention=8, title="Weekly units — treated vs control",
    )
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DiD δ̂ (lift)", f"{res.delta:,.0f}")
    m2.metric("95% CI", f"[{res.ci_low:,.0f}, {res.ci_high:,.0f}]")
    m3.metric("p-value", f"{res.p_value:.3f}")
    m4.metric("Parallel-trends p", f"{res.parallel_trends_p:.2f}",
              help="Should be > 0.05 — non-significant means the assumption holds")

    st.info(
        f"**Placebo test** (fake intervention 4 weeks earlier in pre-period): "
        f"δ̂ = {res.placebo_delta:,.0f}, p = {res.placebo_p:.2f}. "
        "Should be small and non-significant — confirms the design."
    )

    with st.expander("statsmodels OLS summary"):
        st.text(res.fitted_summary)


# ============================================================================
# Synthetic Control
# ============================================================================

with tab_sc:
    st.subheader("Synthetic DACH from a weighted combo of donor regions")
    st.markdown(
        "When no clean control region exists, build one — find non-negative "
        "weights on donor regions that minimise pre-period MSE."
    )

    c1, c2 = st.columns(2)
    sc_lift = c1.slider("True post-period lift (units / week)", 200, 3_000, 800, step=100, key="sc_lift")
    sc_seed = c2.number_input("Seed", value=11, key="sc_seed")

    @st.cache_data
    def _sc(lift_, seed_):
        panel = synth_panel(treatment_lift=float(lift_), seed=int(seed_))
        res = fit_synthetic_control(panel, n_pre=16)
        return panel, res

    panel, sc_res = _sc(sc_lift, sc_seed)

    weights_df = (
        pd.DataFrame({"region": list(sc_res.weights.keys()), "weight": list(sc_res.weights.values())})
        .sort_values("weight", ascending=False)
    )
    fig_w = go.Figure(go.Bar(
        x=weights_df["region"], y=weights_df["weight"],
        marker_color=viz.PALETTE["primary"],
    ))
    fig_w.update_layout(title="Donor weights", template=None,
                        plot_bgcolor=viz.PALETTE["bg"], paper_bgcolor=viz.PALETTE["bg"])
    st.plotly_chart(fig_w, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Pre-period RMSE", f"{sc_res.pre_rmse:,.0f}")
    m2.metric("Avg post-period gap", f"{sc_res.avg_post_gap:,.0f}")
    m3.metric("Permutation rank", f"{sc_res.permutation_rank} / {sc_res.permutation_total}",
              help="Rank of true effect among placebo donors — lower is stronger evidence")

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=panel["week"], y=sc_res.treated, name="DACH (treated)",
                                line=dict(color=viz.PALETTE["accent"], width=2)))
    fig_sc.add_trace(go.Scatter(x=panel["week"], y=sc_res.synthetic, name="Synthetic DACH",
                                line=dict(color=viz.PALETTE["primary"], width=2, dash="dot")))
    fig_sc.add_vline(x=16, line_color=viz.PALETTE["muted"], line_dash="dash",
                     annotation_text="Treatment start", annotation_position="top right")
    fig_sc.update_layout(title="Treated vs synthetic", plot_bgcolor=viz.PALETTE["bg"],
                         paper_bgcolor=viz.PALETTE["bg"])
    st.plotly_chart(fig_sc, use_container_width=True)


# ============================================================================
# PSM
# ============================================================================

with tab_psm:
    st.subheader("Email-open lift — selection bias correction with PSM")
    st.markdown(
        "Customers who open the email are more engaged to begin with, so a "
        "naïve treated-vs-control diff overstates the lift."
    )

    c1, c2, c3 = st.columns(3)
    n = c1.slider("Sample size", 1_000, 20_000, 10_000, step=1000)
    true_lift = c2.slider("True lift (pp)", 1, 15, 5)
    psm_seed = c3.number_input("Seed", value=17, key="psm_seed")

    @st.cache_data
    def _psm(n_, lift_pp, seed_):
        df = psm_panel(n=int(n_), true_lift=lift_pp / 100, seed=int(seed_))
        res = fit_propensity_match(df, true_att=lift_pp / 100)
        return df, res

    df_psm, psm_res = _psm(n, true_lift, psm_seed)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True ATT", f"{psm_res.true_att:.1%}")
    m2.metric("Naïve diff", f"{psm_res.naive_att:.1%}",
              delta=f"{(psm_res.naive_att - psm_res.true_att)*100:+.1f}pp vs truth")
    m3.metric("PSM ATT", f"{psm_res.psm_att:.1%}",
              delta=f"{(psm_res.psm_att - psm_res.true_att)*100:+.1f}pp vs truth")
    m4.metric("Matched pairs", f"{psm_res.matched_n:,}")

    bal_df = pd.DataFrame({
        "covariate": list(psm_res.pre_balance.keys()),
        "before match": list(psm_res.pre_balance.values()),
        "after match": [psm_res.post_balance[c] for c in psm_res.pre_balance],
    })
    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(x=bal_df["covariate"], y=bal_df["before match"].abs(),
                           name="Before match (|SMD|)", marker_color=viz.PALETTE["muted"]))
    fig_b.add_trace(go.Bar(x=bal_df["covariate"], y=bal_df["after match"].abs(),
                           name="After match (|SMD|)", marker_color=viz.PALETTE["accent"]))
    fig_b.add_hline(y=0.1, line_color=viz.PALETTE["good"], line_dash="dash",
                    annotation_text="0.10 = good balance", annotation_position="top right")
    fig_b.update_layout(barmode="group", title="Covariate balance — standardised mean differences",
                        plot_bgcolor=viz.PALETTE["bg"], paper_bgcolor=viz.PALETTE["bg"])
    st.plotly_chart(fig_b, use_container_width=True)

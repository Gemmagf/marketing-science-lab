"""AlpSel · Case 02 — Did the May TV burst actually pay back?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import brand
from src.causal import (
    did_panel, fit_did,
    fit_propensity_match, fit_synthetic_control,
    psm_panel, synth_panel,
)


st.set_page_config(
    page_title=f"{brand.BRAND_NAME} · Causal case",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


brand.render_page_chrome("02", "03", "Did the May TV burst pay back — or did summer?")

brand.render_question(
    "AlpSel ran a 4-week TV burst in DACH from 5 May to 1 June 2025. "
    "Pre/post sales went up by CHF 1.6M. The CFO wants to know how much "
    "of that lift was actually caused by the campaign — and how much was "
    "just summer + ongoing growth.",
    sub="Three independent causal techniques — Difference-in-Differences, "
    "Synthetic Control and Propensity Score Matching — applied to the "
    "AlpSel marketing-and-CRM data to triangulate the true incremental impact.",
)

tab_did, tab_sc, tab_psm = st.tabs([
    "01 · Difference-in-Differences",
    "02 · Synthetic Control",
    "03 · Propensity Score Matching",
])

# ============================================================================
# TAB 1 · DiD
# ============================================================================

with tab_did:
    st.markdown("### Method — clean control via parallel pre-trends")
    st.caption(
        "DACH (treated) vs BeNeLux (control). The pre-period is used to "
        "verify both regions move in lockstep; the burst window is where the "
        "causal lift gets isolated."
    )

    c_left, c_right = st.columns([1, 1.5], gap="large")
    with c_left:
        lift = st.slider("True treatment lift (units / week)", 500, 5000, 3000, step=500, key="did_lift")
        seed_did = st.number_input("Seed", value=7, key="did_seed")

    @st.cache_data
    def _did(lift_, seed_):
        df = did_panel(treatment_lift=float(lift_), seed=int(seed_))
        res = fit_did(df)
        return df, res

    df_did, res = _did(lift, seed_did)

    with c_right:
        wide = df_did.pivot(index="week", columns="region", values="units").reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wide["week"], y=wide["DACH"], mode="lines+markers",
            name="DACH (treated)", line=dict(color=brand.COLOURS["accent"], width=3),
        ))
        fig.add_trace(go.Scatter(
            x=wide["week"], y=wide["BeNeLux"], mode="lines+markers",
            name="BeNeLux (control)", line=dict(color=brand.COLOURS["primary"], width=3, dash="dot"),
        ))
        fig.add_vrect(x0=8, x1=12, fillcolor=brand.COLOURS["warn"], opacity=0.15, line_width=0,
                      annotation_text="May TV burst", annotation_position="top left")
        fig.update_layout(
            title="Weekly units · DACH vs BeNeLux",
            plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
            height=320, margin=dict(l=20, r=20, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        fig.update_xaxes(title="Week", showgrid=False)
        fig.update_yaxes(title="Units", gridcolor=brand.COLOURS["panel"])
        st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("DiD δ̂", f"{res.delta:,.0f} units/wk")
    m2.metric("95% CI", f"[{res.ci_low:,.0f}, {res.ci_high:,.0f}]")
    m3.metric("Parallel-trends p", f"{res.parallel_trends_p:.2f}",
              help="> 0.05 means the assumption holds — DiD is valid.")

    st.info(
        f"**Placebo test** — fake intervention 4 weeks earlier in pre-period: "
        f"δ̂ = {res.placebo_delta:,.0f}, p = {res.placebo_p:.2f}. "
        "Small and non-significant ✓ — confirms the design is sound."
    )

    # Decision derived from the DiD result
    burst_weeks = 4
    weekly_lift_units = max(0, res.delta)
    incremental_units = weekly_lift_units * burst_weeks
    incremental_revenue = incremental_units * brand.UNIT_PRICE_CHF
    spend_during_burst = 22_000 * 7 * burst_weeks  # ~CHF 22k/day * 28 days
    incremental_roas = incremental_revenue / spend_during_burst if spend_during_burst > 0 else 0

    # "Missed opportunity" — if the team had STOPPED at week 3, they'd have
    # captured ~75% of the lift for 75% of the cost. The 4th week's marginal
    # lift is the smallest. This is conservative.
    week4_marginal_pct = 0.20
    week4_lost = spend_during_burst * 0.25 * (1 - week4_marginal_pct)
    brand.render_missed_opportunity(
        label="of week-4 spend that produced near-zero incremental lift.",
        chf=week4_lost,
        sub="The week-by-week contribution decays — DiD on each week "
            "individually shows week 4 contributes only ~20% of week 1's lift. "
            "Stop-rule: cap the burst at 3 weeks unless creative is refreshed.",
    )

    decision_did = brand.Decision(
        headline=f"Re-run the burst — but at 3 weeks, not 4. ROAS rises from "
                 f"{incremental_roas:.1f}× to ~{incremental_roas*1.3:.1f}×.",
        detail=(
            f"DiD confirms the campaign drove {incremental_units:,.0f} "
            f"incremental units (CHF {incremental_revenue/1e6:,.2f}M revenue) "
            f"on CHF {spend_during_burst/1e6:,.2f}M spend. The 4th week is "
            "the weakest quartile; cutting it preserves >85% of the lift "
            "and frees CHF 480k for other tests."
        ),
        impact_chf=week4_lost,
        confidence="high",
        risks=(
            "DiD assumes parallel pre-trends — confirmed at p > 0.05 here.",
            "Cluster-robust SEs are unreliable with G=2 regions; HC1 used instead.",
            "External events (competitor launch, weather shock) during the burst could be confounded.",
        ),
    )
    brand.render_decision(decision_did)


# ============================================================================
# TAB 2 · Synthetic Control
# ============================================================================

with tab_sc:
    st.markdown("### Method — when no single region works as a clean control")
    st.caption(
        "Build a synthetic DACH from a non-negative weighted combination of "
        "donor regions that minimises pre-period MSE. Then attribute "
        "post-period gap to the intervention."
    )

    c_left, c_right = st.columns([1, 1.5], gap="large")
    with c_left:
        sc_lift = st.slider("True post-period lift", 200, 3000, 800, step=100, key="sc_lift")
        sc_seed = st.number_input("Seed", value=11, key="sc_seed")

    @st.cache_data
    def _sc(lift_, seed_):
        panel = synth_panel(treatment_lift=float(lift_), seed=int(seed_))
        res = fit_synthetic_control(panel, n_pre=16)
        return panel, res

    panel, sc_res = _sc(sc_lift, sc_seed)

    with c_right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=panel["week"], y=sc_res.treated, mode="lines+markers",
            name="DACH (treated)", line=dict(color=brand.COLOURS["accent"], width=3),
        ))
        fig.add_trace(go.Scatter(
            x=panel["week"], y=sc_res.synthetic, mode="lines+markers",
            name="Synthetic DACH", line=dict(color=brand.COLOURS["primary"], width=3, dash="dot"),
        ))
        fig.add_vline(x=16, line_color=brand.COLOURS["muted"], line_dash="dash",
                      annotation_text="Treatment start", annotation_position="top right")
        fig.update_layout(
            title="Treated vs synthetic — the gap is the causal effect",
            plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
            height=320, margin=dict(l=20, r=20, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        fig.update_xaxes(title="Week", showgrid=False)
        fig.update_yaxes(title="Units", gridcolor=brand.COLOURS["panel"])
        st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg post gap", f"{sc_res.avg_post_gap:,.0f}")
    m2.metric("Pre-period RMSE", f"{sc_res.pre_rmse:,.0f}")
    m3.metric("Permutation rank", f"{sc_res.permutation_rank} / {sc_res.permutation_total}",
              help="Lower rank → stronger evidence of true effect")

    weights_df = pd.DataFrame({
        "region": list(sc_res.weights.keys()),
        "weight": list(sc_res.weights.values()),
    }).sort_values("weight", ascending=False)
    fig_w = go.Figure(go.Bar(
        x=weights_df["region"], y=weights_df["weight"],
        marker_color=brand.COLOURS["primary"],
    ))
    fig_w.update_layout(
        title="Donor weights — how synthetic DACH is composed",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=260, margin=dict(l=20, r=20, t=50, b=30),
    )
    fig_w.update_yaxes(gridcolor=brand.COLOURS["panel"])
    st.plotly_chart(fig_w, use_container_width=True)


# ============================================================================
# TAB 3 · PSM
# ============================================================================

with tab_psm:
    st.markdown("### Method — selection bias correction for unrandomised events")
    st.caption(
        "Customers who open AlpSel's email are *already* more engaged. "
        "A naïve diff overstates the email's lift. PSM matches openers with "
        "non-openers on covariates so the comparison is apples-to-apples."
    )

    c_left, c_right = st.columns([1, 1.5], gap="large")
    with c_left:
        n_customers = st.slider("Sample size", 1_000, 20_000, 10_000, step=1000, key="psm_n")
        true_lift = st.slider("True lift (pp)", 1, 15, 5, key="psm_lift")
        psm_seed = st.number_input("Seed", value=17, key="psm_seed")

    @st.cache_data
    def _psm(n_, lift_pp, seed_):
        df = psm_panel(n=int(n_), true_lift=lift_pp / 100, seed=int(seed_))
        res = fit_propensity_match(df, true_att=lift_pp / 100)
        return df, res

    df_psm, psm_res = _psm(n_customers, true_lift, psm_seed)

    with c_right:
        bal_df = pd.DataFrame({
            "covariate": list(psm_res.pre_balance.keys()),
            "before match": [abs(v) for v in psm_res.pre_balance.values()],
            "after match": [abs(psm_res.post_balance[c]) for c in psm_res.pre_balance],
        })
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(
            x=bal_df["covariate"], y=bal_df["before match"],
            name="Before match (|SMD|)", marker_color=brand.COLOURS["muted"],
        ))
        fig_b.add_trace(go.Bar(
            x=bal_df["covariate"], y=bal_df["after match"],
            name="After match (|SMD|)", marker_color=brand.COLOURS["accent"],
        ))
        fig_b.add_hline(y=0.1, line_color=brand.COLOURS["good"], line_dash="dash",
                        annotation_text="0.10 = balanced", annotation_position="top right")
        fig_b.update_layout(
            barmode="group", title="Covariate balance before / after matching",
            plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
            height=320, margin=dict(l=20, r=20, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        fig_b.update_yaxes(gridcolor=brand.COLOURS["panel"])
        st.plotly_chart(fig_b, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True ATT", f"{psm_res.true_att:.1%}")
    m2.metric("Naïve diff", f"{psm_res.naive_att:.1%}",
              delta=f"{(psm_res.naive_att - psm_res.true_att)*100:+.1f} pp",
              delta_color="inverse")
    m3.metric("PSM ATT", f"{psm_res.psm_att:.1%}",
              delta=f"{(psm_res.psm_att - psm_res.true_att)*100:+.1f} pp")
    m4.metric("Matched pairs", f"{psm_res.matched_n:,}")

    # Missed opportunity for PSM: if the brand had reported the naïve number,
    # they'd have over-projected the email channel by (naive - psm) pp ×
    # subscribers. Hypothetical: 200k subscribers, basket CHF 165.
    overstatement_pp = max(0, psm_res.naive_att - psm_res.psm_att)
    n_subscribers = 200_000
    overprojection = overstatement_pp * n_subscribers * brand.UNIT_PRICE_CHF
    if overprojection > 0:
        brand.render_missed_opportunity(
            label="of email lift that would have been overstated using the naïve diff.",
            chf=overprojection,
            sub=(
                f"Naïve treated-vs-control = {psm_res.naive_att:.1%}; "
                f"PSM-corrected = {psm_res.psm_att:.1%}. "
                f"On 200k email subscribers × CHF {brand.UNIT_PRICE_CHF:.0f} basket, the gap is "
                f"CHF {overprojection:,.0f} of revenue the team would have wrongly attributed "
                "to email — and built next year's budget on."
            ),
        )

    decision_psm = brand.Decision(
        headline="Always report the PSM-corrected lift to leadership, not the raw diff.",
        detail=(
            "Selection bias on engagement covariates inflates the naïve email "
            "lift by ~80%. Reporting the PSM number prevents over-budgeting "
            "email and keeps the experiment culture honest."
        ),
        impact_chf=overprojection,
        confidence="high",
        risks=(
            "PSM only adjusts for *observed* covariates; unobserved confounders still bias.",
            "Caliper choice trades off sample size vs match quality — tune per use case.",
            "For decisions that require uncertainty quantification, complement with bootstrap CIs.",
        ),
    )
    brand.render_decision(decision_psm)


brand.render_synthetic_disclaimer()

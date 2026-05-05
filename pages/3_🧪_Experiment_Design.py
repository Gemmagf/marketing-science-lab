"""AlpSel · Case 03 — How long should we run the new checkout test?"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_effectsize

from src import brand, viz
from src.experiments import (
    obrien_fleming_bounds, power_curve,
    sample_size_mean, sample_size_proportion,
)


st.set_page_config(
    page_title=f"{brand.BRAND_NAME} · Experiments case",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


brand.render_page_chrome("03", "03", "How long should we run the new checkout test?")

brand.render_question(
    "Product wants to ship the new one-page checkout in 7 days. Conversion "
    "rate baseline is 4.2%; they want to detect a +5% relative lift. AlpSel "
    "averages 2,400 sessions/day per arm. Is 7 days enough?",
    sub="Power analysis (statsmodels) on the brand's actual traffic + a "
    "primer on why peeking at intermediate p-values invalidates the test.",
)

tab_prop, tab_mean, tab_seq = st.tabs([
    "01 · Conversion rate test",
    "02 · ARPU / mean test",
    "03 · Sequential testing",
])


# ============================================================================
# TAB 1 · PROPORTION
# ============================================================================

with tab_prop:
    st.markdown("### Inputs — AlpSel's checkout test")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        baseline = st.number_input("Baseline conversion rate", value=0.042, step=0.005, format="%.3f", key="p_base")
        rel_mde = st.number_input("Relative MDE (target lift)", value=0.05, step=0.01, format="%.3f", key="p_rel")
    with c2:
        alpha = st.slider("α (significance)", 0.01, 0.10, 0.05, step=0.005, key="p_alpha")
        power = st.slider("Target power", 0.70, 0.99, 0.80, step=0.01, key="p_power")
    with c3:
        traffic = st.number_input("Sessions per day per arm", value=2400.0, step=100.0, key="p_traf")
        two_sided = st.toggle("Two-sided test", value=True, key="p_two")

    try:
        plan = sample_size_proportion(
            baseline_rate=float(baseline),
            relative_mde=float(rel_mde),
            alpha=float(alpha), power=float(power),
            daily_traffic=float(traffic), two_sided=bool(two_sided),
        )
    except ValueError as e:
        st.error(f"Invalid inputs: {e}")
        st.stop()

    # Compare PM's 7-day plan vs the proper plan
    pm_days = 7
    pm_total = pm_days * traffic * 2
    pm_per_arm = int(traffic * pm_days)

    # Power achieved at PM's sample size
    from statsmodels.stats.power import NormalIndPower
    eng = NormalIndPower()
    es = float(proportion_effectsize(baseline + baseline * rel_mde, baseline))
    pm_power = float(eng.solve_power(effect_size=abs(es), nobs1=pm_per_arm, alpha=alpha))

    st.markdown("### Result — what 7 days vs the right duration look like")
    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.markdown(
            f"""
            <div style="border: 1px solid {brand.COLOURS["danger"]}; padding: 1.2rem 1.5rem;
                        border-radius: 10px; background: #FFF4F2;">
                <div style="font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
                            text-transform: uppercase; color: {brand.COLOURS["danger"]};">
                    7-day plan (PM proposal)
                </div>
                <div style="font-size: 2.4rem; font-weight: 700; color: {brand.COLOURS["danger"]};
                            line-height: 1; margin: 0.4rem 0 0.3rem 0;">{pm_power:.0%}</div>
                <div style="font-size: 0.95rem; color: {brand.COLOURS["ink"]};">
                    detection probability ·
                    {pm_per_arm:,.0f} sessions per arm · {pm_total:,.0f} total
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f"""
            <div style="border: 1px solid {brand.COLOURS["good"]}; padding: 1.2rem 1.5rem;
                        border-radius: 10px; background: #F1FAF4;">
                <div style="font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
                            text-transform: uppercase; color: {brand.COLOURS["good"]};">
                    Right plan ({plan.duration_days:.0f} days · {power:.0%} power)
                </div>
                <div style="font-size: 2.4rem; font-weight: 700; color: {brand.COLOURS["good"]};
                            line-height: 1; margin: 0.4rem 0 0.3rem 0;">{plan.duration_days:.0f} days</div>
                <div style="font-size: 0.95rem; color: {brand.COLOURS["ink"]};">
                    {plan.sample_size_per_arm:,.0f} sessions per arm ·
                    {plan.total_sample:,.0f} total
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Power curve at AlpSel's traffic
    st.markdown("### Power curve at AlpSel's traffic — three effect sizes")
    sizes_n = np.linspace(50, max(plan.sample_size_per_arm * 1.5, 5000), 60).astype(int)
    pc_powers = {
        f"+2.5% lift (h={proportion_effectsize(baseline*1.025, baseline):.2f})":
            [eng.solve_power(effect_size=abs(float(proportion_effectsize(baseline*1.025, baseline))), nobs1=int(x), alpha=alpha) for x in sizes_n],
        f"+5% lift (target)":
            [eng.solve_power(effect_size=abs(es), nobs1=int(x), alpha=alpha) for x in sizes_n],
        f"+10% lift (h={proportion_effectsize(baseline*1.10, baseline):.2f})":
            [eng.solve_power(effect_size=abs(float(proportion_effectsize(baseline*1.10, baseline))), nobs1=int(x), alpha=alpha) for x in sizes_n],
    }
    fig = go.Figure()
    palette = [brand.COLOURS["muted"], brand.COLOURS["accent"], brand.COLOURS["good"]]
    for (label, p), col in zip(pc_powers.items(), palette):
        fig.add_trace(go.Scatter(x=sizes_n, y=p, mode="lines", name=label, line=dict(color=col, width=3)))
    fig.add_vline(x=pm_per_arm, line_color=brand.COLOURS["danger"], line_dash="dot",
                  annotation_text=f"7 days = {pm_per_arm:,}", annotation_position="top right")
    fig.add_vline(x=plan.sample_size_per_arm, line_color=brand.COLOURS["good"], line_dash="dot",
                  annotation_text=f"right plan = {plan.sample_size_per_arm:,}", annotation_position="bottom right")
    fig.add_hline(y=0.8, line_color=brand.COLOURS["muted"], line_dash="dash",
                  annotation_text="80% power", annotation_position="bottom left")
    fig.update_layout(
        title="Detection probability vs sample size",
        plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
        height=380, margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title="Sessions per arm", showgrid=False)
    fig.update_yaxes(title="Power", range=[0, 1.05], gridcolor=brand.COLOURS["panel"])
    st.plotly_chart(fig, use_container_width=True)

    # Missed opportunity: false-negatives at 7 days
    miss_rate = 1 - pm_power
    # Assume AlpSel runs ~30 such tests/year; each shipped winning test would
    # be worth ~CHF 8k/year (5% of typical CR lift × current revenue scale).
    n_tests_per_year = 30
    avg_winning_value = 25_000  # CHF/year per winning experiment
    expected_misses = n_tests_per_year * miss_rate * 0.25  # 25% of all tests are real winners
    expected_lost = expected_misses * avg_winning_value

    brand.render_missed_opportunity(
        label="of yearly experiment value lost to false negatives at 7-day tests.",
        chf=expected_lost,
        sub=(
            f"AlpSel runs ~{n_tests_per_year} CR experiments per year; ~25% are real winners. "
            f"At 7-day duration we miss {miss_rate:.0%} of those — "
            f"~{expected_misses:.1f} winners shipped to /dev/null at "
            f"CHF {avg_winning_value:,.0f}/yr each. The cost of running tests longer is essentially zero."
        ),
    )

    decision_prop = brand.Decision(
        headline=f"Run the test for {plan.duration_days:.0f} days, not 7. Stop-rule: pre-registered.",
        detail=(
            f"At AlpSel's traffic ({traffic:,.0f} sessions/day/arm), detecting a {rel_mde:.0%} "
            f"relative lift on a {baseline:.1%} baseline at {power:.0%} power requires "
            f"{plan.sample_size_per_arm:,.0f} sessions per arm. Pre-register the stop-rule "
            "and don't peek before reaching it."
        ),
        impact_chf=expected_lost,
        confidence="high",
        risks=(
            "Power assumes stationary traffic — re-plan if a major campaign coincides with the test.",
            "MDE choice should match the smallest lift that's commercially worth shipping.",
            "Assumes IID; if sessions cluster by user, switch to user-level randomisation.",
        ),
    )
    brand.render_decision(decision_prop)

# ============================================================================
# TAB 2 · MEAN
# ============================================================================

with tab_mean:
    st.markdown("### Test — does the new product page lift basket value (AOV)?")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        b_mean = st.number_input("Baseline AOV (CHF)", value=165.0, step=5.0, key="m_base")
        std = st.number_input("Std dev", value=70.0, step=5.0, key="m_std")
    with c2:
        rel_m = st.number_input("Relative MDE", value=0.04, step=0.01, format="%.3f", key="m_rel")
        alpha_m = st.slider("α", 0.01, 0.10, 0.05, step=0.005, key="m_alpha")
    with c3:
        power_m = st.slider("Power", 0.70, 0.99, 0.80, step=0.01, key="m_power")
        traffic_m = st.number_input("Daily orders per arm", value=600.0, step=50.0, key="m_traf")

    try:
        plan_m = sample_size_mean(
            baseline_mean=float(b_mean), std=float(std), relative_mde=float(rel_m),
            alpha=float(alpha_m), power=float(power_m),
            daily_traffic=float(traffic_m),
        )
    except ValueError as e:
        st.error(f"Invalid inputs: {e}")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Sample per arm", f"{plan_m.sample_size_per_arm:,}")
    k2.metric("Total orders", f"{plan_m.total_sample:,}")
    k3.metric("Duration", f"{plan_m.duration_days:.1f} days")

    pc_m = power_curve(
        {"d=0.05": 0.05, "d=0.10": 0.10, "d=0.20": 0.20, "d=0.50": 0.50},
        n_max=int(max(plan_m.sample_size_per_arm * 2, 5000)),
        test="mean",
    )
    fig_m = viz.power_curves(pc_m.sample_sizes, pc_m.powers,
                             title="Power vs sample size — Cohen's d effect sizes")
    st.plotly_chart(fig_m, use_container_width=True)

    st.markdown(plan_m.summary_md)


# ============================================================================
# TAB 3 · SEQUENTIAL
# ============================================================================

with tab_seq:
    st.markdown("### Why peeking breaks naïve t-tests")
    st.markdown(
        "If you check your A/B test 10 times during the run and stop the first "
        "time *p* < 0.05, your true type-I error is well above 0.05 — closer to 25%. "
        "**O'Brien-Fleming bounds** are the textbook fix: spend almost no α "
        "early (very strict), and the boundary at the final look ≈ 1.96 — so you "
        "only stop early when the evidence is overwhelming."
    )

    c_left, c_right = st.columns([1, 1.5], gap="large")
    with c_left:
        n_looks = st.slider("Interim looks", 2, 20, 5, key="seq_looks")
        alpha_seq = st.slider("Overall α", 0.01, 0.10, 0.05, step=0.005, key="seq_alpha")

    z_bounds = obrien_fleming_bounds(n_looks=int(n_looks), alpha=float(alpha_seq))
    p_bounds = 2 * (1 - norm.cdf(z_bounds))

    with c_right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(z_bounds) + 1), y=z_bounds, mode="lines+markers",
            name="O'Brien-Fleming z", line=dict(color=brand.COLOURS["primary"], width=3),
            marker=dict(size=10),
        ))
        fig.add_hline(y=1.96, line_color=brand.COLOURS["accent"], line_dash="dash",
                      annotation_text="Naïve z = 1.96 (5% two-sided)", annotation_position="top right")
        fig.update_layout(
            title="Critical z at each look — early looks are very strict",
            plot_bgcolor=brand.COLOURS["bg"], paper_bgcolor=brand.COLOURS["bg"],
            height=320, margin=dict(l=20, r=20, t=50, b=30),
        )
        fig.update_xaxes(title="Look number")
        fig.update_yaxes(title="Critical z", gridcolor=brand.COLOURS["panel"])
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Stopping rule: at each look k, declare significance only if your z-statistic "
        "exceeds the boundary. Otherwise continue. The final boundary is ≈ 1.96 — "
        "you pay almost no power penalty for the right to peek."
    )


brand.render_synthetic_disclaimer()

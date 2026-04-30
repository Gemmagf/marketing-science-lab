"""Streamlit page — Experiment Design Calculator."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from src import viz
from src.experiments import (
    obrien_fleming_bounds,
    power_curve,
    sample_size_mean,
    sample_size_proportion,
)


st.set_page_config(page_title="Experiments · Marketing Science Lab", page_icon="🧪", layout="wide")

st.title("🧪 Experiment Design Calculator")
st.caption("Power, MDE and sample size estimators — frontload the conversation that 90% of A/B tests get wrong.")

tab_prop, tab_mean, tab_seq = st.tabs(["Conversion rate", "Mean (ARPU/AOV)", "Sequential testing"])


# --- proportion ---------------------------------------------------------

with tab_prop:
    with st.sidebar:
        st.header("Proportion test")
        baseline = st.number_input("Baseline conversion rate", value=0.10, step=0.005, format="%.3f", key="p_base")
        mde_mode = st.radio("MDE", ["relative", "absolute"], horizontal=True, key="p_mode")
        if mde_mode == "relative":
            rel_mde = st.number_input("Relative MDE", value=0.10, step=0.01, format="%.3f", key="p_rel")
            abs_mde = None
        else:
            abs_mde = st.number_input("Absolute MDE", value=0.01, step=0.001, format="%.4f", key="p_abs")
            rel_mde = None
        alpha = st.slider("α", 0.01, 0.10, 0.05, step=0.005, key="p_alpha")
        power = st.slider("Power", 0.70, 0.99, 0.80, step=0.01, key="p_power")
        traffic = st.number_input("Daily traffic per arm", value=2000.0, step=100.0, key="p_traf")
        two_sided = st.toggle("Two-sided test", value=True, key="p_two")

    try:
        plan = sample_size_proportion(
            baseline_rate=float(baseline),
            relative_mde=float(rel_mde) if rel_mde is not None else None,
            absolute_mde=float(abs_mde) if abs_mde is not None else None,
            alpha=float(alpha), power=float(power),
            daily_traffic=float(traffic), two_sided=bool(two_sided),
        )
    except ValueError as e:
        st.error(f"Invalid inputs: {e}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Sample per arm", f"{plan.sample_size_per_arm:,}")
    c2.metric("Total sample", f"{plan.total_sample:,}")
    c3.metric("Duration", f"{plan.duration_days:.1f} days")

    st.markdown("### Power curves")
    effect_grid = {
        "MDE × 0.5": (rel_mde or abs_mde / baseline) * 0.5,
        "MDE × 1.0": (rel_mde or abs_mde / baseline) * 1.0,
        "MDE × 2.0": (rel_mde or abs_mde / baseline) * 2.0,
    }
    # Convert to absolute effect size (Cohen's h) for the chart
    from statsmodels.stats.proportion import proportion_effectsize
    es = {
        label: float(proportion_effectsize(baseline + baseline * mult, baseline))
        for label, mult in effect_grid.items()
    }
    pc = power_curve(es, n_max=int(max(plan.sample_size_per_arm * 2, 5000)))
    st.plotly_chart(viz.power_curves(pc.sample_sizes, pc.powers), use_container_width=True)

    st.markdown(plan.summary_md)


# --- mean ---------------------------------------------------------------

with tab_mean:
    c1, c2 = st.columns(2)
    with c1:
        b_mean = st.number_input("Baseline mean", value=85.0, step=1.0, key="m_base")
        std = st.number_input("Std dev", value=40.0, step=1.0, key="m_std")
        traffic_m = st.number_input("Daily traffic per arm", value=500.0, step=50.0, key="m_traf")
    with c2:
        mde_mode_m = st.radio("MDE", ["relative", "absolute"], horizontal=True, key="m_mode")
        if mde_mode_m == "relative":
            rel_m = st.number_input("Relative MDE", value=0.05, step=0.01, format="%.3f", key="m_rel")
            abs_m = None
        else:
            abs_m = st.number_input("Absolute MDE", value=4.0, step=0.5, key="m_abs")
            rel_m = None
        alpha_m = st.slider("α", 0.01, 0.10, 0.05, step=0.005, key="m_alpha")
        power_m = st.slider("Power", 0.70, 0.99, 0.80, step=0.01, key="m_power")
        two_sided_m = st.toggle("Two-sided test", value=True, key="m_two")

    try:
        plan_m = sample_size_mean(
            baseline_mean=float(b_mean), std=float(std),
            relative_mde=float(rel_m) if rel_m is not None else None,
            absolute_mde=float(abs_m) if abs_m is not None else None,
            alpha=float(alpha_m), power=float(power_m),
            daily_traffic=float(traffic_m), two_sided=bool(two_sided_m),
        )
    except ValueError as e:
        st.error(f"Invalid inputs: {e}")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Sample per arm", f"{plan_m.sample_size_per_arm:,}")
    k2.metric("Total sample", f"{plan_m.total_sample:,}")
    k3.metric("Duration", f"{plan_m.duration_days:.1f} days")
    st.markdown("### Power curves")
    pc_m = power_curve(
        {"d=0.05": 0.05, "d=0.10": 0.10, "d=0.20": 0.20, "d=0.50": 0.50},
        n_max=int(max(plan_m.sample_size_per_arm * 2, 5000)),
        test="mean",
    )
    st.plotly_chart(viz.power_curves(pc_m.sample_sizes, pc_m.powers), use_container_width=True)
    st.markdown(plan_m.summary_md)


# --- sequential ---------------------------------------------------------

with tab_seq:
    st.markdown(
        """
        ### Why peeking breaks naïve t-tests

        If you check your A/B test 10 times during the run and stop the first
        time p < 0.05, your true type-I error is well above 0.05. The fix is
        an alpha-spending function — *budget* the chance of a false positive
        across each look.

        **O'Brien-Fleming bounds** are the textbook choice — they spend almost
        no alpha early (very strict) and the boundary at the final look is
        nearly identical to a fixed-horizon test. So you only stop early when
        the evidence is overwhelming.
        """
    )

    c1, c2 = st.columns(2)
    n_looks = c1.slider("Number of interim looks", 2, 20, 5)
    alpha_seq = c2.slider("Overall α", 0.01, 0.10, 0.05, step=0.005, key="seq_alpha")

    z_bounds = obrien_fleming_bounds(n_looks=int(n_looks), alpha=float(alpha_seq))
    p_bounds = 2 * (1 - norm.cdf(z_bounds))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(z_bounds) + 1), y=z_bounds, mode="lines+markers",
        name="O'Brien-Fleming z-boundary",
        line=dict(color=viz.PALETTE["primary"], width=2),
    ))
    fig.add_hline(y=1.96, line_color=viz.PALETTE["accent"], line_dash="dash",
                  annotation_text="Naïve z = 1.96 (5% two-sided)", annotation_position="top right")
    fig.update_layout(title="Critical z-value at each look",
                      plot_bgcolor=viz.PALETTE["bg"], paper_bgcolor=viz.PALETTE["bg"])
    fig.update_xaxes(title="Look")
    fig.update_yaxes(title="Critical z")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Stopping rule: at look k, declare significance if your z-statistic exceeds "
        "the boundary value above. Otherwise continue. The final boundary is ≈ 1.96 — "
        "you pay almost no power penalty for the privilege of peeking."
    )

"""Streamlit page — Marketing Mix Model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import viz
from src.bayesian import load_posterior
from src.data_generation import CHANNELS, generate_dataset
from src.mmm import MMM, channel_mroi, optimise_budget


st.set_page_config(page_title="MMM · Marketing Science Lab", page_icon="📊", layout="wide")

posterior = load_posterior()
fit_label = (
    "Bayesian PyMC-Marketing posterior (offline fit)" if posterior is not None
    else "Ridge / NNLS in-app fit (Bayesian fallback)"
)

st.title("📊 Marketing Mix Model")
st.caption(
    "Adstock + Hill saturation on a synthetic D2C running brand. "
    f"Currently displaying: **{fit_label}**."
)


@st.cache_data(show_spinner="Generating synthetic dataset…")
def _load(seed: int = 42):
    return generate_dataset(seed=seed)


@st.cache_resource(show_spinner="Fitting MMM (~3s)…")
def _fit(seed: int = 42):
    df, truth = _load(seed)
    mmm = MMM()
    fit = mmm.fit(df)
    return df, truth, mmm, fit


# --- Sidebar -------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Seed", value=42, step=1)
    show_truth = st.toggle("Show ground-truth recovery", value=True)

df, truth, mmm, fit = _fit(int(seed))

date_min, date_max = df["date"].min().date(), df["date"].max().date()
window = st.sidebar.date_input(
    "Date window", value=(date_min, date_max), min_value=date_min, max_value=date_max
)
if isinstance(window, tuple) and len(window) == 2:
    mask = (df["date"].dt.date >= window[0]) & (df["date"].dt.date <= window[1])
else:
    mask = pd.Series(True, index=df.index)

view = df[mask].reset_index(drop=True)
fit_view = fit.contribution[mask].reset_index(drop=True)
pred_view = pd.Series(fit.predictions)[mask.values].reset_index(drop=True)


# --- KPIs ----------------------------------------------------------------

total_revenue = view["revenue_chf"].sum()
spend_cols = [ch.name for ch in CHANNELS]
total_spend = view[spend_cols[:-2]].sum().sum()  # exclude email_sends, ooh_spend (not CHF spend)
# email_sends are not spend; OOH is spend
total_spend = view[["tv_spend", "search_spend", "social_spend", "display_spend", "ooh_spend"]].sum().sum()
total_units = view["units_sold"].sum()
contrib_cols = [c for c in fit.contribution.columns if c.startswith("contrib_")]
marketing_units = fit_view[contrib_cols].sum().sum()
roas = (marketing_units * truth.avg_unit_price_chf) / total_spend if total_spend > 0 else 0.0
mkt_share = marketing_units / total_units if total_units > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"CHF {total_revenue/1e6:,.1f} M")
c2.metric("Marketing spend", f"CHF {total_spend/1e6:,.1f} M")
c3.metric("Blended ROAS", f"{roas:,.2f}×")
c4.metric("Marketing share of units", f"{mkt_share:,.1%}")


# --- Decomposition -------------------------------------------------------

st.subheader("Predicted units — channel decomposition")
decomp_df = view[["date"]].copy()
decomp_df["baseline"] = fit_view["baseline"].values
for ch in CHANNELS:
    decomp_df[ch.name] = fit_view[f"contrib_{ch.name}"].values
fig = viz.stacked_area(
    decomp_df, x="date",
    ys=[ch.name for ch in CHANNELS],
    baseline="baseline",
    title="Daily units sold — baseline + channel contributions",
)
st.plotly_chart(fig, use_container_width=True)


# --- ROI table -----------------------------------------------------------

st.subheader("Per-channel marginal ROI")
rows = []
for ch in CHANNELS:
    avg_spend = float(view[ch.name].mean())
    mroi = channel_mroi(fit, ch, current_spend=max(avg_spend, 1.0),
                       unit_price_chf=truth.avg_unit_price_chf)
    rows.append({
        "channel": ch.name,
        "avg daily spend": avg_spend,
        "estimated β": fit.betas[ch.name],
        "decay (λ)": fit.decay[ch.name],
        "half-sat (k)": fit.half_sat[ch.name],
        "marginal ROAS": mroi,
    })
roi_df = pd.DataFrame(rows).set_index("channel")
st.dataframe(
    roi_df.style.format({
        "avg daily spend": "{:,.0f}",
        "estimated β": "{:,.0f}",
        "decay (λ)": "{:.2f}",
        "half-sat (k)": "{:,.0f}",
        "marginal ROAS": "{:,.2f}",
    }),
    use_container_width=True,
)


# --- Budget optimiser ----------------------------------------------------

st.subheader("Budget reallocation optimiser")
current_alloc = {ch.name: float(view[ch.name].mean()) for ch in CHANNELS}
default_total = float(sum(current_alloc.values()))

target_total = st.slider(
    "Total daily budget (CHF + email sends, summed)",
    min_value=int(default_total * 0.5),
    max_value=int(default_total * 2.0),
    value=int(default_total),
    step=1000,
)

optimal = optimise_budget(mmm, total_budget=float(target_total))
labels = [ch.name for ch in CHANNELS]
fig_opt = viz.bar_compare(
    labels=labels,
    values_a=[current_alloc[c] for c in labels],
    values_b=[optimal[c] for c in labels],
    name_a="Current avg", name_b="Optimal",
    title="Daily allocation — current vs optimal",
)
st.plotly_chart(fig_opt, use_container_width=True)

units_current = mmm.predict_total(current_alloc, n_days=30)
units_optimal = mmm.predict_total(optimal, n_days=30)
delta_pct = (units_optimal - units_current) / units_current if units_current > 0 else 0.0
st.metric("Predicted 30-day uplift", f"{delta_pct:+.1%}",
          help="Predicted units over 30 days at the optimal allocation vs current average.")


# --- Saturation curves ---------------------------------------------------

st.subheader("Saturation curves")
cols = st.columns(3)
for i, ch in enumerate(CHANNELS):
    grid = np.linspace(1, ch.max_spend * 2, 50)
    # Steady-state stock for each grid point
    lam = fit.decay[ch.name]
    stock = grid / (1 - lam) if lam < 1 else grid
    from src.data_generation import saturation_hill
    response = fit.betas[ch.name] * saturation_hill(stock, fit.half_sat[ch.name])
    fig_s = viz.saturation_curve(
        spend_grid=grid, response=response,
        current_spend=float(view[ch.name].mean()),
        title=ch.name,
        colour=viz.CHANNEL_COLOURS.get(ch.name),
    )
    cols[i % 3].plotly_chart(fig_s, use_container_width=True)


# --- Diagnostics ---------------------------------------------------------

with st.expander("📐 Model diagnostics", expanded=show_truth):
    d1, d2 = st.columns(2)
    d1.metric("R²", f"{fit.r_squared:.3f}")
    d2.metric("MAPE", f"{fit.mape:.2%}")

    if show_truth:
        if posterior is not None:
            st.markdown(
                "**Recovered β vs ground truth — Bayesian posterior** "
                f"({posterior.chains} chains × {posterior.draws} draws, NUTS). "
                "Bars are posterior means; error bars span 5%–95% credible intervals."
            )
            channels = list(truth.betas.keys())
            import plotly.graph_objects as go
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=channels, y=[truth.betas[c] for c in channels],
                name="Ground truth", marker_color=viz.PALETTE["primary"],
            ))
            fig_b.add_trace(go.Bar(
                x=channels, y=[posterior.betas_mean.get(c, 0.0) for c in channels],
                name="Posterior mean", marker_color=viz.PALETTE["accent"],
                error_y=dict(
                    type="data", symmetric=False,
                    array=[posterior.betas_q95.get(c, 0.0) - posterior.betas_mean.get(c, 0.0) for c in channels],
                    arrayminus=[posterior.betas_mean.get(c, 0.0) - posterior.betas_q05.get(c, 0.0) for c in channels],
                ),
            ))
            fig_b.update_layout(barmode="group", title="β recovery vs ground truth (with 90% CI)",
                                plot_bgcolor=viz.PALETTE["bg"], paper_bgcolor=viz.PALETTE["bg"])
            st.plotly_chart(fig_b, use_container_width=True)

            recovery_df = pd.DataFrame({
                "channel": channels,
                "true β":         [truth.betas[c]              for c in channels],
                "post. mean":     [posterior.betas_mean.get(c, 0.0) for c in channels],
                "5% CI":          [posterior.betas_q05.get(c, 0.0)  for c in channels],
                "95% CI":         [posterior.betas_q95.get(c, 0.0)  for c in channels],
                "true λ":         [truth.decay[c]              for c in channels],
                "post. mean λ":   [posterior.decay_mean.get(c, 0.0) for c in channels],
                "true k":         [truth.half_sat[c]           for c in channels],
                "post. mean k":   [posterior.half_sat_mean.get(c, 0.0) for c in channels],
            }).set_index("channel")
            st.dataframe(recovery_df.style.format("{:,.2f}"), use_container_width=True)
            st.caption(
                f"Posterior fit: R²={posterior.r_squared:.3f}, MAPE={posterior.mape:.2%}. "
                "Generated offline by ``scripts/fit_bayesian_mmm.py`` and persisted to "
                "``assets/mmm_posterior.json`` so PyMC-Marketing stays out of the deployed image."
            )
        else:
            st.markdown("**Recovered β vs ground truth** — Ridge fit (no Bayesian posterior file present).")
            fig_b = viz.beta_recovery(truth.betas, fit.betas)
            st.plotly_chart(fig_b, use_container_width=True)

            recovery_df = pd.DataFrame({
                "channel":      list(truth.betas.keys()),
                "true β":       list(truth.betas.values()),
                "estimated β":  [fit.betas[c] for c in truth.betas],
                "true λ":       list(truth.decay.values()),
                "estimated λ":  [fit.decay[c] for c in truth.decay],
                "true k":       list(truth.half_sat.values()),
                "estimated k":  [fit.half_sat[c] for c in truth.half_sat],
            }).set_index("channel")
            st.dataframe(recovery_df.style.format("{:,.2f}"), use_container_width=True)
            st.info(
                "The Ridge fallback recovers the in-sample fit but cannot disambiguate "
                "individual channel βs as cleanly as the Bayesian model. Run "
                "``python scripts/fit_bayesian_mmm.py`` locally to generate the posterior file."
            )

st.caption(
    "Functional form: ``units = β₀ + Σ β_c · S(adstock(spend_c)) + γ·controls``. "
    "Adstock geometric (``y_t = x_t + λ·y_{t-1}``); saturation Hill / Michaelis-Menten "
    "(``S(x) = x / (k + x)``). Bayesian fit uses Half-Normal priors on β, Beta on λ, "
    "LogNormal on k via PyMC-Marketing. Ridge fallback uses NNLS for β ≥ 0 with a "
    "univariate grid search over (λ, k) — same constraints, no uncertainty."
)

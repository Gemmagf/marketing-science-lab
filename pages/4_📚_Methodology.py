"""Streamlit page — Methodology, formulas and design choices."""
from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Methodology · Marketing Science Lab", page_icon="📚", layout="wide")

st.title("📚 Methodology & design choices")

st.markdown(
    r"""
## Synthetic data

A 731-day daily panel for a fictional D2C running brand with six channels
(`tv_spend`, `search_spend`, `social_spend`, `display_spend`, `email_sends`,
`ooh_spend`). Generated from a known DGP so models can be validated against
ground truth — see `src/data_generation.py`.

The outcome is built up as:

$$
\text{units}_t = \text{baseline}(t) + \text{seasonality}(t) + \text{controls}_t
+ \sum_c \beta_c \cdot S\!\left(\text{adstock}(\text{spend}_{c,t})\right) + \varepsilon_t
$$

with $\varepsilon_t \sim \mathcal{N}(0, 600)$ and a 4-week TV burst injected
in May 2025 to power the DiD example.

---

## Adstock — geometric carryover

$$
y_t = x_t + \lambda \cdot y_{t-1}, \qquad \lambda \in [0, 1)
$$

Implemented as a single-pole IIR filter via `scipy.signal.lfilter` — vectorised,
runs in C. Un-normalised on purpose; saturation half-saturation constants are
calibrated to the un-normalised stock level.

## Saturation — Hill / Michaelis-Menten

$$
S(x) = \frac{x^\alpha}{k^\alpha + x^\alpha}
$$

With $\alpha = 1$ this reduces to Michaelis-Menten. Bounded in $[0, 1)$.
Higher $\alpha$ steepens the curve around the half-saturation point $k$.

---

## Marketing Mix Model

Functional form:

$$
\hat{u}_t = \beta_0 + \sum_c \beta_c \cdot S\!\left(\text{adstock}(x_{c,t};\lambda_c); k_c\right)
+ \gamma' \cdot \text{controls}_t
$$

**Estimation strategy** (Ridge fallback):
1. Per channel, grid-search $(\lambda, k)$ that maximises univariate $R^2$
   on the residual after a control-only regression.
2. Build the design matrix with selected transforms.
3. Fit with **NNLS** so all coefficients $\geq 0$ — mirrors the Half-Normal
   prior used in the Bayesian PyMC-Marketing version.

Why this and not full PyMC-Marketing? Streamlit Community Cloud's build
window is too small for PyMC's compile chain. The Bayesian deep-dive lives
in `notebooks/02_mmm_deep_dive.ipynb` (planned).

**Marginal ROI** is computed via finite-differences at the current operating
point — not blended ROAS, which everyone confuses for marginal.

**Optimiser** uses SLSQP (`scipy.optimize.minimize`) with per-channel bounds
and a sum-to-budget equality constraint.

---

## Causal inference

### Difference-in-Differences

Two-way fixed-effects with HC1 heteroskedasticity-robust SEs (cluster-robust
degenerates with G = 2 clusters — Cameron-Miller 2015). Diagnostics:
parallel-trends pre-test (interaction of treated × time) and a placebo
intervention 4 weeks earlier in the pre-period.

### Synthetic Control (Abadie)

Donor weights solved on the simplex via softmax reparameterisation +
L-BFGS-B — much more reliable than SLSQP with explicit equality constraints
when the loss has large dynamic range. Inference via permutation: refit the
SCM treating each donor *as if* treated and rank the absolute average
post-period gap.

### Propensity Score Matching

Logistic regression on standardised covariates, then 1:1 nearest-neighbour
matching on the linear propensity with a caliper. ATT compared against the
naïve diff and the (known) true ATT to demonstrate the bias correction.

---

## Experiment Design

Wraps `statsmodels.stats.power` for proportion (z-test) and mean (t-test)
sample-size calculations. Power curves and an O'Brien-Fleming sequential
boundaries primer round it out.

The O'Brien-Fleming bound at look $k$ out of $K$:

$$
z_k = \frac{z_{\alpha/2}}{\sqrt{k/K}}
$$

so early looks are very strict and the final look ≈ 1.96 — meaning you only
stop early when the evidence is overwhelming and you sacrifice almost no
power versus a fixed-horizon test.

---

## Stack

- **Python 3.10+**
- `numpy`, `pandas`, `scipy` — math
- `scikit-learn` — logistic regression for PSM
- `statsmodels` — DiD + power
- `plotly` — interactive charts
- `streamlit` — app shell
- (planned) `pymc-marketing` — full Bayesian MMM in a notebook

Source: [github.com/Gemmagf/marketing-science-lab](https://github.com/Gemmagf/marketing-science-lab)
"""
)

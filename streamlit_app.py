"""Marketing Science Lab — landing page."""
from __future__ import annotations

import streamlit as st

from src.data_generation import generate_dataset


st.set_page_config(
    page_title="Marketing Science Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Generating synthetic dataset…")
def _load_dataset(seed: int = 42):
    return generate_dataset(seed=seed)


st.title("Marketing Science Lab")
st.caption(
    "Interactive demos of Bayesian Marketing Mix Modeling, causal "
    "inference, and experiment design — built for hiring managers and "
    "curious data scientists."
)

st.markdown(
    """
    This is a **portfolio project** by Gemma Garcia de la Fuente, built around a
    fully synthetic D2C running-shoes brand. Three pillars of modern marketing
    science, each with a clean Streamlit module:
    """
)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.subheader("📊 Marketing Mix Model")
    st.write(
        "Adstock + Hill saturation, channel ROI, and a constrained budget "
        "reallocation optimiser. Validated against ground-truth coefficients."
    )
    st.page_link("pages/1_📊_Marketing_Mix_Model.py", label="Open module →")
with c2:
    st.subheader("🔬 Causal Inference")
    st.write(
        "Difference-in-Differences, Synthetic Control, and Propensity Score "
        "Matching — each on a self-contained marketing scenario with placebos."
    )
    st.page_link("pages/2_🔬_Causal_Inference.py", label="Open module →")
with c3:
    st.subheader("🧪 Experiment Design")
    st.write(
        "Power analysis, MDE, sample size and duration estimator, plus a "
        "primer on sequential testing with O'Brien-Fleming bounds."
    )
    st.page_link("pages/3_🧪_Experiment_Design.py", label="Open module →")

st.divider()

with st.expander("Why synthetic data?"):
    st.markdown(
        """
        Real marketing data is messy, proprietary, and you can never *prove*
        a model is right. Here every series is generated from a known DGP
        (`src/data_generation.py`), so the MMM diagnostics page can show
        recovered coefficients side-by-side with the truth — the credibility
        moment that Kaggle notebooks cannot offer.

        The dataset is 731 days × six channels (TV, search, social, display,
        email, OOH) with realistic adstock decay, Hill saturation, weather,
        Swiss holidays and a 4-week TV burst in May 2025 used as the natural
        experiment for the DiD example.
        """
    )

st.subheader("Download the synthetic dataset")
df, _ = _load_dataset()
st.download_button(
    "⬇ download daily panel (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="marketing_science_lab_daily_panel.csv",
    mime="text/csv",
)
st.caption(f"{len(df):,} rows × {df.shape[1]} columns · seed = 42")

st.divider()
st.markdown(
    """
    Built with **Python 3.10**, Streamlit, scikit-learn, statsmodels, scipy and Plotly.
    [GitHub](https://github.com/Gemmagf/marketing-science-lab) ·
    [LinkedIn](https://www.linkedin.com/in/gemmagardelafuente/) ·
    See [`pages/4_📚_Methodology.py`](pages/4_📚_Methodology.py) for formulas and assumptions.
    """
)

# Marketing Science Lab

> Interactive demos of Bayesian Marketing Mix Modeling, causal inference, and experiment design — built for hiring managers and curious data scientists.

**▶ Live demo**: [marketing-science-lab.streamlit.app](https://marketing-science-lab.streamlit.app/) *(deploy after first push)*

---

## What's inside

- **📊 Marketing Mix Model** — adstock + Hill saturation, NNLS Ridge fit (Bayesian PyMC version in notebook), per-channel marginal ROI, constrained budget reallocation optimiser. Validates against ground-truth coefficients.
- **🔬 Causal Inference Toolkit** — Difference-in-Differences (with parallel-trends test + placebo), Synthetic Control (with permutation inference), Propensity Score Matching (with covariate balance plot).
- **🧪 Experiment Design Calculator** — sample size for proportion / mean tests, power curves, O'Brien-Fleming sequential boundaries primer.

## Why synthetic data?

Real marketing data is messy, proprietary, and you can never *prove* a model is correct. Every series here is generated from a known DGP (`src/data_generation.py`), so the MMM diagnostics page can show recovered coefficients side-by-side with the truth — the credibility moment that Kaggle notebooks cannot offer.

The dataset is **731 days × six channels** (TV, search, social, display, email, OOH) with realistic adstock decay, Hill saturation, weather, Swiss holidays and a 4-week TV burst in May 2025 used as the natural experiment for the DiD example.

## Stack

Python 3.10 · Streamlit · scikit-learn · statsmodels · scipy · Plotly

## Run locally

```bash
git clone https://github.com/Gemmagf/marketing-science-lab.git
cd marketing-science-lab
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tests

```bash
pip install pytest
pytest -q
```

## Project layout

```
marketing-science-lab/
├── streamlit_app.py            # landing page
├── pages/                      # multi-page Streamlit nav
├── src/
│   ├── data_generation.py      # synthetic D2C panel + ground truth
│   ├── mmm.py                  # MMM (Ridge fallback, ROI, optimiser)
│   ├── causal.py               # DiD + Synthetic Control + PSM
│   ├── experiments.py          # power, MDE, sample size, sequential
│   └── viz.py                  # Plotly helpers
├── tests/                      # pytest suite (60+ tests, runs in <5s)
├── requirements.txt
└── LICENSE                     # MIT
```

## Methodology

Formulas, design choices and references are on the in-app
[Methodology page](pages/4_📚_Methodology.py).

## Author

**Gemma Garcia de la Fuente** — Senior Data Scientist & Product Owner.

[LinkedIn](https://www.linkedin.com/in/gemmagardelafuente/) · [Portfolio](https://github.com/Gemmagf)

## License

MIT — see [LICENSE](LICENSE).

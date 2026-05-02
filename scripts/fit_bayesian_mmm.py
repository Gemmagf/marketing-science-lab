"""Offline Bayesian MMM fit — persists the posterior to a JSON file.

Run locally with PyMC-Marketing installed:

    pip install pymc-marketing>=0.7
    python3 scripts/fit_bayesian_mmm.py

The Streamlit app reads ``assets/mmm_posterior.json`` when present and
shows the Bayesian results; if missing it falls back to the Ridge / NNLS
MMM that ships in ``src/mmm.py``. This separation means PyMC stays out of
``requirements.txt`` and Streamlit Cloud builds in seconds.

Production pattern: train offline (here), serve online (Streamlit).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import CHANNELS, generate_dataset


def _import_pymc_marketing():
    try:
        from pymc_marketing.mmm import MMM as PyMCMMM
        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        return PyMCMMM, GeometricAdstock, LogisticSaturation
    except ImportError as e:
        raise SystemExit(
            "pymc-marketing is required for the Bayesian fit. "
            "Install with: pip install 'pymc-marketing>=0.7'\n"
            f"(Underlying error: {e})"
        )


def fit_and_dump(
    output_path: Path,
    seed: int = 42,
    chains: int = 2,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
) -> None:
    PyMCMMM, GeometricAdstock, LogisticSaturation = _import_pymc_marketing()

    print(f"Generating synthetic dataset (seed={seed})...")
    df, truth = generate_dataset(seed=seed)

    channel_cols = [c.name for c in CHANNELS]
    control_cols = ["is_weekend", "is_holiday", "temperature_c", "promo_pct"]

    print(f"Fitting Bayesian MMM ({chains} chains × {draws} draws)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mmm = PyMCMMM(
            date_column="date",
            channel_columns=channel_cols,
            control_columns=control_cols,
            adstock=GeometricAdstock(l_max=21),
            saturation=LogisticSaturation(),
            yearly_seasonality=4,
        )
        mmm.fit(
            X=df[["date", *channel_cols, *control_cols]],
            y=df["units_sold"],
            chains=chains,
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )

    print("Extracting posterior summaries...")
    # pymc-marketing 0.19: mmm.fit_result is already the posterior xarray Dataset
    fit_result = mmm.fit_result
    posterior = fit_result.posterior if hasattr(fit_result, "posterior") else fit_result

    # Channel and target scalers — needed to un-normalise the saturation_beta
    # back to the original units (units sold per CHF or per send).
    channel_scales: dict[str, float] = {}
    target_scale: float = 1.0
    try:
        for c in channel_cols:
            channel_scales[c] = float(df[c].max())
        target_scale = float(df["units_sold"].max())
    except Exception as e:
        print(f"  could not derive scales from data ({e})")

    def _summary(var: str, dim_name: str | None = None):
        """Return per-channel mean / 5% / 95% as dict keyed by channel."""
        arr = posterior[var]
        # mean & quantiles across chain+draw
        mean = arr.mean(dim=["chain", "draw"])
        q05 = arr.quantile(0.05, dim=["chain", "draw"])
        q95 = arr.quantile(0.95, dim=["chain", "draw"])
        if dim_name and dim_name in arr.dims:
            return {
                str(c): {
                    "mean": float(mean.sel({dim_name: c})),
                    "q05": float(q05.sel({dim_name: c})),
                    "q95": float(q95.sel({dim_name: c})),
                }
                for c in arr.coords[dim_name].values
            }
        return {"mean": float(mean), "q05": float(q05), "q95": float(q95)}

    out: dict = {
        "seed": seed,
        "chains": chains,
        "draws": draws,
        "tune": tune,
        "fit_method": "pymc_marketing",
        "saturation_kind": "logistic",
        "n_obs": len(df),
        "channels": channel_cols,
        "controls": control_cols,
        "channel_scales": channel_scales,
        "target_scale": target_scale,
        "ground_truth": {
            "betas": truth.betas,
            "decay": truth.decay,
            "half_sat": truth.half_sat,
        },
    }

    # PyMC-Marketing variable names. These vary slightly across versions.
    # Try the common ones; tolerate misses.
    var_dim_pairs = [
        ("saturation_beta", "channel"),
        ("adstock_alpha", "channel"),
        ("saturation_lam", "channel"),
        ("intercept", None),
        ("gamma_control", "control"),
        ("y_sigma", None),
    ]
    for var, dim in var_dim_pairs:
        if var in posterior:
            out[var] = _summary(var, dim)
            print(f"  ✓ {var}")
        else:
            print(f"  - {var} not in posterior, skipping")

    # Compute prediction R² and MAPE on the in-sample fit
    try:
        yhat = mmm.predict(df[["date", *channel_cols, *control_cols]])
        yhat = np.asarray(yhat).ravel()
    except Exception as e:
        print(f"  predict() failed ({e}); R²/MAPE skipped")
        yhat = None
    if yhat is not None:
        y = df["units_sold"].to_numpy(dtype=float)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out["r_squared"] = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        mask = y > 0
        out["mape"] = float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))
    else:
        out["r_squared"] = 0.0
        out["mape"] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print(f"R² = {out['r_squared']:.3f}, MAPE = {out['mape']:.2%}")

    # Show recovery vs ground truth in ORIGINAL scale (un-normalise β)
    if "saturation_beta" in out and channel_scales and target_scale > 0:
        print("\nβ recovery vs ground truth (units per CHF/send):")
        print(f"  {'channel':<18}{'truth':>10}{'mean':>12}{'5%':>10}{'95%':>10}{'err':>8}")
        for ch in channel_cols:
            t = truth.betas[ch]
            est = out["saturation_beta"][ch]
            cs = channel_scales[ch]
            scale = target_scale / cs if cs > 0 else 0
            mean_orig = est["mean"] * scale
            q05_orig = est["q05"] * scale
            q95_orig = est["q95"] * scale
            err = (mean_orig - t) / t * 100 if t else 0
            print(
                f"  {ch:<18}{t:>10,.0f}{mean_orig:>12,.0f}"
                f"{q05_orig:>10,.0f}{q95_orig:>10,.0f}{err:>7.0f}%"
            )

    if "adstock_alpha" in out:
        print("\nλ (adstock decay) recovery:")
        print(f"  {'channel':<18}{'truth':>8}{'mean':>8}{'5%':>8}{'95%':>8}")
        for ch in channel_cols:
            t = truth.decay[ch]
            est = out["adstock_alpha"][ch]
            print(f"  {ch:<18}{t:>8.2f}{est['mean']:>8.2f}{est['q05']:>8.2f}{est['q95']:>8.2f}")


if __name__ == "__main__":
    fit_and_dump(
        output_path=ROOT / "assets" / "mmm_posterior.json",
        seed=42,
    )

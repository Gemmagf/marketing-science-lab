"""Plotly helpers — single source of truth for palette, layout and style."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Brand-leaning palette: deep navy + On-style coral as accent
PALETTE = {
    "primary":   "#0A2540",   # deep navy
    "accent":    "#FF553F",   # coral
    "muted":     "#7A8AA0",
    "good":      "#1F9D55",
    "warn":      "#E0A800",
    "bg":        "#FFFFFF",
    "panel":     "#F4F6FA",
}

CHANNEL_COLOURS: dict[str, str] = {
    "tv_spend":      "#0A2540",
    "search_spend":  "#FF553F",
    "social_spend":  "#1F9D55",
    "display_spend": "#E0A800",
    "email_sends":   "#7A4DC8",
    "ooh_spend":     "#7A8AA0",
}


def _layout(title: str | None = None) -> dict:
    return dict(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=18, color=PALETTE["primary"])),
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor=PALETTE["bg"],
        font=dict(family="sans-serif", color=PALETTE["primary"]),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor=PALETTE["panel"]),
    )


def line(
    df: pd.DataFrame,
    x: str,
    y: str | Sequence[str],
    title: str | None = None,
    colours: Mapping[str, str] | None = None,
) -> go.Figure:
    """Multi-line chart with consistent styling."""
    fig = go.Figure()
    ys = [y] if isinstance(y, str) else list(y)
    for col in ys:
        colour = (colours or {}).get(col) or CHANNEL_COLOURS.get(col, PALETTE["primary"])
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], mode="lines", name=col, line=dict(color=colour, width=2),
        ))
    fig.update_layout(**_layout(title))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=PALETTE["panel"], zerolinecolor=PALETTE["panel"])
    return fig


def stacked_area(
    df: pd.DataFrame,
    x: str,
    ys: Sequence[str],
    baseline: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Stacked area for channel decomposition."""
    fig = go.Figure()
    if baseline is not None:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[baseline], mode="lines", name=baseline,
            line=dict(color=PALETTE["muted"], width=1),
            stackgroup="one",
        ))
    for col in ys:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], mode="lines", name=col,
            line=dict(width=0),
            stackgroup="one",
            fillcolor=CHANNEL_COLOURS.get(col, PALETTE["primary"]),
        ))
    fig.update_layout(**_layout(title))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=PALETTE["panel"])
    return fig


def bar_compare(
    labels: Sequence[str],
    values_a: Sequence[float],
    values_b: Sequence[float],
    name_a: str = "Current",
    name_b: str = "Optimal",
    title: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(labels), y=list(values_a), name=name_a, marker_color=PALETTE["muted"]))
    fig.add_trace(go.Bar(x=list(labels), y=list(values_b), name=name_b, marker_color=PALETTE["accent"]))
    fig.update_layout(barmode="group", **_layout(title))
    fig.update_yaxes(gridcolor=PALETTE["panel"])
    return fig


def saturation_curve(
    spend_grid: np.ndarray,
    response: np.ndarray,
    current_spend: float | None = None,
    title: str | None = None,
    colour: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spend_grid, y=response, mode="lines",
        line=dict(color=colour or PALETTE["primary"], width=2),
        name="response",
    ))
    if current_spend is not None:
        fig.add_vline(
            x=current_spend, line_color=PALETTE["accent"], line_dash="dash",
            annotation_text=f"Current: {current_spend:,.0f}", annotation_position="top right",
        )
    fig.update_layout(**_layout(title))
    fig.update_xaxes(title="Spend (CHF)", showgrid=False)
    fig.update_yaxes(title="Incremental units", gridcolor=PALETTE["panel"])
    return fig


def beta_recovery(
    truth: Mapping[str, float],
    estimated: Mapping[str, float],
    title: str = "Recovered β vs ground truth",
) -> go.Figure:
    channels = list(truth.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=channels, y=[truth[c] for c in channels], name="Ground truth",
        marker_color=PALETTE["primary"],
    ))
    fig.add_trace(go.Bar(
        x=channels, y=[estimated[c] for c in channels], name="Estimated",
        marker_color=PALETTE["accent"],
    ))
    fig.update_layout(barmode="group", **_layout(title))
    fig.update_yaxes(gridcolor=PALETTE["panel"])
    return fig


def did_plot(
    df: pd.DataFrame,
    date_col: str,
    treated_col: str,
    control_col: str,
    intervention,
    title: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[treated_col], mode="lines",
        name="Treated", line=dict(color=PALETTE["accent"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[control_col], mode="lines",
        name="Control", line=dict(color=PALETTE["primary"], width=2, dash="dot"),
    ))
    # `intervention` is a date string for daily axes, or an int for weekly axes
    if isinstance(intervention, (int, float, np.integer, np.floating)):
        x_intervention = float(intervention)
    else:
        x_intervention = pd.Timestamp(intervention)
    fig.add_vline(
        x=x_intervention, line_color=PALETTE["muted"], line_dash="dash",
        annotation_text="Intervention", annotation_position="top right",
    )
    fig.update_layout(**_layout(title))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=PALETTE["panel"])
    return fig


def power_curves(
    sample_sizes: np.ndarray,
    powers: dict[str, np.ndarray],
    title: str = "Power vs sample size",
) -> go.Figure:
    fig = go.Figure()
    accent_cycle = [PALETTE["primary"], PALETTE["accent"], PALETTE["good"], PALETTE["warn"]]
    for i, (label, p) in enumerate(powers.items()):
        fig.add_trace(go.Scatter(
            x=sample_sizes, y=p, mode="lines", name=label,
            line=dict(color=accent_cycle[i % len(accent_cycle)], width=2),
        ))
    fig.add_hline(y=0.8, line_color=PALETTE["muted"], line_dash="dash",
                  annotation_text="80% power", annotation_position="bottom right")
    fig.update_layout(**_layout(title))
    fig.update_xaxes(title="Sample size per arm", showgrid=False)
    fig.update_yaxes(title="Power", range=[0, 1.05], gridcolor=PALETTE["panel"])
    return fig

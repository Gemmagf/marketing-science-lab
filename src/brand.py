"""AlpSel — brand identity and copy for the case-study framing.

Single source of truth for product name, palette, channel labels and the
recurring "Question / Decision / Missed opportunity" structure used on
every page.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import streamlit as st


# --- Identity ------------------------------------------------------------

BRAND_NAME = "AlpSel"
BRAND_TAGLINE = "Swiss premium outdoor — sold direct."
BRAND_DESCRIPTION = (
    "AlpSel is a (fictional) Switzerland-based D2C retailer of premium "
    "outdoor and performance gear. The marketing science decisions on this "
    "site are made on a fully simulated dataset designed to mirror the "
    "spend, seasonality and channel behaviour of a real European D2C brand."
)

# Palette — alpine navy + glacier white + lift coral
COLOURS = {
    "primary":   "#0A2540",   # alpine navy
    "accent":    "#FF553F",   # lift coral
    "good":      "#1F9D55",   # forest
    "warn":      "#E0A800",   # lift yellow
    "danger":    "#D64545",
    "muted":     "#7A8AA0",
    "panel":     "#F4F6FA",
    "bg":        "#FFFFFF",
    "ink":       "#0A2540",
}

# Friendly channel labels for the dashboards
CHANNEL_LABELS: dict[str, str] = {
    "tv_spend":      "TV (linear & CTV)",
    "search_spend":  "Paid search",
    "social_spend":  "Social (Meta + TikTok)",
    "display_spend": "Display & retargeting",
    "email_sends":   "Email & CRM",
    "ooh_spend":     "OOH (DOOH + transit)",
}

# Realistic-ish CPC / CPM caps so the dashboards feel like a real brand
UNIT_PRICE_CHF = 165.0  # avg basket value


# --- Case-study helpers --------------------------------------------------

@dataclass(frozen=True)
class Decision:
    headline: str            # one line: the recommendation
    detail: str              # one paragraph: the action to take
    impact_chf: float        # incremental revenue / lost revenue (CHF)
    confidence: str          # "high" | "medium" | "low"
    risks: tuple[str, ...]   # bulleted caveats


def render_question(text: str, sub: str | None = None) -> None:
    """Top-of-page business question — the framing every consultant uses."""
    st.markdown(
        f"""
        <div style="background: {COLOURS["panel"]}; padding: 1.5rem 1.8rem;
                    border-left: 4px solid {COLOURS["primary"]}; border-radius: 8px;
                    margin-bottom: 1.5rem;">
            <div style="font-size: 0.8rem; font-weight: 700; letter-spacing: 0.12em;
                        text-transform: uppercase; color: {COLOURS["accent"]};
                        margin-bottom: 0.4rem;">The question</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: {COLOURS["ink"]};
                        line-height: 1.3;">{text}</div>
            {f'<div style="font-size: 1rem; color: {COLOURS["muted"]}; margin-top: 0.6rem;">{sub}</div>' if sub else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision(d: Decision) -> None:
    """Final block on every page — the decision the brand should take."""
    sign = "+" if d.impact_chf >= 0 else ""
    impact_color = COLOURS["good"] if d.impact_chf >= 0 else COLOURS["danger"]
    impact_label = "Incremental revenue" if d.impact_chf >= 0 else "Revenue at risk"
    conf_color = {"high": COLOURS["good"], "medium": COLOURS["warn"], "low": COLOURS["danger"]}[d.confidence]

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLOURS["primary"]} 0%, #1A3550 100%);
                    color: white; padding: 2rem 2.2rem; border-radius: 12px;
                    margin-top: 2rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: baseline; gap: 1rem; margin-bottom: 0.8rem;">
                <span style="font-size: 0.8rem; font-weight: 700; letter-spacing: 0.14em;
                             text-transform: uppercase; color: {COLOURS["accent"]};">Decision</span>
                <span style="font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
                             text-transform: uppercase; padding: 0.15rem 0.6rem;
                             background: {conf_color}; border-radius: 12px;">{d.confidence} confidence</span>
            </div>
            <div style="font-size: 1.7rem; font-weight: 700; line-height: 1.25;
                        margin-bottom: 0.7rem;">{d.headline}</div>
            <div style="font-size: 1rem; line-height: 1.55; color: #C8D2E0;
                        margin-bottom: 1.2rem;">{d.detail}</div>
            <div style="display: flex; gap: 2rem; align-items: center;">
                <div>
                    <div style="font-size: 0.75rem; color: #8AA0BC; letter-spacing: 0.1em;
                                text-transform: uppercase;">{impact_label}</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: {impact_color};
                                line-height: 1; margin-top: 0.2rem;">
                        {sign}CHF {abs(d.impact_chf):,.0f}
                    </div>
                </div>
                <div style="border-left: 1px solid #2A4060; padding-left: 1.5rem; flex: 1;">
                    <div style="font-size: 0.75rem; color: #8AA0BC; letter-spacing: 0.1em;
                                text-transform: uppercase; margin-bottom: 0.3rem;">Risks & caveats</div>
                    <ul style="margin: 0; padding-left: 1.2rem; color: #C8D2E0; font-size: 0.9rem;">
                        {"".join(f"<li>{r}</li>" for r in d.risks)}
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_missed_opportunity(label: str, chf: float, sub: str | None = None) -> None:
    """Big red callout — the cost of doing nothing."""
    st.markdown(
        f"""
        <div style="background: #FFF4F2; border: 1px solid #F5C6BD;
                    padding: 1.4rem 1.7rem; border-radius: 10px;
                    margin: 1.5rem 0;">
            <div style="font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
                        text-transform: uppercase; color: {COLOURS["danger"]};
                        margin-bottom: 0.35rem;">⚠ Missed opportunity</div>
            <div style="display: flex; align-items: baseline; gap: 1rem;">
                <div style="font-size: 2.4rem; font-weight: 700; color: {COLOURS["danger"]};
                            line-height: 1;">CHF {chf:,.0f}</div>
                <div style="font-size: 1.05rem; color: {COLOURS["ink"]};
                            font-weight: 500;">{label}</div>
            </div>
            {f'<div style="font-size: 0.92rem; color: {COLOURS["muted"]}; margin-top: 0.6rem; line-height: 1.4;">{sub}</div>' if sub else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_synthetic_disclaimer() -> None:
    """Tiny, unobtrusive note at the bottom of each page."""
    st.markdown(
        f"""
        <div style="font-size: 0.78rem; color: {COLOURS["muted"]};
                    margin-top: 3rem; padding-top: 1rem;
                    border-top: 1px solid {COLOURS["panel"]};">
            AlpSel is a fictional brand. All numbers come from a calibrated synthetic
            dataset (see <em>Methodology</em>) so every recommendation can be checked
            against the known data-generating process.
        </div>
        """,
        unsafe_allow_html=True,
    )

"""AlpSel — brand identity + native-Streamlit case-study helpers.

Single source of truth for product name, palette, channel labels and the
recurring "Question / Decision / Missed opportunity" structure used on
every page. All rendering uses native Streamlit components (no custom
HTML) for consistent, polished output.
"""
from __future__ import annotations

from dataclasses import dataclass

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

COLOURS = {
    "primary":   "#0A2540",
    "accent":    "#FF553F",
    "good":      "#1F9D55",
    "warn":      "#E0A800",
    "danger":    "#D64545",
    "muted":     "#7A8AA0",
    "panel":     "#F4F6FA",
    "bg":        "#FFFFFF",
    "ink":       "#0A2540",
}

CHANNEL_LABELS: dict[str, str] = {
    "tv_spend":      "TV (linear & CTV)",
    "search_spend":  "Paid search",
    "social_spend":  "Social (Meta + TikTok)",
    "display_spend": "Display & retargeting",
    "email_sends":   "Email & CRM",
    "ooh_spend":     "OOH (DOOH + transit)",
    "leaflet_spend": "Leaflets & in-store promo",
}

UNIT_PRICE_CHF = 165.0


# --- Case-study helpers --------------------------------------------------

@dataclass(frozen=True)
class Decision:
    headline: str
    detail: str
    impact_chf: float
    confidence: str           # "high" | "medium" | "low"
    risks: tuple[str, ...]


def render_page_chrome(case_no: str, total_cases: str, title: str) -> None:
    """Top of every case-study page: brand mark + case number + title."""
    st.caption(f"**{BRAND_NAME}** · Case {case_no} of {total_cases}")
    st.title(title)


def render_question(text: str, sub: str | None = None) -> None:
    """Top-of-page business question."""
    st.subheader(f"📋 The question")
    st.markdown(f"**{text}**")
    if sub:
        st.caption(sub)
    st.write("")


def render_decision(d: Decision) -> None:
    """Final block on every page — the decision the brand should take."""
    st.write("")
    st.subheader("✅ Decision")

    # Headline + detail in a bordered container (native Streamlit)
    with st.container(border=True):
        st.markdown(f"### {d.headline}")
        st.markdown(d.detail)

        st.write("")
        c1, c2, c3 = st.columns([1, 1, 2])
        sign = "+" if d.impact_chf >= 0 else "-"
        impact_label = "Incremental revenue" if d.impact_chf >= 0 else "Revenue at risk"
        c1.metric(impact_label, f"{sign}CHF {abs(d.impact_chf):,.0f}")

        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}[d.confidence]
        c2.metric("Confidence", f"{conf_emoji} {d.confidence.title()}")

        with c3:
            st.markdown("**Risks & caveats**")
            for r in d.risks:
                st.markdown(f"- {r}")


def render_missed_opportunity(label: str, chf: float, sub: str | None = None) -> None:
    """Big red callout — the cost of doing nothing."""
    st.write("")
    msg = f"### ⚠ Missed opportunity\n\n**CHF {chf:,.0f}** — {label}"
    if sub:
        msg += f"\n\n{sub}"
    st.error(msg)


def render_synthetic_disclaimer() -> None:
    """Tiny note at the bottom of each page."""
    st.write("")
    st.caption(
        f"AlpSel is a fictional brand. All numbers come from a calibrated "
        f"synthetic dataset (see Methodology) so every recommendation can be "
        f"checked against the known data-generating process."
    )

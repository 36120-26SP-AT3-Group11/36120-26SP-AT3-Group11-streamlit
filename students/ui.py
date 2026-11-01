"""Shared design kit for the Group 11 app.

One look for every page: the colours, fonts, cards, gauges and messages all come from here.

    from students import ui
    ui.apply_theme()                     # once, in app/main.py
    ui.section("What drives the forecast", "Optional caption.", accent=ui.ACCENTS["A"])
    ui.score_ring(62.4, "#22c55e")       # donut showing a 0-100 score
"""

import altair as alt
import pandas as pd
import streamlit as st

# ---------- Palette ----------
BG = "#0b1020"
SURFACE = "#121a2e"
BORDER = "#24304d"
TEXT = "#e6e9ef"
MUTED = "#94a3b8"
BRAND_GRADIENT = "linear-gradient(120deg, #6366f1 0%, #22d3ee 50%, #a855f7 100%)"

# One accent colour per student, used for page banners, card borders and charts
ACCENTS = {"A": "#10b981", "B": "#f43f5e", "C": "#8b5cf6", "D": "#f59e0b"}

ACCENT = "#22d3ee"       # forecast markers, single-series lines
NEUTRAL_BAR = "#818cf8"  # feature importance bars
GOOD = "#22c55e"
BAD = "#ef4444"

# Five-step scale for 0-100 scores, worst to best (reverse it when higher is worse)
SCALE_5 = ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e"]


# ---------- Global styling ----------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stCaption {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.01em; }}

/* App background: soft glow in the corners */
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(900px 400px at 0% -10%, rgba(99,102,241,0.18), transparent 60%),
    radial-gradient(800px 400px at 100% 0%, rgba(34,211,238,0.12), transparent 60%),
    {BG};
}}
[data-testid="stHeader"] {{
  background: rgba(11,16,32,0.88);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid {BORDER};
}}
/* Centre the top navigation links in the header */
[data-testid="stToolbar"] .rc-overflow {{
  justify-content: center;
}}
@media (min-width: 1500px) {{
  /* offset by the width of the Share/GitHub buttons on the right so the links sit at the true centre */
  [data-testid="stToolbar"] .rc-overflow {{ padding-left: 200px; }}
}}
/* Bigger top navigation links */
[data-testid="stTopNavLink"] {{
  font-size: 1.05rem;
  padding: 0.4rem 0.8rem;
  border-radius: 10px;
}}
[data-testid="stTopNavLink"] span {{
  font-size: 1.05rem;
}}
[data-testid="stTopNavLink"] [data-testid="stIconMaterial"] {{
  font-size: 1.3rem;
}}
/* Slightly bigger coin buttons */
.st-key-coin_pills [data-testid="stButtonGroup"] button {{
  font-size: 0.98rem;
  padding: 0.4rem 1rem;
  min-height: 2.4rem;
}}
.st-key-coin_pills [data-testid="stButtonGroup"] button p {{
  font-size: 0.98rem;
}}
/* Centre the coin picker, also when it wraps onto two lines */
.st-key-coin_pills {{
  align-self: center;
}}
.st-key-coin_pills [data-testid="stButtonGroup"] > div {{
  justify-content: center;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
  border-right: 1px solid {BORDER};
}}

/* Metrics look like cards */
[data-testid="stMetric"] {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 14px;
  padding: 14px 16px;
  min-height: 132px;   /* same size whether or not a metric shows a change */
}}
[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; }}

/* Bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px; }}

/* Expanders */
[data-testid="stExpander"] details {{ border-radius: 12px; border-color: {BORDER}; background: {SURFACE}; }}

/* Our own components */
.g11-hero {{
  border-radius: 22px; padding: 28px 30px; margin-bottom: 18px; text-align: center;
  background: {BRAND_GRADIENT};
  box-shadow: 0 18px 50px rgba(99,102,241,0.25);
}}
.g11-hero h1 {{ color: white; margin: 0; font-size: 2.2rem; line-height: 1.15; }}
.g11-hero p {{ color: rgba(255,255,255,0.92); margin: 8px 0 14px; font-size: 1.02rem; }}
.g11-chip {{
  display: inline-block; padding: 4px 12px; margin: 0 6px 6px 0; border-radius: 999px;
  background: rgba(255,255,255,0.18); color: white; font-size: 0.82rem; font-weight: 600;
  border: 1px solid rgba(255,255,255,0.25);
}}
.g11-banner {{
  border-radius: 18px; padding: 18px 22px; margin: 4px 0 16px;
  background: {SURFACE}; border: 1px solid {BORDER};
  border-left: 6px solid var(--accent);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}}
.g11-banner .kicker {{ color: var(--accent); font-weight: 700; font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.08em; }}
.g11-banner .title {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700;
  color: {TEXT}; margin: 2px 0 4px; }}
.g11-banner .desc {{ color: {MUTED}; font-size: 0.95rem; margin: 0; }}
.g11-card-head {{
  display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
}}
.g11-dot {{ width: 12px; height: 12px; border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 12px var(--accent); }}
.g11-card-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05rem; color: {TEXT};
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }}
.g11-card-owner {{ margin-left: auto; color: {MUTED}; font-size: 0.78rem; white-space: nowrap; }}
.g11-section {{
  display: flex; align-items: center; gap: 10px; margin: 26px 0 4px;
}}
.g11-section .bar {{ width: 6px; height: 26px; border-radius: 4px; background: var(--accent); }}
.g11-section .label {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; color: {TEXT}; }}
.g11-level {{
  display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700;
  background: color-mix(in srgb, var(--level) 22%, transparent); color: var(--level);
  border: 1px solid color-mix(in srgb, var(--level) 55%, transparent);
}}
.g11-footer {{ color: {MUTED}; font-size: 0.8rem; text-align: center; margin-top: 30px; }}
</style>
"""


def apply_theme() -> None:
    """Inject the shared styling. Call once at the top of app/main.py."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------- Page structure ----------
def hero(title: str, subtitle: str, chips: list[str]) -> None:
    """Gradient banner for the home page."""
    chip_html = "".join(f"<span class='g11-chip'>{c}</span>" for c in chips)
    st.markdown(f"<div class='g11-hero'><h1>{title}</h1><p>{subtitle}</p>{chip_html}</div>",
                unsafe_allow_html=True)


def page_banner(kicker: str, title: str, description: str, accent: str) -> None:
    """Same banner at the top of every student page, in that student's accent colour."""
    st.markdown(
        f"<div class='g11-banner' style='--accent:{accent}'>"
        f"<div class='kicker'>{kicker}</div><div class='title'>{title}</div>"
        + (f"<p class='desc'>{description}</p>" if description else "")
        + "</div>", unsafe_allow_html=True)


def card_header(title: str, owner: str, accent: str) -> None:
    """Coloured dot, title and owner, for the home-page cards."""
    st.markdown(
        f"<div class='g11-card-head' style='--accent:{accent}'><span class='g11-dot'></span>"
        f"<span class='g11-card-title'>{title}</span>"
        f"<span class='g11-card-owner'>{owner}</span></div>", unsafe_allow_html=True)


def section(title: str, caption: str | None = None, accent: str = ACCENT) -> None:
    """Section title with a coloured bar, and an optional caption."""
    st.markdown(f"<div class='g11-section' style='--accent:{accent}'><span class='bar'></span>"
                f"<span class='label'>{title}</span></div>", unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def level_badge(label: str, colour: str) -> None:
    st.markdown(f"<span class='g11-level' style='--level:{colour}'>&#9679; {label}</span>",
                unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown(f"<div class='g11-footer'>{text}</div>", unsafe_allow_html=True)


# ---------- Score visuals ----------
def band_for(score: float, bands: list[tuple[float, float, str, str]]) -> tuple[str, str]:
    """(label, colour) of the band that contains the score."""
    for start, end, label, colour in bands:
        if start <= score < end:
            return label, colour
    return bands[-1][2], bands[-1][3]


def score_ring(score: float, colour: str, size: int = 190) -> alt.Chart:
    """Donut filled up to the score (0-100), with the number in the middle."""
    data = pd.DataFrame({"part": ["score", "rest"], "value": [score, 100 - score]})
    ring = alt.Chart(data).mark_arc(innerRadius=size * 0.34, cornerRadius=8).encode(
        theta=alt.Theta("value:Q", stack=True, sort=None),
        order=alt.Order("part:N", sort="descending"),
        color=alt.Color("part:N", scale=alt.Scale(domain=["score", "rest"],
                                                  range=[colour, "#1f2a44"]), legend=None),
        tooltip=alt.value(None))
    label = alt.Chart(pd.DataFrame({"t": [f"{score:.1f}"]})).mark_text(
        fontSize=size * 0.17, fontWeight="bold", color=TEXT, font="Space Grotesk").encode(text="t:N")
    sub = alt.Chart(pd.DataFrame({"t": ["/ 100"]})).mark_text(
        fontSize=size * 0.07, color=MUTED, dy=size * 0.13).encode(text="t:N")
    return (ring + label + sub).properties(width=size, height=size).configure_view(strokeWidth=0)


def band_gauge(score: float, bands: list[tuple[float, float, str, str]], height: int = 70) -> alt.Chart:
    """Horizontal 0-100 bar with coloured bands and a marker at the score."""
    data = pd.DataFrame([{"start": s, "end": min(e, 100), "band": label} for s, e, label, _ in bands])
    colours = alt.Scale(domain=[b[2] for b in bands], range=[b[3] for b in bands])
    base = alt.Chart(data).mark_bar(height=22, opacity=0.75, cornerRadius=4).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[0, 100]), title=None),
        x2="end:Q",
        color=alt.Color("band:N", scale=colours, legend=None),
        tooltip=["band:N", "start:Q", "end:Q"])
    point = pd.DataFrame({"score": [score]})
    tick = alt.Chart(point).mark_tick(thickness=4, size=34, color="white").encode(
        x="score:Q", tooltip=alt.Tooltip("score:Q", title="Forecast", format=".1f"))
    arrow = alt.Chart(point).mark_point(shape="triangle-down", size=200, filled=True,
                                        color="white", yOffset=-20).encode(x="score:Q")
    return (base + tick + arrow).properties(height=height)


# ---------- Inputs and errors ----------
def missing_inputs_warning(missing: list[str], labels: dict[str, str]) -> None:
    if missing:
        names = ", ".join(labels.get(f, f) for f in missing)
        st.warning(f"Some inputs were unavailable, so this forecast is less accurate than usual. "
                   f"Missing: {names}. This usually means a teammate's service is waking up or "
                   f"down; the model is trained to handle it.", icon=":material/info:")


def inputs_table(features_used: dict, feature_sources: dict, labels: dict[str, str]) -> None:
    with st.expander("Inputs used for this forecast", icon=":material/table_chart:"):
        table = pd.DataFrame({
            "Feature": [labels.get(f, f) for f in features_used],
            "Value": [None if v is None else round(v, 4) for v in features_used.values()],
            "Source": [feature_sources.get(f, "") for f in features_used],
        })
        st.dataframe(table, hide_index=True, width="stretch")


def api_error(message: str, retry_key: str, caches: tuple = ()) -> None:
    """Error box with a retry button that clears only this page's cached API calls."""
    st.error(message, icon=":material/cloud_off:")
    if st.button("Try again", key=retry_key, icon=":material/refresh:"):
        for fetch in caches:
            fetch.clear()
        st.rerun()


# ---------- Model information ----------
def importance_chart(importance: dict[str, float], labels: dict[str, str], accent: str = NEUTRAL_BAR,
                     height: int = 200) -> None:
    imp = (pd.DataFrame({"feature": list(importance), "importance": list(importance.values())})
           .assign(label=lambda d: d["feature"].map(lambda f: labels.get(f, f)))
           .sort_values("importance", ascending=False))
    bars = alt.Chart(imp).mark_bar(color=accent, cornerRadiusEnd=6).encode(
        x=alt.X("importance:Q", title="Share of model gain", axis=alt.Axis(format="%")),
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
        tooltip=["label:N", alt.Tooltip("importance:Q", format=".1%")])
    text = bars.mark_text(align="left", dx=6, color=TEXT).encode(text=alt.Text("importance:Q", format=".0%"))
    st.altair_chart((bars + text).properties(height=height), width="stretch")


def limitations(text: str) -> None:
    """Same closing note on every page."""
    st.info(f"**Limitations:** {text} This is decision support, not financial advice.",
            icon=":material/lightbulb:")
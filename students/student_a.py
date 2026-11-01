"""Student A - Asset Performance Index (API) page.

All numbers come from the Student A FastAPI service over HTTP; no model runs in Streamlit.
Layout and colours come from the shared design kit in students/ui.py.

Use from the main app:
    from students import student_a
    student_a.render(token)       # full page (app/main.py draws the page banner)
    student_a.home_chart(token)   # compact visual for the home page

Run on its own for testing:
    streamlit run students/student_a.py
"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

try:
    from students import ui
except ImportError:  # running this file on its own
    import ui

API_URL = os.getenv("STUDENT_A_API_URL", "https://group11-studenta-api.onrender.com").rstrip("/")
TOKENS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XMR", "XLM", "TRX"]
TIMEOUT = 90  # free Render services can take up to a minute to wake up

TITLE = "Asset Performance Index"
ACCENT = ui.ACCENTS["A"]
DESCRIPTION = ("A 0-100 score combining the 20-day return (40%), trend strength (30%), trading "
               "volume compared to normal (20%) and price stability (10%). Higher means stronger "
               "performance. The model forecasts it 7 days ahead.")

# Higher is better, so the scale runs red to green
BANDS = [(0, 20, "Very weak", ui.SCALE_5[0]), (20, 40, "Weak", ui.SCALE_5[1]),
         (40, 60, "Neutral", ui.SCALE_5[2]), (60, 80, "Strong", ui.SCALE_5[3]),
         (80, 101, "Very strong", ui.SCALE_5[4])]

FEATURE_LABELS = {
    "Return20": "20-day return (Student A)",
    "TrendStrength": "Trend strength (Student B)",
    "RelativeVolume": "Relative volume (Student D)",
    "Volatility20": "20-day volatility (Student C)",
}

# The four parts of today's score, with their colours
COMPONENTS = [("Return strength", 40, "#22d3ee"), ("Trend strength", 30, "#a855f7"),
              ("Relative volume", 20, "#f59e0b"), ("Price stability", 10, "#10b981")]


# ---------- API calls ----------
class ApiError(Exception):
    pass


def _call(method: str, path: str, payload: dict | None = None) -> dict:
    try:
        response = requests.request(method, f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    except requests.Timeout:
        raise ApiError("The Asset Performance API did not respond in time. It may be waking up; "
                       "please try again in a minute.")
    except requests.RequestException:
        raise ApiError("The Asset Performance API could not be reached.")
    try:
        body = response.json()
    except ValueError:
        raise ApiError(f"The Asset Performance API returned an unexpected response "
                       f"(HTTP {response.status_code}).")
    if response.status_code != 200:
        raise ApiError(body.get("error", f"HTTP {response.status_code}") if isinstance(body, dict)
                       else f"HTTP {response.status_code}")
    return body


# Successful responses are cached; errors are raised (never cached), so a retry calls the API again.
@st.cache_data(ttl=600, show_spinner=False)
def get_prediction(token: str) -> dict:
    return _call("POST", "/predict", {"token": token, "forecast_horizon": 7})


@st.cache_data(ttl=600, show_spinner=False)
def get_features(token: str, days: int = 90) -> dict:
    return _call("POST", "/features", {"token": token, "days": days})


@st.cache_data(ttl=3600, show_spinner=False)
def get_metadata() -> dict:
    return _call("GET", "/model-metadata")


CACHES = (get_prediction, get_features, get_metadata)


def _safe(fetch, *args):
    # Run an API call; return (data, error message)
    try:
        return fetch(*args), None
    except ApiError as exc:
        return None, str(exc)


def _delta(pred: dict) -> str | None:
    current = pred.get("current_api")
    return None if current is None else f"{pred['prediction'] - current:+.1f} vs today"


def _components(features: dict) -> pd.DataFrame | None:
    # Points each part adds to today's score, from the brief's formula (not the model)
    if any(features.get(f) is None for f in FEATURE_LABELS):
        return None
    parts = [min(max((features["Return20"] + 0.20) / 0.40, 0), 1),
             min(max((features["TrendStrength"] + 0.10) / 0.20, 0), 1),
             min(features["RelativeVolume"], 2) / 2,
             1 - features["Volatility20"]]
    return pd.DataFrame([{"part": name, "points": weight * p, "max": weight, "colour": colour}
                         for (name, weight, colour), p in zip(COMPONENTS, parts)])


# ---------- Home-page visual ----------
def home_chart(token: str) -> None:
    with st.spinner("Getting the performance forecast..."):
        pred, error = _safe(get_prediction, token)
    if error:
        st.warning(f"Performance forecast unavailable: {error}")
        return
    label, _ = ui.band_for(pred["prediction"], BANDS)
    st.metric(f"{token} performance on {pred['prediction_date']}",
              f"{pred['prediction']:.1f} ({label})", delta=_delta(pred))
    st.altair_chart(ui.band_gauge(pred["prediction"], BANDS, height=50), width="stretch")


# ---------- Full page ----------
def render(token: str) -> None:
    st.caption(DESCRIPTION)

    # 1) Forecast
    with st.spinner("Getting the forecast (the first request can take up to a minute while the "
                    "service wakes up)..."):
        pred, error = _safe(get_prediction, token)

    if error:
        ui.api_error(f"Performance forecast unavailable. {error}", "retry_a", CACHES)
    else:
        score, current = pred["prediction"], pred.get("current_api")
        label, colour = ui.band_for(score, BANDS)

        with st.container(border=True):
            ring_col, info_col = st.columns([1, 2], vertical_alignment="center")
            with ring_col:
                st.altair_chart(ui.score_ring(score, colour), width="content")
            with info_col:
                ui.level_badge(label, colour)
                m1, m2 = st.columns(2)
                m1.metric(f"Forecast for {pred['prediction_date']}", f"{score:.1f}", delta=_delta(pred))
                m2.metric(f"Today ({pred['feature_date']})",
                          "n/a" if current is None else f"{current:.1f}")
                if current is not None:
                    direction = ("strengthen" if score > current + 1
                                 else "weaken" if score < current - 1 else "stay about the same")
                    st.write(f"Performance for **{token}** is expected to **{direction}** "
                             f"over the next 7 days.")
            st.altair_chart(ui.band_gauge(score, BANDS), width="stretch")

        ui.missing_inputs_warning(pred.get("missing_features", []), FEATURE_LABELS)

        # 2) What today's score is made of
        parts = _components(pred["features_used"])
        if parts is not None:
            ui.section("What today's score is made of",
                       "Points each part adds to today's score, out of its maximum.", ACCENT)
            base = alt.Chart(parts).encode(y=alt.Y("part:N", sort=None, title=None))
            track = base.mark_bar(color="#1f2a44", cornerRadiusEnd=6).encode(
                x=alt.X("max:Q", title="Points", scale=alt.Scale(domain=[0, 40])))
            fill = base.mark_bar(cornerRadiusEnd=6).encode(
                x="points:Q",
                color=alt.Color("colour:N", scale=None, legend=None),
                tooltip=["part:N", alt.Tooltip("points:Q", format=".1f"), alt.Tooltip("max:Q", title="Max")])
            text = base.mark_text(align="left", dx=6, color=ui.TEXT).encode(
                x="points:Q", text=alt.Text("points:Q", format=".1f"))
            st.altair_chart((track + fill + text).properties(height=200), width="stretch")

        ui.inputs_table(pred["features_used"], pred["feature_sources"], FEATURE_LABELS)

    # 3) Recent price and return (Student A's own features)
    ui.section("Recent price and 20-day return",
               "20-day return carries 40% of the score. Beyond +20% or below -20% it stops "
               "changing the score.", ACCENT)
    feats, error = _safe(get_features, token, 90)
    if error:
        st.warning(f"Price history unavailable. {error}")
    else:
        hist = pd.DataFrame(feats["history"])
        hist["date"] = pd.to_datetime(hist["date"])
        area = alt.Chart(hist).mark_area(
            line={"color": "#22d3ee"},
            color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                               stops=[alt.GradientStop(color="rgba(34,211,238,0.02)", offset=0),
                                      alt.GradientStop(color="rgba(34,211,238,0.45)", offset=1)])
        ).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("close:Q", title="USD", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("close:Q", title="Close", format=",.4f")],
        ).properties(height=240)
        returns = alt.Chart(hist.assign(direction=hist["Return20"].ge(0).map(
            {True: "Gain", False: "Loss"}))).mark_bar(cornerRadius=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Return20:Q", title="20-day return", axis=alt.Axis(format="%")),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["Gain", "Loss"],
                                                           range=[ui.GOOD, ui.BAD]), legend=None),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("Return20:Q", format=".2%")],
        ).properties(height=150)
        st.altair_chart(alt.vconcat(area, returns).resolve_scale(x="shared"), width="stretch")

    # 4) What drives the forecast
    ui.section("What drives the forecast",
               "Trend strength drives most predictions. Return20 overlaps with it heavily, "
               "while relative volume adds information no other input has.", ACCENT)
    meta, error = _safe(get_metadata)
    if error:
        st.warning(f"Model information unavailable. {error}")
    else:
        ui.importance_chart(meta["feature_importance"], FEATURE_LABELS, accent=ACCENT)
        m, b = meta["test_metrics_2025"], meta["baseline_metrics_2025"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Average error (2025)", f"{m['MAE']:.2f} pts",
                  delta=f"{m['MAE'] - b['MAE']:+.2f} vs baseline", delta_color="inverse")
        k2.metric("R² (2025)", f"{m['R2']:.2f}", delta=f"{m['R2'] - b['R2']:+.2f} vs baseline")
        k3.metric("Forecast horizon", f"{meta['forecast_horizon']} days")
        with st.expander("About the model", icon=":material/model_training:"):
            st.markdown(
                f"- **Algorithm:** {meta['algorithm']}, tuned with Optuna\n"
                f"- **Trained on:** daily data for all 8 coins, {meta['training_period']}\n"
                f"- **Baseline:** assuming next week's score equals today's "
                f"(average error {b['MAE']:.2f} points)\n"
                f"- **Missing inputs:** {meta['missing_value_handling']}")

    ui.limitations("Forecasts use past prices only and cannot anticipate sudden shocks. The model is "
                   "better at signalling the direction of performance than exact values, and rarely "
                   "predicts extreme scores. Read a high score alongside the Market Risk page.")


# ---------- Standalone preview: streamlit run students/student_a.py ----------
if __name__ == "__main__":
    st.set_page_config(page_title=TITLE, layout="wide")
    ui.apply_theme()
    selected = st.sidebar.selectbox("Cryptocurrency", TOKENS)
    ui.page_banner("Student A · XGBoost regression · 7 days ahead", f"{TITLE} · {selected}", "", ACCENT)
    render(selected)

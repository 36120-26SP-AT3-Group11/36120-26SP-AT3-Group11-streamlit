"""Student B - Market Risk Index (MRI) page.

All numbers come from the Student B FastAPI service over HTTP; no model runs in Streamlit.

Use from the main app:
    from students import student_b
    student_b.render(token)       # full page / tab
    student_b.home_chart(token)   # compact visual for the home page

Run on its own for testing:
    streamlit run students/student_b.py
"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("STUDENT_B_API_URL", "https://group11-studentb-api.onrender.com").rstrip("/")
TOKENS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XMR", "XLM", "TRX"]
# Free Render services sleep when idle; waking one (and the teammate APIs it calls) can take a minute.
TIMEOUT = 90

# Display bands for the 0-100 score (40+ matches the "high-risk week" threshold used in the notebook)
RISK_BANDS = [(20, "Low", "#2e7d32"), (40, "Moderate", "#f9a825"),
              (60, "High", "#ef6c00"), (101, "Very high", "#c62828")]

FEATURE_LABELS = {
    "Return20": "20-day return (Student A)",
    "Volatility20": "20-day volatility (Student C)",
    "MaxPrice20": "20-day high (Student C)",
    "GR": "Gap risk (Student C)",
    "DR": "Drawdown from 20-day high",
    "MRI_now": "Today's MRI",
    "TrendStrength": "Trend strength (Student B)",
}


# ---------- API calls ----------
class ApiError(Exception):
    """Raised when the Student B API is unreachable or returns an error."""


def _call(method: str, path: str, payload: dict | None = None) -> dict:
    try:
        response = requests.request(method, f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    except requests.Timeout:
        raise ApiError("The Market Risk API did not respond in time. It may be waking up; "
                       "please try again in a minute.")
    except requests.RequestException:
        raise ApiError("The Market Risk API could not be reached.")
    try:
        body = response.json()
    except ValueError:
        raise ApiError(f"The Market Risk API returned an unexpected response (HTTP {response.status_code}).")
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


def _safe(fetch, *args):
    """Run an API call; return (data, error message)."""
    try:
        return fetch(*args), None
    except ApiError as exc:
        return None, str(exc)


# ---------- Helpers ----------
def risk_band(score: float) -> tuple[str, str]:
    for upper, label, colour in RISK_BANDS:
        if score < upper:
            return label, colour
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def risk_gauge(score: float, height: int = 70) -> alt.Chart:
    """Horizontal 0-100 bar with coloured risk bands and a marker at the forecast."""
    bands = pd.DataFrame([{"start": lo, "end": min(hi, 100), "band": label}
                          for (lo, (hi, label, _)) in zip([0, 20, 40, 60], RISK_BANDS)])
    colours = alt.Scale(domain=[b[1] for b in RISK_BANDS], range=[b[2] for b in RISK_BANDS])
    base = alt.Chart(bands).mark_bar(height=22, opacity=0.55).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[0, 100]), title=None),
        x2="end:Q",
        color=alt.Color("band:N", scale=colours, legend=None),
        tooltip=["band:N", "start:Q", "end:Q"])
    # Bright blue marker: visible on both the light and dark Streamlit themes
    point = pd.DataFrame({"score": [score]})
    line = alt.Chart(point).mark_tick(thickness=4, size=34, color="#2196f3").encode(
        x="score:Q", tooltip=alt.Tooltip("score:Q", title="Forecast", format=".1f"))
    arrow = alt.Chart(point).mark_point(shape="triangle-down", size=180, filled=True, color="#2196f3",
                                        yOffset=-20).encode(x="score:Q")
    return (base + line + arrow).properties(height=height)


def _retry_button(key: str):
    if st.button("Try again", key=key):
        for fetch in (get_prediction, get_features, get_metadata):   # only this page's cache
            fetch.clear()
        st.rerun()


# ---------- Home-page visual ----------
def home_chart(token: str) -> None:
    """Compact MRI forecast for the home page (one visual per student)."""
    st.markdown("**Market Risk Index, 7 days ahead**")
    with st.spinner("Getting the risk forecast..."):
        pred, error = _safe(get_prediction, token)
    if error:
        st.warning(f"Risk forecast unavailable: {error}")
        return
    label, _ = risk_band(pred["prediction"])
    delta = (None if pred.get("current_mri") is None
             else f"{pred['prediction'] - pred['current_mri']:+.1f} vs today")
    st.metric(f"{token} risk on {pred['prediction_date']}", f"{pred['prediction']:.1f} ({label})",
              delta=delta, delta_color="inverse")
    st.altair_chart(risk_gauge(pred["prediction"], height=50), width="stretch")


# ---------- Full page ----------
def render(token: str) -> None:
    st.header(f"Market Risk Index: {token}")
    st.caption("Forecast of the Market Risk Index (MRI) 7 days ahead. MRI is a 0-100 score combining "
               "20-day volatility (40%), drawdown from the 20-day high (30%), downside 20-day "
               "return (20%) and gap risk (10%). Higher means riskier.")

    # 1) Risk score
    with st.spinner("Getting the forecast (the first request can take up to a minute while the "
                    "service wakes up)..."):
        pred, error = _safe(get_prediction, token)

    if error:
        st.error(f"Risk forecast unavailable. {error}")
        _retry_button("retry_predict")
    else:
        score = pred["prediction"]
        label, colour = risk_band(score)
        current = pred.get("current_mri")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Forecast risk on {pred['prediction_date']}", f"{score:.1f}",
                  delta=None if current is None else f"{score - current:+.1f} vs today",
                  delta_color="inverse")
        c2.metric(f"Today's risk ({pred['feature_date']})", "n/a" if current is None else f"{current:.1f}")
        c3.markdown(f"**Risk level**<br><span style='font-size:1.6rem;color:{colour}'>"
                    f"&#9679; {label}</span>", unsafe_allow_html=True)
        st.altair_chart(risk_gauge(score), width="stretch")

        if current is not None:
            direction = "rise" if score > current + 1 else "fall" if score < current - 1 else "stay about the same"
            st.write(f"Risk for **{token}** is expected to **{direction}** over the next 7 days "
                     f"({current:.1f} today, {score:.1f} on {pred['prediction_date']}).")

        if pred.get("reduced_accuracy"):
            missing = ", ".join(FEATURE_LABELS.get(f, f) for f in pred["missing_features"])
            st.warning(f"Some inputs were unavailable, so this forecast is less accurate than usual. "
                       f"Missing: {missing}. This usually means a teammate's service is waking up or "
                       f"down; the model is trained to handle it.")

        with st.expander("Inputs used for this forecast"):
            table = pd.DataFrame({
                "Feature": [FEATURE_LABELS.get(f, f) for f in pred["features_used"]],
                "Value": [None if v is None else round(v, 4) for v in pred["features_used"].values()],
                "Source": [pred["feature_sources"].get(f, "") for f in pred["features_used"]],
            })
            st.dataframe(table, hide_index=True, width="stretch")

    # 2) Recent price trend (Student B's own features)
    st.subheader("Recent price trend")
    feats, error = _safe(get_features, token, 90)
    if error:
        st.warning(f"Price history unavailable. {error}")
    else:
        hist = pd.DataFrame(feats["history"])
        hist["date"] = pd.to_datetime(hist["date"])
        lines = hist.melt(id_vars="date", value_vars=["close", "MA5", "MA20"], var_name="series",
                          value_name="price")
        lines["series"] = lines["series"].map({"close": "Close", "MA5": "5-day average",
                                               "MA20": "20-day average"})
        price_chart = alt.Chart(lines).mark_line().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("price:Q", title="USD", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("date:T"), "series:N", alt.Tooltip("price:Q", format=",.4f")],
        ).properties(height=260)
        st.altair_chart(price_chart, width="stretch")

        trend = hist[["date", "TrendStrength"]].assign(direction=lambda d: d["TrendStrength"].ge(0).map(
            {True: "Up", False: "Down"}))
        trend_chart = alt.Chart(trend).mark_bar().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("TrendStrength:Q", title="Trend strength", axis=alt.Axis(format="%")),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["Up", "Down"],
                                                           range=["#2e7d32", "#c62828"]), legend=None),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("TrendStrength:Q", format=".2%")],
        ).properties(height=180)
        st.altair_chart(trend_chart, width="stretch")
        st.caption("Trend strength = 5-day average / 20-day average - 1. Sharp falls and sharp rallies "
                   "are both followed by higher risk, which makes it the model's most important input.")

    # 3) Feature importance and model information
    st.subheader("What drives the forecast")
    meta, error = _safe(get_metadata)
    if error:
        st.warning(f"Model information unavailable. {error}")
    else:
        imp = (pd.DataFrame({"feature": list(meta["feature_importance"]),
                             "importance": list(meta["feature_importance"].values())})
               .assign(label=lambda d: d["feature"].map(lambda f: FEATURE_LABELS.get(f, f)))
               .sort_values("importance", ascending=False))
        imp_chart = alt.Chart(imp).mark_bar(color="#5c6bc0").encode(
            x=alt.X("importance:Q", title="Share of model gain", axis=alt.Axis(format="%")),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
            tooltip=["label:N", alt.Tooltip("importance:Q", format=".1%")],
        ).properties(height=230)
        st.altair_chart(imp_chart, width="stretch")

        m = meta["test_metrics"]
        with st.expander("About the model"):
            st.markdown(
                f"- **Algorithm:** {meta['algorithm']}, forecast horizon {meta['forecast_horizon']} days\n"
                f"- **Trained on:** {meta['training_data']['source']}, "
                f"{meta['training_data']['from']} to {meta['training_data']['to']} (all 8 coins)\n"
                f"- **Test accuracy:** average error {m['MAE']:.2f} points "
                f"(vs {meta['baseline']['test_mae']:.2f} for assuming next week's risk equals today's); "
                f"direction of change correct {m['Direction accuracy']:.0%} of the time\n"
                f"- **Missing inputs:** {meta['missing_value_handling']}")

    st.info("Limitations: forecasts use past prices only and cannot anticipate sudden shocks; the model "
            "tends to under-estimate risk in very turbulent weeks, so treat a high current risk as a "
            "warning on its own. This is decision support, not financial advice.")


# ---------- Standalone preview: streamlit run students/student_b.py ----------
if __name__ == "__main__":
    st.set_page_config(page_title="Market Risk Index", layout="wide")
    selected = st.sidebar.selectbox("Cryptocurrency", TOKENS)
    st.sidebar.caption(f"API: {API_URL}")
    with st.sidebar:
        st.divider()
        st.caption("Home-page visual preview")
        home_chart(selected)
    render(selected)

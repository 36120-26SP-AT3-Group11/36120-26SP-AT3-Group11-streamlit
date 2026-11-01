"""Student D - Trading Opportunity Rating (TOR) page.

Shows the outputs required by the brief - Recommendation, Prediction Confidence Score and
Feature Importance - plus Student D's volume features. All numbers come from the Student D
FastAPI service over HTTP; no model runs in Streamlit. Each section degrades on its own, so if
one call fails the rest of the page still renders. Layout and colours come from students/ui.py.

Use from the main app:
    from students import student_d
    student_d.render(token)       # full page (app/main.py draws the page banner)
    student_d.home_chart(token)   # compact visual for the home page

Run on its own for testing:
    streamlit run students/student_d.py
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

DEFAULT_API_URL = "https://group11-studentd-api.onrender.com"
TOKENS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XMR", "XLM", "TRX"]
CONNECT_TIMEOUT = 8  # seconds
READ_TIMEOUT = 60  # free Render services can take up to a minute to wake up
CACHE_SECONDS = 300
FORECAST_HORIZON = 7

TITLE = "Trading Opportunity Rating"
ACCENT = ui.ACCENTS["D"]
DESCRIPTION = (f"Should you buy, hold or sell over the next {FORECAST_HORIZON} days? A LightGBM classifier "
               "rates each coin from Strong Sell to Strong Buy using momentum (Student A), trend "
               "(Student B), volatility (Student C) and trading volume (Student D).")

TOR_LABELS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
COLOUR_BY_RATING = dict(zip(TOR_LABELS, ui.SCALE_5))

FEATURE_LABELS = {
    "Return20": "20-day return (Student A)",
    "MA5_MA20": "MA5 / MA20 trend ratio (Student B)",
    "Volatility20": "20-day volatility (Student C)",
    "RelativeVolume": "Relative volume (Student D)",
    "OpportunityScore": "Opportunity score (derived)",
}
SOURCE_LABELS = {
    "student_a_api": "Student A API",
    "student_b_api": "Student B API",
    "student_c_api": "Student C API",
    "student_d": "Student D (this API)",
    "derived": "Derived from inputs",
    "unavailable": "Unavailable",
}


# ---------- API calls ----------
class ApiUnavailableError(Exception):
    """Raised for any failure to obtain a usable response from the Student D API."""


def api_url() -> str:
    """API base URL from Streamlit secrets, then the environment, then the default."""
    try:
        url = st.secrets.get("STUDENT_D_API_URL")
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        url = None
    return (url or os.getenv("STUDENT_D_API_URL") or DEFAULT_API_URL).rstrip("/")


def _error_message(response: requests.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])
    return None


def _request(method: str, base_url: str, path: str, payload: dict | None = None) -> dict:
    """One HTTP call with a single retry (the first call may wake a sleeping instance)."""
    last = "unknown error"
    for _attempt in range(2):
        try:
            response = requests.request(
                method, f"{base_url}{path}", json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
        except requests.Timeout:
            last = "the request timed out (the service may be starting up)"
            continue
        except requests.ConnectionError:
            last = "the service could not be reached"
            continue
        except requests.RequestException as exc:
            raise ApiUnavailableError(f"request failed ({type(exc).__name__})") from exc
        if response.status_code >= 500:
            last = _error_message(response) or f"server error (HTTP {response.status_code})"
            continue
        if response.status_code != 200:
            raise ApiUnavailableError(
                _error_message(response) or f"unexpected response (HTTP {response.status_code})"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ApiUnavailableError("the service returned invalid JSON") from exc
    raise ApiUnavailableError(last)


# Only successful responses are cached (exceptions are never cached by st.cache_data).
@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def fetch_prediction(base_url: str, token: str) -> dict:
    return _request("POST", base_url, "/predict", {"token": token, "forecast_horizon": FORECAST_HORIZON})


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def fetch_features(base_url: str, token: str, days: int) -> dict:
    return _request("POST", base_url, "/features", {"token": token, "history_days": days})


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_metadata(base_url: str) -> dict:
    return _request("GET", base_url, "/model-metadata")


CACHES = (fetch_prediction, fetch_features, fetch_metadata)


# ---------- Charts ----------
def _probability_chart(probabilities: dict, height) -> alt.Chart:
    df = pd.DataFrame({"Rating": TOR_LABELS, "Probability": [probabilities.get(r, 0) for r in TOR_LABELS]})
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=6).encode(
        y=alt.Y("Rating:N", sort=TOR_LABELS, title=None),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
        color=alt.Color("Rating:N", scale=alt.Scale(domain=TOR_LABELS, range=ui.SCALE_5), legend=None),
        tooltip=["Rating", alt.Tooltip("Probability:Q", format=".1%")],
    )
    text = bars.mark_text(align="left", dx=6, color=ui.TEXT).encode(
        text=alt.Text("Probability:Q", format=".0%"))
    return (bars + text).properties(height=height)


# ---------- Page sections ----------
def _show_prediction(token: str, prediction: dict) -> None:
    rating = prediction["prediction"]
    confidence = float(prediction["confidence"])
    current = prediction.get("current_opportunity") or {}

    with st.container(border=True):
        info_col, chart_col = st.columns([1, 1], vertical_alignment="center")
        with info_col:
            ui.level_badge(rating, COLOUR_BY_RATING.get(rating, ui.MUTED))
            m1, m2 = st.columns(2)
            m1.metric(f"Rating for {prediction['prediction_date']}", rating)
            m2.metric("Prediction confidence", f"{confidence:.0%}")
            st.progress(min(max(confidence, 0.0), 1.0), text="Probability of the predicted rating")
            if current.get("rating"):
                st.write(f"Today's data ({prediction['as_of_date']}) implies **{current['rating']}** "
                         f"(opportunity score {current['opportunity_score']:.3f}); the model expects "
                         f"**{rating}** for {token} in {FORECAST_HORIZON} days.")
            else:
                st.caption("Today's rating needs every input, so it is hidden while another service is down.")
        with chart_col:
            st.altair_chart(_probability_chart(prediction["class_probabilities"], alt.Step(34)),
                            width="stretch")
            st.caption(f"Probability of each rating in {FORECAST_HORIZON} days.")

    ui.missing_inputs_warning(prediction.get("missing_features") or [], FEATURE_LABELS)


def _show_volume_history(features: dict) -> None:
    ui.section("Trading volume (Student D's features)",
               "Relative volume compares today's volume with its 20-day average. Above 1 means busier "
               "than normal; it is capped at 2 in the opportunity score.", ACCENT)
    history = pd.DataFrame(features["history"])
    history["date"] = pd.to_datetime(history["date"])

    volume = alt.Chart(history).mark_bar(color=ui.NEUTRAL_BAR, opacity=0.5).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("volume:Q", title="Daily volume"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("volume:Q", format=",.0f")],
    )
    sma = alt.Chart(history).mark_line(color=ACCENT, strokeWidth=2).encode(
        x="date:T", y="VolumeSMA20:Q", tooltip=[alt.Tooltip("VolumeSMA20:Q", format=",.0f")]
    )
    relative = alt.Chart(history).mark_line(color=ui.ACCENT).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("RelativeVolume:Q", title="Relative volume"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("RelativeVolume:Q", format=".2f")],
    )
    guides = alt.Chart(pd.DataFrame({"y": [1.0, 2.0]})).mark_rule(
        strokeDash=[4, 4], color=ui.MUTED).encode(y="y:Q")

    left, right = st.columns(2)
    left.altair_chart((volume + sma).properties(height=260, title="Volume and its 20-day average"),
                      width="stretch")
    right.altair_chart((relative + guides).properties(height=260, title="Relative volume"),
                       width="stretch")
    st.caption(f"Data source: {features.get('data_source', 'n/a')}, last complete day "
               f"{features.get('as_of_date', 'n/a')}.")


def _show_model(metadata: dict) -> None:
    ui.section("What drives the forecast",
               "The short-term trend (MA5 / MA20) carries most of the model's decisions, followed by "
               "the 20-day return.", ACCENT)
    importance = metadata.get("feature_importance", {})
    if importance:
        ui.importance_chart(importance, FEATURE_LABELS, accent=ACCENT)
    else:
        st.info("The API returned no feature importance values.")

    evaluation = metadata.get("evaluation", {})
    test, persistence = evaluation.get("test"), evaluation.get("baseline_persistence")
    if test and persistence:
        k1, k2, k3 = st.columns(3)
        k1.metric("Macro-F1 (test)", f"{test['macro_f1']:.3f}",
                  delta=f"{test['macro_f1'] - persistence['macro_f1']:+.3f} vs baseline")
        k2.metric("Accuracy (test)", f"{test['accuracy']:.0%}",
                  delta=f"{(test['accuracy'] - persistence['accuracy']) * 100:+.0f} pts vs baseline")
        k3.metric("Forecast horizon", f"{metadata.get('forecast_horizon', FORECAST_HORIZON)} days")

    with st.expander("About the model", icon=":material/model_training:"):
        training = metadata.get("training_data", {})
        st.markdown(
            f"- **Target:** {metadata.get('target', 'n/a')}\n"
            f"- **Algorithm:** {metadata.get('algorithm', 'n/a')}, tuned with Optuna\n"
            f"- **Trained on:** {training.get('n_rows', 'n/a')} daily rows, "
            f"{training.get('start_date', '?')} to {training.get('end_date', '?')}, from "
            f"{training.get('source', 'n/a')}\n"
            "- **Baseline:** assuming the rating in 7 days equals today's rating"
        )
        rules = metadata.get("classification_rules", {})
        if rules:
            st.markdown("**Rating rules** (OS = 0.40·M + 0.30·T + 0.20·V − 0.10·R)")
            st.dataframe(pd.DataFrame({"Rating": list(rules), "Rule": list(rules.values())}),
                         hide_index=True, width="stretch")
        rows = [
            {"Scenario": name.replace("_", " "),
             **{k: scores.get(k) for k in ("accuracy", "balanced_accuracy", "macro_f1")}}
            for name, scores in evaluation.items()
        ]
        if rows:
            st.markdown("**Held-out test performance**, including scenarios where a teammate's service is down")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---------- Home-page visual ----------
def home_chart(token: str) -> None:
    with st.spinner("Getting the trading opportunity forecast..."):
        try:
            prediction = fetch_prediction(api_url(), token)
        except ApiUnavailableError as exc:
            st.warning(f"Trading opportunity forecast unavailable: {exc}.")
            return
    st.metric(f"{token} rating on {prediction['prediction_date']}", prediction["prediction"],
              f"{float(prediction['confidence']):.0%} confidence", delta_color="off")
    st.altair_chart(_probability_chart(prediction["class_probabilities"], alt.Step(20)), width="stretch")
    if prediction.get("missing_features"):
        st.caption("Some inputs were unavailable, so confidence is lower.")


# ---------- Full page ----------
def render(token: str) -> None:
    base_url = api_url()
    st.caption(DESCRIPTION)

    with st.spinner("Getting the forecast (the first request can take up to a minute while the "
                    "service wakes up)..."):
        results, errors = {}, {}
        for name, fetch in (
            ("prediction", lambda: fetch_prediction(base_url, token)),
            ("metadata", lambda: fetch_metadata(base_url)),
            ("features", lambda: fetch_features(base_url, token, 180)),
        ):
            try:
                results[name] = fetch()
            except ApiUnavailableError as exc:
                errors[name] = str(exc)

    if not results:
        ui.api_error(f"The Trading Opportunity API is unavailable right now: {errors['prediction']}. "
                     "The other pages are not affected.", "retry_d", CACHES)
        return

    prediction, metadata, features = (results.get(k) for k in ("prediction", "metadata", "features"))
    if prediction:
        _show_prediction(token, prediction)
        ui.inputs_table(
            prediction.get("features_used", {}),
            {f: SOURCE_LABELS.get(s, s) for f, s in prediction.get("feature_sources", {}).items()},
            FEATURE_LABELS,
        )
    else:
        ui.api_error(f"The recommendation is unavailable right now: {errors['prediction']}.",
                     "retry_d", CACHES)

    if features:
        _show_volume_history(features)
    else:
        st.warning(f"Volume history is unavailable: {errors['features']}.")

    if metadata:
        _show_model(metadata)
    else:
        st.warning(f"Model information is unavailable: {errors['metadata']}.")

    ui.limitations("Ratings use past prices and volume only and cannot anticipate news or sudden shocks. "
                   "The model beats the 'rating stays the same' baseline only narrowly and rarely "
                   "predicts Strong Sell. Read a Buy alongside the Market Risk page.")


# ---------- Standalone preview: streamlit run students/student_d.py ----------
if __name__ == "__main__":
    st.set_page_config(page_title=TITLE, layout="wide")
    ui.apply_theme()
    selected = st.sidebar.selectbox("Cryptocurrency", TOKENS)
    ui.page_banner("Student D · LightGBM classification · 7 days ahead", f"{TITLE} · {selected}", "", ACCENT)
    render(selected)

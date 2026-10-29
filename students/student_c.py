"""Student C tab - Trend Category (TC).

Shows the three outputs required by the brief - Current Trend, Prediction Confidence Score and
Feature Importance - using only HTTP calls to Student C's FastAPI service (no model inference
happens in Streamlit). If the API is slow, asleep or down, the page explains what happened and
the rest of the application keeps working.

Usage from ``app/main.py``::

    from students import student_c
    student_c.render(token)       # full page
    student_c.home_chart(token)   # one small visual for the home page
"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "https://group11-studentc-api.onrender.com"
CONNECT_TIMEOUT = 8  # seconds
READ_TIMEOUT = 60  # free Render instances need ~30-60 s to wake up
CACHE_SECONDS = 300

CLASS_ORDER = ["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]
CLASS_STYLE = {
    "Strong Downtrend": ("🔴", "#c0392b"),
    "Downtrend": ("🟠", "#e67e22"),
    "Sideways": ("⚪", "#95a5a6"),
    "Uptrend": ("🟢", "#27ae60"),
    "Strong Uptrend": ("🟩", "#16a085"),
}


class ApiUnavailableError(Exception):
    """Raised for any failure to obtain a usable response from the Student C API."""


def api_url() -> str:
    """API base URL from Streamlit secrets, then the environment, then the default."""
    try:
        url = st.secrets.get("STUDENT_C_API_URL")
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        url = None
    return (url or os.getenv("STUDENT_C_API_URL") or DEFAULT_API_URL).rstrip("/")


def _request(
    method: str, base_url: str, path: str, payload: dict | None = None
) -> dict:
    """One HTTP call with a single retry (the first call may wake a sleeping instance)."""
    last = "unknown error"
    for _attempt in range(2):
        try:
            response = requests.request(
                method,
                f"{base_url}{path}",
                json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
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
            last = (
                _error_message(response)
                or f"server error (HTTP {response.status_code})"
            )
            continue
        if response.status_code != 200:
            raise ApiUnavailableError(
                _error_message(response)
                or f"unexpected response (HTTP {response.status_code})"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ApiUnavailableError("the service returned invalid JSON") from exc
    raise ApiUnavailableError(last)


def _error_message(response: requests.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict) and body.get("error"):
        return str(body["error"])
    return None


# Only successful responses are cached (exceptions are never cached by st.cache_data).
@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def fetch_prediction(base_url: str, token: str) -> dict:
    return _request(
        "POST", base_url, "/predict", {"token": token, "forecast_horizon": 7}
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_metadata(base_url: str) -> dict:
    return _request("GET", base_url, "/model-metadata")


def _label(name: str) -> str:
    return f"{CLASS_STYLE.get(name, ('', ''))[0]} {name}".strip()


def _color_scale() -> alt.Scale:
    return alt.Scale(domain=CLASS_ORDER, range=[CLASS_STYLE[c][1] for c in CLASS_ORDER])


def _show_prediction(token: str, result: dict) -> None:
    current = result.get("current_trend")
    st.caption(
        "Current trend uses today's MA5, MA20 and 20-day return; the prediction is the expected "
        f"category in {result.get('forecast_horizon', 7)} days "
        f"(by {result.get('prediction_date', 'n/a')})."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Current trend", _label(current) if current else "Unavailable")
    col2.metric("Predicted trend (+7 days)", _label(result["prediction"]))
    confidence = float(result["confidence"])
    col3.metric("Prediction confidence", f"{confidence:.0%}")
    st.progress(
        min(max(confidence, 0.0), 1.0), text="Confidence in the predicted class"
    )

    if result.get("degraded"):
        status = result.get("upstream_status", {})
        missing = ", ".join(result.get("missing_features", [])) or "some inputs"
        st.warning(
            f"Degraded prediction: {missing} could not be retrieved from the other services "
            f"(Student A: {status.get('student_a', '?')}; Student B: {status.get('student_b', '?')}). "
            "The model handled the gap with missing-value logic, so treat the confidence with care."
        )

    probs = pd.DataFrame(
        {
            "Class": CLASS_ORDER,
            "Probability": [
                result["class_probabilities"].get(c, 0) for c in CLASS_ORDER
            ],
        }
    )
    chart = (
        alt.Chart(probs)
        .mark_bar()
        .encode(
            y=alt.Y("Class:N", sort=CLASS_ORDER, title=None),
            x=alt.X(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("Class:N", scale=_color_scale(), legend=None),
            tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(
            height=alt.Step(34),
            title=f"{token}: probability of each trend category in 7 days",
        )
    )
    st.altair_chart(chart, width="stretch")

    with st.expander("Model inputs used for this prediction"):
        inputs = result.get("inputs", {})
        st.dataframe(
            pd.DataFrame(
                {
                    "Input": list(inputs),
                    "Value": [
                        "missing" if v is None else f"{v:.6g}" for v in inputs.values()
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )


def _show_metadata(meta: dict) -> None:
    st.subheader("Feature importance")
    importance = meta.get("feature_importance", {})
    if importance:
        df = pd.DataFrame(
            {"Feature": list(importance), "Importance": list(importance.values())}
        )
        chart = (
            alt.Chart(df)
            .mark_bar(color="#6c5ce7")
            .encode(
                y=alt.Y("Feature:N", sort="-x", title=None),
                x=alt.X(
                    "Importance:Q",
                    axis=alt.Axis(format="%"),
                    title="Share of total gain",
                ),
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")],
            )
            .properties(height=alt.Step(40))
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info("The API returned no feature importance values.")

    with st.expander("About the model"):
        st.markdown(
            f"**Target:** {meta.get('target', 'n/a')}  \n"
            f"**Algorithm:** {meta.get('algorithm', 'n/a')}  \n"
            f"**Forecast horizon:** {meta.get('forecast_horizon', 'n/a')} days  \n"
            f"**Features:** {', '.join(meta.get('features', [])) or 'n/a'}"
        )
        thresholds = meta.get("class_thresholds")
        if thresholds:
            st.table(
                pd.DataFrame(
                    {"Category": list(thresholds), "Rule": list(thresholds.values())}
                )
            )
        scores = meta.get("test_metrics", {})
        if scores.get("model"):
            st.markdown(
                f"**Held-out test performance:** accuracy {scores['model'].get('accuracy', 'n/a')}, "
                f"macro-F1 {scores['model'].get('macro_f1', 'n/a')}"
            )


def render(token: str) -> None:
    """Render the Trend Category tab for ``token`` (e.g. ``"BTC"``)."""
    base_url = api_url()
    st.header("Trend Category (TC)")
    st.caption(
        f"Where is {token} heading over the next 7 days? XGBoost classifier served by Student C's API."
    )

    if st.button("Refresh", key=f"tc_refresh_{token}"):
        fetch_prediction.clear()

    with st.spinner(
        "Contacting the Trend Category API (it may take up to a minute to wake up)..."
    ):
        prediction_error = metadata_error = None
        result = meta = None
        try:
            result = fetch_prediction(base_url, token)
        except ApiUnavailableError as exc:
            prediction_error = str(exc)
        try:
            meta = fetch_metadata(base_url)
        except ApiUnavailableError as exc:
            metadata_error = str(exc)

    if result:
        _show_prediction(token, result)
    else:
        st.error(
            f"The Trend Category prediction is unavailable right now: {prediction_error}. "
            "The other tabs are not affected. Please try again in a moment."
        )
    if meta:
        _show_metadata(meta)
    elif metadata_error:
        st.warning(f"Model details are unavailable: {metadata_error}.")


def home_chart(token: str) -> None:
    """One small visual for the home page: probability of each trend category in 7 days."""
    base_url = api_url()
    st.subheader("Trend Category (7 days)")
    try:
        result = fetch_prediction(base_url, token)
    except ApiUnavailableError:
        st.info("Trend Category is unavailable right now.")
        return
    label = _label(result["prediction"])
    st.metric("Predicted trend", label, f"{float(result['confidence']):.0%} confidence", delta_color="off")
    probs = pd.DataFrame(
        {
            "Class": CLASS_ORDER,
            "Probability": [result["class_probabilities"].get(c, 0) for c in CLASS_ORDER],
        }
    )
    chart = (
        alt.Chart(probs)
        .mark_bar()
        .encode(
            y=alt.Y("Class:N", sort=CLASS_ORDER, title=None),
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Class:N", scale=_color_scale(), legend=None),
            tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=alt.Step(22))
    )
    st.altair_chart(chart, width="stretch")
    if result.get("degraded"):
        st.caption("Some inputs were unavailable, so confidence is lower.")
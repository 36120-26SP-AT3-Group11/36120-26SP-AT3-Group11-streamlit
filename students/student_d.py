"""Student D tab - Trading Opportunity Rating (TOR).

Shows the outputs required by the brief - Recommendation, Prediction Confidence Score and
Feature Importance - plus Student D's volume features, using only HTTP calls to Student D's
FastAPI service (no model inference happens in Streamlit). Every section degrades on its own:
if one call fails the rest of the page still renders, and if the API is down the page explains
what happened while the rest of the application keeps working.

Usage from ``app/main.py``::

    from students import student_d
    student_d.render(token)       # full page
    student_d.home_chart(token)   # one small visual for the home page
"""

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "https://at3-student-d-tor-api.onrender.com"
CONNECT_TIMEOUT = 8  # seconds
READ_TIMEOUT = 60  # free Render instances need ~30-60 s to wake up
CACHE_SECONDS = 300
FORECAST_HORIZON = 7

TOR_LABELS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
RATING_COLOURS = ["#c0392b", "#e67e22", "#95a5a6", "#2ecc71", "#1e8449"]
COLOUR_BY_RATING = dict(zip(TOR_LABELS, RATING_COLOURS))
SOURCE_LABELS = {
    "student_a_api": "Student A API",
    "student_b_api": "Student B API",
    "student_c_api": "Student C API",
    "student_d": "Student D (this API)",
    "derived": "Derived from inputs",
    "unavailable": "Unavailable",
}


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


def _rating_badge(rating: str | None, caption: str) -> None:
    colour = COLOUR_BY_RATING.get(rating, "#7f8c8d")
    st.markdown(
        f"<div style='font-size:0.85rem;opacity:0.7'>{caption}</div>"
        f"<div style='font-size:2rem;font-weight:700;color:{colour}'>{rating or 'Unavailable'}</div>",
        unsafe_allow_html=True,
    )


def _probability_chart(probabilities: dict, height) -> alt.Chart:
    df = pd.DataFrame({"Rating": TOR_LABELS, "Probability": [probabilities.get(r, 0) for r in TOR_LABELS]})
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("Rating:N", sort=TOR_LABELS, title=None),
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Rating:N", scale=alt.Scale(domain=TOR_LABELS, range=RATING_COLOURS), legend=None),
            tooltip=["Rating", alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=height)
    )


def _show_prediction(token: str, prediction: dict) -> None:
    st.subheader("7-day recommendation")
    col_rec, col_conf, col_now = st.columns(3)
    with col_rec:
        _rating_badge(prediction["prediction"], f"Predicted rating for {prediction['prediction_date']}")
    with col_conf:
        confidence = float(prediction["confidence"])
        st.metric("Prediction confidence", f"{confidence:.0%}")
        st.progress(min(max(confidence, 0.0), 1.0), text="Probability of the predicted rating")
    with col_now:
        current = prediction.get("current_opportunity") or {}
        _rating_badge(current.get("rating"), f"Rating implied by data on {prediction['as_of_date']}")
        if current.get("opportunity_score") is not None:
            st.caption(f"Opportunity score: {current['opportunity_score']:.3f}")
        elif not current.get("rating"):
            st.caption("Needs every input; unavailable while another service is down.")

    st.altair_chart(
        _probability_chart(prediction["class_probabilities"], alt.Step(34)).properties(
            title=f"{token}: probability of each rating in {FORECAST_HORIZON} days"
        ),
        width="stretch",
    )

    missing = prediction.get("missing_features") or []
    if missing:
        st.warning(
            f"Degraded prediction: {', '.join(missing)} unavailable because another student's service "
            "did not respond. The LightGBM model handled the gap with its missing-value logic, so treat "
            "the confidence with care."
        )


def _show_inputs(prediction: dict, metadata: dict | None) -> None:
    st.subheader("Model inputs used")
    descriptions = (metadata or {}).get("feature_descriptions", {})
    sources = prediction.get("feature_sources", {})
    rows = [
        {
            "Feature": name,
            "Value": value,
            "Source": SOURCE_LABELS.get(sources.get(name), "Unknown"),
            "Description": descriptions.get(name, ""),
        }
        for name, value in prediction.get("features_used", {}).items()
    ]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"Value": st.column_config.NumberColumn(format="%.4f")},
    )


def _show_feature_importance(metadata: dict) -> None:
    st.subheader("Feature importance")
    importance = metadata.get("feature_importance", {})
    if not importance:
        st.info("The API returned no feature importance values.")
        return
    df = pd.DataFrame({"Feature": list(importance), "Importance": list(importance.values())})
    df["Description"] = df["Feature"].map(metadata.get("feature_descriptions", {}))
    chart = (
        alt.Chart(df)
        .mark_bar(color="#f59e0b")
        .encode(
            y=alt.Y("Feature:N", sort="-x", title=None),
            x=alt.X("Importance:Q", axis=alt.Axis(format="%"), title="Share of total gain"),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%"), "Description"],
        )
        .properties(height=alt.Step(40))
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"{metadata.get('algorithm', 'LightGBM')}, gain-based importance: higher means the feature "
        "contributed more to the model's splits."
    )


def _show_volume_history(features: dict) -> None:
    st.subheader("Volume features (owned by Student D)")
    history = pd.DataFrame(features["history"])
    history["date"] = pd.to_datetime(history["date"])

    volume = alt.Chart(history).mark_bar(color="#94a3b8", opacity=0.6).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("volume:Q", title="Daily volume"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("volume:Q", format=",.0f")],
    )
    sma = alt.Chart(history).mark_line(color="#f59e0b").encode(
        x="date:T", y="VolumeSMA20:Q", tooltip=[alt.Tooltip("VolumeSMA20:Q", format=",.0f")]
    )
    relative = alt.Chart(history).mark_line(color="#8b5cf6").encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("RelativeVolume:Q", title="Relative volume"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("RelativeVolume:Q", format=".2f")],
    )
    guides = alt.Chart(pd.DataFrame({"y": [1.0, 2.0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")

    left, right = st.columns(2)
    left.altair_chart((volume + sma).properties(height=260, title="Volume and its 20-day average"),
                      width="stretch")
    right.altair_chart(
        (relative + guides).properties(height=260, title="Relative volume (capped at 2 in the score)"),
        width="stretch",
    )
    st.caption(f"Data source: {features.get('data_source', 'n/a')}, last complete day "
               f"{features.get('as_of_date', 'n/a')}.")


def _show_model_details(metadata: dict) -> None:
    with st.expander("About the model"):
        st.markdown(
            f"**Target:** {metadata.get('target', 'n/a')}  \n"
            f"**Algorithm:** {metadata.get('algorithm', 'n/a')} (tuned with Optuna)  \n"
            f"**Forecast horizon:** {metadata.get('forecast_horizon', 'n/a')} days  \n"
            f"**Features:** {', '.join(metadata.get('features', [])) or 'n/a'}"
        )
        rules = metadata.get("classification_rules", {})
        if rules:
            st.markdown("**Rating rules** (OS = 0.40·M + 0.30·T + 0.20·V − 0.10·R)")
            st.table(pd.DataFrame({"Rating": list(rules), "Rule": list(rules.values())}))
        evaluation = metadata.get("evaluation", {})
        rows = [
            {"Scenario": name.replace("_", " "),
             **{k: scores.get(k) for k in ("accuracy", "balanced_accuracy", "macro_f1")}}
            for name, scores in evaluation.items()
        ]
        if rows:
            st.markdown("**Held-out test performance** (including upstream-outage scenarios)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        training = metadata.get("training_data", {})
        st.caption(
            f"Trained on {training.get('n_rows', 'n/a')} rows ({training.get('start_date', '?')} to "
            f"{training.get('end_date', '?')}) from {training.get('source', 'n/a')}. "
            "For decision support only, not financial advice."
        )


def render(token: str) -> None:
    """Render the Trading Opportunity Rating tab for ``token`` (e.g. ``"BTC"``)."""
    base_url = api_url()
    st.header("Trading Opportunity Rating (TOR)")
    st.caption(
        f"Should you buy, hold or sell {token} over the next {FORECAST_HORIZON} days? LightGBM classifier "
        "served by Student D's API, using momentum, trend, volume and volatility."
    )

    if st.button("Refresh", key=f"tor_refresh_{token}"):
        fetch_prediction.clear()
        fetch_features.clear()

    with st.spinner("Contacting the Trading Opportunity API (it may take up to a minute to wake up)..."):
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
        st.error(
            f"The Trading Opportunity API is unavailable right now: {errors['prediction']}. "
            "The other tabs are not affected. Please try again in a moment."
        )
        return

    prediction, metadata, features = (results.get(k) for k in ("prediction", "metadata", "features"))
    if prediction:
        _show_prediction(token, prediction)
    else:
        st.error(f"The recommendation is unavailable right now: {errors['prediction']}.")

    left, right = st.columns(2)
    with left:
        if metadata:
            _show_feature_importance(metadata)
        else:
            st.warning(f"Feature importance is unavailable: {errors['metadata']}.")
    with right:
        if prediction:
            _show_inputs(prediction, metadata)

    if features:
        _show_volume_history(features)
    else:
        st.warning(f"Volume history is unavailable: {errors['features']}.")

    if metadata:
        _show_model_details(metadata)


def home_chart(token: str) -> None:
    """One small visual for the home page: the 7-day recommendation and its class probabilities."""
    base_url = api_url()
    st.subheader("Trading Opportunity (7 days)")
    try:
        prediction = fetch_prediction(base_url, token)
    except ApiUnavailableError:
        st.info("Trading Opportunity Rating is unavailable right now.")
        return
    st.metric("Recommendation", prediction["prediction"],
              f"{float(prediction['confidence']):.0%} confidence", delta_color="off")
    st.altair_chart(_probability_chart(prediction["class_probabilities"], alt.Step(22)), width="stretch")
    if prediction.get("missing_features"):
        st.caption("Some inputs were unavailable, so confidence is lower.")

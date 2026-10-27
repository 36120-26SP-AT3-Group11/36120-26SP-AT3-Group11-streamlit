"""Crypto Financial Intelligence Platform - Group 11 (36120 AT3).

Entry point:  streamlit run app/main.py

Pages: Home (coin selection + one visual per student) and one page per student.
Each student's module in students/ provides two functions:

    render(token)      -> the student's full page
    home_chart(token)  -> one compact visual for the home page

A module that does not define them yet shows a placeholder, and an error inside one student's
code is contained to that section, so the app keeps working if any one API or page fails.
"""

import importlib
import sys
from pathlib import Path

import streamlit as st

# Make the repository root importable (Streamlit puts only app/ on the path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOKENS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XMR", "XLM", "TRX"]
TOKEN_NAMES = {"BTC": "Bitcoin", "ETH": "Ethereum", "XRP": "Ripple", "SOL": "Solana",
               "DOGE": "Dogecoin", "XMR": "Monero", "XLM": "Stellar", "TRX": "Tron"}

STUDENTS = [
    # (module, page title, short title, icon, url path)
    ("student_a", "Asset Performance Index", "Asset Performance (API)", ":material/trending_up:", "asset-performance"),
    ("student_b", "Market Risk Index", "Market Risk (MRI)", ":material/warning:", "market-risk"),
    ("student_c", "Trend Category", "Trend Category (TC)", ":material/insights:", "trend-category"),
    ("student_d", "Trading Opportunity Rating", "Trading Opportunity (TOR)", ":material/swap_vert:", "trading-opportunity"),
]

st.set_page_config(page_title="Crypto Financial Intelligence Platform", page_icon=":material/monitoring:",
                   layout="wide")


def load_module(name: str):
    """Import students/<name>.py; None if it fails to import (e.g. a syntax error)."""
    try:
        return importlib.import_module(f"students.{name}")
    except Exception as exc:
        st.session_state.setdefault("import_errors", {})[name] = f"{type(exc).__name__}: {exc}"
        return None


MODULES = {name: load_module(name) for name, *_ in STUDENTS}


def run_section(module_name: str, function: str, token: str, title: str) -> None:
    """Call students/<module>.<function>(token), containing any error or missing function."""
    module = MODULES.get(module_name)
    fn = getattr(module, function, None) if module else None
    if fn is None:
        reason = st.session_state.get("import_errors", {}).get(module_name)
        if reason:
            st.error(f"{title}: this section could not be loaded ({reason}).")
        else:
            st.info(f"{title}: coming soon.")
        return
    try:
        fn(token)
    except Exception as exc:  # one student's failure must not stop the app
        st.error(f"{title}: something went wrong while displaying this section ({type(exc).__name__}).")


def selected_token() -> str:
    return st.session_state.get("token", TOKENS[0])


# ---------- Pages ----------
def home() -> None:
    st.title("Crypto Financial Intelligence Platform")
    st.write("Explore market conditions and 7-day machine-learning forecasts for eight cryptocurrencies. "
             "Pick a coin below, then open the detailed pages in the sidebar.")
    st.selectbox("Cryptocurrency", TOKENS, key="token",
                 format_func=lambda t: f"{t} - {TOKEN_NAMES[t]}")
    token = selected_token()

    st.subheader(f"{TOKEN_NAMES[token]} ({token}) at a glance")
    st.caption("Forecasts come from four independent FastAPI services. If one is unavailable, the rest "
               "of the page keeps working. The first load can take up to a minute while services wake up.")
    cols = st.columns(2) + st.columns(2)
    for col, (module_name, title, *_rest) in zip(cols, STUDENTS):
        with col.container(border=True):
            run_section(module_name, "home_chart", token, title)

    st.divider()
    st.caption("Group 11 · 36120 Advanced Machine Learning Application · For decision support only, "
               "not financial advice.")


def student_page(module_name: str, title: str):
    def page() -> None:
        st.sidebar.selectbox("Cryptocurrency", TOKENS, key="token",
                             format_func=lambda t: f"{t} - {TOKEN_NAMES[t]}")
        run_section(module_name, "render", selected_token(), title)
    page.__name__ = f"page_{module_name}"
    return page


# Keep the coin choice when switching pages (Streamlit drops widget state for widgets not shown)
if "token" in st.session_state:
    st.session_state.token = st.session_state.token

pages = [st.Page(home, title="Home", icon=":material/home:", default=True)] + [
    st.Page(student_page(module_name, title), title=short, icon=icon, url_path=path)
    for module_name, title, short, icon, path in STUDENTS
]
st.navigation(pages).run()

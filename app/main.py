"""Crypto Financial Intelligence Platform - Group 11 (36120 AT3).

Entry point:  streamlit run app/main.py

Pages: Home (coin selection + one visual per student) and one page per student.
Each student's module in students/ provides two functions:

    render(token)      -> the student's full page
    home_chart(token)  -> one compact visual for the home page

A module that does not define them yet shows a placeholder, and an error inside one student's
code is contained to that section, so the app keeps working if any one API or page fails.
Shared styling (colours, fonts, banners, cards) lives in students/ui.py.
"""

import importlib
import sys
from pathlib import Path

import streamlit as st

# Make the repository root importable (Streamlit puts only app/ on the path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from students import ui  # noqa: E402

TOKENS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XMR", "XLM", "TRX"]
TOKEN_NAMES = {"BTC": "Bitcoin", "ETH": "Ethereum", "XRP": "Ripple", "SOL": "Solana",
               "DOGE": "Dogecoin", "XMR": "Monero", "XLM": "Stellar", "TRX": "Tron"}

STUDENTS = [
    # (module, letter, page title, short title, icon, url path, model)
    ("student_a", "A", "Asset Performance Index", "Asset Performance (API)",
     ":material/trending_up:", "asset-performance", "XGBoost regression"),
    ("student_b", "B", "Market Risk Index", "Market Risk (MRI)",
     ":material/warning:", "market-risk", "LightGBM regression"),
    ("student_c", "C", "Trend Category", "Trend Category (TC)",
     ":material/insights:", "trend-category", "XGBoost classification"),
    ("student_d", "D", "Trading Opportunity Rating", "Trading Opportunity (TOR)",
     ":material/swap_vert:", "trading-opportunity", "LightGBM classification"),
]

st.set_page_config(page_title="Crypto Financial Intelligence Platform", page_icon=":material/monitoring:",
                   layout="wide")
ui.apply_theme()


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
            st.error(f"{title}: this section could not be loaded ({reason}).", icon=":material/error:")
        else:
            st.info(f"{title}: coming soon.", icon=":material/hourglass_top:")
        return
    try:
        fn(token)
    except Exception as exc:  # one student's failure must not stop the app
        st.error(f"{title}: something went wrong while displaying this section ({type(exc).__name__}).",
                 icon=":material/error:")


def selected_token() -> str:
    return st.session_state.setdefault("token", TOKENS[0])


def _pick_coin() -> None:
    """Coin buttons write into the shared coin choice (clicking the selected coin again is ignored)."""
    if st.session_state.get("coin_pills"):
        st.session_state.token = st.session_state.coin_pills


def coin_picker() -> str:
    """Same coin buttons at the top of every page; the choice is kept when switching pages."""
    token = selected_token()
    st.session_state.coin_pills = token
    st.pills("Cryptocurrency", TOKENS, key="coin_pills", on_change=_pick_coin,
             label_visibility="collapsed", format_func=lambda t: f"{t} · {TOKEN_NAMES[t]}")
    return token


def explore_other_pages(current: str | None) -> None:
    """Row of coloured links to the other pages, at the bottom of every page."""
    ui.section("Explore other forecasts", accent="#94a3b8")
    others = [s for s in STUDENTS if s[0] != current]
    cols = st.columns(len(others) + 1)
    with cols[0].container(border=True):
        ui.card_header("Home", "All coins", ui.ACCENT)
        st.page_link(HOME, label="Open", icon=":material/home:", width="stretch")
    for col, (module_name, letter, _title, short, icon, *_rest) in zip(cols[1:], others):
        with col.container(border=True):
            ui.card_header(short.split(" (")[0], f"Student {letter}", ui.ACCENTS[letter])
            st.page_link(PAGES[module_name], label="Open", icon=icon, width="stretch")


# ---------- Pages ----------
PAGES: dict = {}


def home() -> None:
    ui.hero("Crypto Financial Intelligence Platform",
            "Explore market conditions and 7-day machine-learning forecasts for eight cryptocurrencies.",
            ["4 ML services", "7-day horizon", "Live Kraken data", "8 coins"])

    token = coin_picker()
    st.markdown(f"## {TOKEN_NAMES[token]} ({token}) at a glance")
    st.caption("Each card comes from a separate API. If one is unavailable, the others keep working.")

    cols = st.columns(2) + st.columns(2)
    for col, (module_name, letter, title, *_rest) in zip(cols, STUDENTS):
        with col.container(border=True):
            ui.card_header(title, f"Student {letter}", ui.ACCENTS[letter])
            run_section(module_name, "home_chart", token, title)
            st.page_link(PAGES[module_name], label="Open full page", icon=":material/arrow_forward:")

    footer()


def footer() -> None:
    ui.footer("Group 11 · 36120 Advanced Machine Learning Application · Four independent FastAPI "
              "services on Render. Free services sleep when idle, so the first load can take up to "
              "a minute. For decision support only, not financial advice.")


def student_page(module_name: str, letter: str, title: str, model: str):
    def page() -> None:
        token = coin_picker()
        ui.page_banner(f"Student {letter} · {model} · 7 days ahead",
                       f"{title} · {TOKEN_NAMES[token]} ({token})", "", ui.ACCENTS[letter])
        run_section(module_name, "render", token, title)
        explore_other_pages(module_name)
        footer()
    page.__name__ = f"page_{module_name}"
    return page


HOME = st.Page(home, title="Home", icon=":material/home:", default=True)
for module_name, letter, title, short, icon, path, model in STUDENTS:
    PAGES[module_name] = st.Page(student_page(module_name, letter, title, model),
                                 title=short, icon=icon, url_path=path)

# Pages are listed in a menu bar across the top, so there is no sidebar
st.navigation([HOME, *PAGES.values()], position="top").run()

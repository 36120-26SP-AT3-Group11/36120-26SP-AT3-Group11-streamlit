import streamlit as st

st.title("Crypto Financial Intelligence Platform")
token = st.selectbox("Choose a cryptocurrency",
                     ["BTC", "ETH", "XRP", "SOL", "DOGE", "XMR", "XLM", "TRX"])
st.write(f"You selected {token}")
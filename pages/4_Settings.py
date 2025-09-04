import streamlit as st
import sys
import os

# Add src to path to import dashboard_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dashboard_utils import get_tracked_tickers, add_ticker, remove_ticker

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Ticker Management")
st.markdown("Manage the list of stock tickers that the application tracks and processes.")

# --- Add Ticker Section ---
st.markdown("### Add a New Ticker")
with st.form(key="add_ticker_form"):
    new_ticker = st.text_input("Enter a stock ticker to add (e.g., GOOG, MSFT):", "").upper()
    submit_button = st.form_submit_button(label="Add Ticker")

    if submit_button:
        if new_ticker:
            add_ticker(new_ticker)
        else:
            st.warning("Please enter a ticker.")

st.markdown("---")

# --- Currently Tracked Tickers Section ---
st.markdown("### Currently Tracked Tickers")
st.info("Removing a ticker will stop the pipeline from fetching new data for it. Existing data will be kept.")

tracked_tickers = get_tracked_tickers()

if not tracked_tickers:
    st.info("No tickers are currently being tracked.")
else:
    # Create a grid-like layout for the tickers
    num_columns = 3
    cols = st.columns(num_columns)
    for i, ticker in enumerate(tracked_tickers):
        with cols[i % num_columns]:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #262730;
                    border-radius: 5px;
                    padding: 10px;
                    text-align: center;
                    margin-bottom: 10px;
                ">
                    <h4>{ticker}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Center the button below the box
            button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
            with button_col2:
                if st.button(f"Remove", key=f"remove_{ticker}"):
                    remove_ticker(ticker)
                    st.experimental_rerun()

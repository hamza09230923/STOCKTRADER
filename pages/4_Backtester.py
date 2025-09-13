import streamlit as st

st.set_page_config(
    page_title="Backtester",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Sentiment-Based Trading Strategy Backtester")
st.markdown("This page allows you to backtest a simple trading strategy based on financial news sentiment.")
st.markdown("---")

# --- Add src to path ---
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dashboard_utils import load_data
from src.backtesting_engine import SentimentStrategy, run_backtest

# --- Load Data ---
data_df, source = load_data()

if data_df is None:
    st.error("Data could not be loaded. Please check the data sources.")
    st.stop()

# --- UI Components ---
st.subheader("Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    # Ticker selection
    available_tickers = sorted(data_df['Ticker'].unique())
    selected_ticker = st.selectbox("Select a stock ticker:", available_tickers)

with col2:
    # Date range selection
    min_date = data_df['Date'].min().date()
    max_date = data_df['Date'].max().date()
    start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)

with col3:
    # Initial cash
    initial_cash = st.number_input("Initial cash", min_value=1000, value=10000, step=1000)

st.subheader("Strategy Parameters")
param_col1, param_col2 = st.columns(2)

with param_col1:
    buy_threshold = st.slider("Buy Sentiment Threshold", 0.0, 1.0, 0.5, 0.05)
with param_col2:
    sell_threshold = st.slider("Sell Sentiment Threshold", -1.0, 0.0, -0.2, 0.05)

# --- Backtest Execution ---
if st.button("Run Backtest"):
    if start_date > end_date:
        st.error("Error: Start date must be before end date.")
    else:
        with st.spinner("Running backtest..."):
            # Prepare data for the backtest
            backtest_data = data_df[
                (data_df['Ticker'] == selected_ticker) &
                (data_df['Date'].dt.date >= start_date) &
                (data_df['Date'].dt.date <= end_date)
            ].copy()

            # The backtesting library expects specific column names.
            # We assume the dataframe has 'Open', 'High', 'Low', 'Close', 'Volume' from Yahoo Finance.
            # We also need to provide the sentiment score. Let's use vader_compound for now.
            if 'vader_compound' in backtest_data.columns:
                backtest_data['sentiment_score'] = backtest_data['vader_compound']
            else:
                st.error("Fatal: 'vader_compound' column not found in the data.")
                st.stop()

            # Set the Date as the index
            backtest_data = backtest_data.set_index('Date')

            # Configure the strategy with user-defined parameters
            class CustomSentimentStrategy(SentimentStrategy):
                buy_sentiment_threshold = buy_threshold
                sell_sentiment_threshold = sell_threshold

            try:
                stats = run_backtest(backtest_data, CustomSentimentStrategy, cash=initial_cash)

                st.subheader("Backtest Results")
                st.write(f"Results for **{selected_ticker}** from **{start_date}** to **{end_date}**")

                # Display key metrics
                st.metric("Return [%]", f"{stats['Return [%]']:.2f}")
                st.metric("Win Rate [%]", f"{stats['Win Rate [%]']:.2f}")
                st.metric("Max. Drawdown [%]", f"{stats['Max. Drawdown [%]']:.2f}")
                st.metric("# Trades", stats['# Trades'])

                st.subheader("Full Statistics")
                st.dataframe(stats)

                # Note on plotting
                st.info("Note: Plot generation is a planned future improvement. The `backtesting.py` library's default plot opens a new browser window, which is not ideal for Streamlit.")

            except Exception as e:
                st.error(f"An error occurred during the backtest: {e}")

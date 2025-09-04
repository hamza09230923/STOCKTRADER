import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from streamlit_autorefresh import st_autorefresh

# Add src to path to import dashboard_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dashboard_utils import load_data, add_ticker_to_config
import subprocess

st.set_page_config(
    page_title="Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Analysis Dashboard")

# --- Data Loading ---
data_df, source = load_data()
if data_df is None:
    st.error("Data could not be loaded. Please ensure the pipeline has run successfully.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")

tickers = sorted(data_df['Ticker'].unique())
if not tickers:
    st.warning("No tickers found in the data. Please run the data pipeline.")
    st.stop()
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)

chart_type = st.sidebar.radio("Select Chart Type", ["Line", "Candlestick"], index=1)

ticker_df = data_df[data_df['Ticker'] == selected_ticker]
if ticker_df.empty:
    st.warning(f"No data found for ticker: {selected_ticker}")
    st.stop()

min_date = ticker_df['Date'].min().date()
max_date = ticker_df['Date'].max().date()

selected_start_date, selected_end_date = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# --- Add New Ticker ---
st.sidebar.markdown("---")
st.sidebar.header("Add New Ticker")
new_ticker = st.sidebar.text_input("Enter a new stock ticker (e.g., GOOGL):")
if st.sidebar.button("Add Ticker"):
    if new_ticker:
        st.sidebar.info(f"Adding {new_ticker.upper()}... this may take a moment.")
        add_ticker_to_config(new_ticker)

        # Run the pipeline for the new ticker
        result = subprocess.run(
            ["python", "run_pipeline.py", "--ticker", new_ticker, "--skip-db"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            st.sidebar.success(f"Ticker {new_ticker.upper()} added successfully!")
            # Clear cache and rerun
            st.cache_data.clear()

            st.rerun()
        else:
            st.sidebar.error(f"Failed to add ticker. Error:\n{result.stderr}")
    else:
        st.sidebar.warning("Please enter a ticker.")

if selected_start_date > selected_end_date:
    st.sidebar.error("Error: Start date must be before end date.")
    st.stop()

filtered_df = ticker_df[
    (ticker_df['Date'].dt.date >= selected_start_date) &
    (ticker_df['Date'].dt.date <= selected_end_date)
]

# --- Main Page ---
st.subheader(f"Displaying data for: **{selected_ticker}**")

# --- Key Metrics ---
if not filtered_df.empty:
    st.markdown("### Key Metrics")
    start_price = filtered_df['Close'].iloc[0]
    end_price = filtered_df['Close'].iloc[-1]
    price_change = end_price - start_price
    price_change_pct = (price_change / start_price) * 100 if start_price != 0 else 0
    avg_finbert_score = filtered_df['finbert_avg_score'].mean()
    total_articles = filtered_df['article_count'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label=f"Closing Price", value=f"${end_price:,.2f}")
    col2.metric(label="Price Change", value=f"${price_change:,.2f}", delta=f"{price_change_pct:.2f}%")
    col3.metric(label="Avg. FinBERT Score", value=f"{avg_finbert_score:.3f}")
    col4.metric(label="Total News Articles", value=f"{int(total_articles)}")
else:
    st.warning("No data available for the selected date range.")

st.markdown("---")

# --- Charts ---
if not filtered_df.empty:
    st.markdown("### Price and Sentiment Analysis")

    # --- Price and Volume Chart ---
    fig_price = go.Figure()

    if chart_type == "Line":
        fig_price.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], name='Close Price', line=dict(color='#00C4FF', width=2)))
    else:
        fig_price.add_trace(go.Candlestick(x=filtered_df['Date'], open=filtered_df['Open'], high=filtered_df['High'], low=filtered_df['Low'], close=filtered_df['Close'], name='Price', increasing_line_color='#00C4FF', decreasing_line_color='#FF6B6B'))

    fig_price.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Volume'], name='Volume', yaxis='y2', marker_color='rgba(0, 196, 255, 0.2)'))
    fig_price.update_layout(title=f'<b>Price and Volume</b>', template="plotly_dark", yaxis=dict(title='Price (USD)'), yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False), legend=dict(x=0.01, y=0.99), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_price, use_container_width=True)

    # --- Sentiment and Article Count Charts ---
    col1, col2 = st.columns(2)
    with col1:
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['vader_avg_score'], name='VADER Score', mode='lines', line_color='#29B6F6'))
        fig_sentiment.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['finbert_avg_score'], name='FinBERT Score', mode='lines', line_color='#FFEE58'))
        fig_sentiment.update_layout(title=f'<b>Daily Average Sentiment Scores</b>', template="plotly_dark", yaxis=dict(title='Avg. Sentiment Score'), legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig_sentiment, use_container_width=True)
    with col2:
        fig_articles = go.Figure()
        fig_articles.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['article_count'], name='Article Count', marker_color='rgba(0, 196, 255, 0.5)'))
        fig_articles.update_layout(title=f'<b>Daily News Article Count</b>', template="plotly_dark", yaxis=dict(title='Number of Articles'))
        st.plotly_chart(fig_articles, use_container_width=True)
else:
    st.warning("No data available to display charts for the selected date range.")

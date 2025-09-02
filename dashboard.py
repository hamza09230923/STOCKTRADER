import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import psycopg2
from streamlit_autorefresh import st_autorefresh

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Stock & Sentiment Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Data Loading & Caching
# ==============================================================================
@st.cache_data
def load_data():
    """
    Loads the processed stock and sentiment data, trying the database first
    and falling back to a local CSV.
    """
    try:
        conn = psycopg2.connect(
            dbname="stock_sentiment", user="postgres", password="password",
            host="localhost", port="5432", connect_timeout=3
        )
        st.info("Database connection successful. Loading data...")
        df = pd.read_sql("SELECT * FROM stock_data ORDER BY \"Date\" ASC", conn)
        df['Date'] = pd.to_datetime(df['Date'])
        conn.close()
        return df
    except psycopg2.OperationalError:
        st.warning("Could not connect to the database. Falling back to local CSV file.")
        try:
            df = pd.read_csv("data/processed_data.csv", parse_dates=['Date'])
            return df
        except FileNotFoundError:
            st.error("Fatal Error: Could not connect to the database AND the fallback file 'data/processed_data.csv' was not found.")
            return None

# Load the base data
data_df = load_data()

# ==============================================================================
# Sidebar Controls & Data Filtering
# ==============================================================================
st.sidebar.header("Dashboard Controls")

if data_df is None:
    st.sidebar.error("Data could not be loaded. Dashboard cannot proceed.")
    st.stop()

# Ticker selection
tickers = sorted(data_df['Ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)

# Chart type selection
chart_type = st.sidebar.radio("Select Chart Type", ["Line", "Candlestick"], index=1)  # Default to Candlestick

# Filter data for the selected ticker to find its date range
ticker_df = data_df[data_df['Ticker'] == selected_ticker]
min_date = ticker_df['Date'].min().date()
max_date = ticker_df['Date'].max().date()

# Date range selection
selected_start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
selected_end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

if selected_start_date > selected_end_date:
    st.sidebar.error("Error: Start date must be before end date.")
    st.stop()

# Create the final filtered DataFrame based on user selections
filtered_df = ticker_df[
    (ticker_df['Date'].dt.date >= selected_start_date) &
    (ticker_df['Date'].dt.date <= selected_end_date)
]

# Auto-refresh control
st.sidebar.header("Live Update")
refresh_interval_minutes = st.sidebar.number_input(
    "Auto-refresh interval (minutes)",
    min_value=0, max_value=60, value=0, step=5,
    help="Set to 0 to disable auto-refresh. Refreshes the page to get new data from the source."
)
if refresh_interval_minutes > 0:
    st_autorefresh(interval=refresh_interval_minutes * 60 * 1000, key="datarefresh")

# About section
st.sidebar.header("About")
st.sidebar.info(
    "This dashboard visualizes stock prices and financial news sentiment. "
    "The data pipeline is built with Python, and the dashboard is powered by Streamlit."
)

# ==============================================================================
# Main Page Content
# ==============================================================================
st.title("Stock Price and News Sentiment Tracker")
st.subheader(f"Displaying data for: {selected_ticker}")

# --- Key Metrics ---
st.header("Key Metrics")
if not filtered_df.empty:
    start_price = filtered_df['Close'].iloc[0]
    end_price = filtered_df['Close'].iloc[-1]
    price_change = end_price - start_price
    price_change_pct = (price_change / start_price) * 100
    avg_finbert_score = filtered_df['finbert_avg_score'].mean()
    total_articles = filtered_df['article_count'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label=f"Closing Price ({selected_end_date.strftime('%Y-%m-%d')})", value=f"${end_price:,.2f}")
    col2.metric(label="Price Change (in range)", value=f"${price_change:,.2f}", delta=f"{price_change_pct:.2f}%")
    col3.metric(label="Avg. FinBERT Score", value=f"{avg_finbert_score:.3f}")
    col4.metric(label="Total News Articles", value=f"{int(total_articles)}")
else:
    st.warning("No data available for the selected date range.")

# --- Charts ---
st.header("Charts")
if not filtered_df.empty:
    # --- Price and Volume Chart ---
    fig_price = go.Figure()

    if chart_type == "Line":
        fig_price.add_trace(go.Scatter(
            x=filtered_df['Date'], y=filtered_df['Close'], name='Close Price',
            line=dict(color='#00C4FF', width=2)  # Futuristic cyan
        ))
    elif chart_type == "Candlestick":
        fig_price.add_trace(go.Candlestick(
            x=filtered_df['Date'], open=filtered_df['Open'], high=filtered_df['High'],
            low=filtered_df['Low'], close=filtered_df['Close'], name='Price',
            increasing_line_color='#00C4FF', decreasing_line_color='#FF6B6B'  # Cyan up, Red down
        ))

    # Add volume bars on a secondary y-axis
    fig_price.add_trace(go.Bar(
        x=filtered_df['Date'], y=filtered_df['Volume'], name='Volume',
        yaxis='y2', marker_color='rgba(0, 196, 255, 0.2)'  # Transparent cyan
    ))

    fig_price.update_layout(
        title=f'<b>Price and Volume for {selected_ticker} ({chart_type} Chart)</b>',
        template="plotly_dark",
        yaxis=dict(title='Price (USD)', gridcolor='rgba(255,255,255,0.1)'),
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --- Sentiment and Article Count Charts ---
    col1, col2 = st.columns(2)
    with col1:
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['vader_avg_score'], name='VADER Score', mode='lines', line_color='#29B6F6'))  # Light Blue
        fig_sentiment.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['finbert_avg_score'], name='FinBERT Score', mode='lines', line_color='#FFEE58'))  # Yellow
        fig_sentiment.update_layout(
            title=f'<b>Daily Average Sentiment Scores</b>',
            template="plotly_dark",
            yaxis=dict(title='Average Sentiment Score', gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    with col2:
        fig_articles = go.Figure()
        fig_articles.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['article_count'], name='Article Count', marker_color='rgba(0, 196, 255, 0.5)'))  # Semi-transparent cyan
        fig_articles.update_layout(
            title=f'<b>Daily News Article Count</b>',
            template="plotly_dark",
            yaxis=dict(title='Number of Articles', gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_articles, use_container_width=True)
else:
    st.warning("No data available to display charts for the selected date range.")

# --- Raw Data View ---
with st.expander("View Raw Data for Selection"):
    st.dataframe(filtered_df)

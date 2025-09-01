import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

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
# Main Dashboard
# ==============================================================================
st.title("Stock Price and News Sentiment Tracker  Fintech SaaS")

st.markdown("""
Welcome to the dashboard. This tool visualizes the relationship between stock prices and the sentiment of financial news.
Use the sidebar to select a stock ticker and a date range to explore the data.
""")

# ==============================================================================
# Data Loading & Caching
# ==============================================================================

@st.cache_data
def load_data():
    """
    Loads the processed stock and sentiment data. This function is cached
    to ensure data is loaded only once per session, improving performance.
    It first attempts to load from a PostgreSQL database and falls back to a
    local CSV file if the database connection fails.
    """
    try:
        # Attempt to connect to the database
        # In a real-world app, these credentials would be managed more securely
        conn = psycopg2.connect(
            dbname="stock_sentiment",
            user="postgres",
            password="password",
            host="localhost",
            port="5432",
            connect_timeout=3  # Set a short timeout
        )
        st.info("Database connection successful. Loading data...")
        df = pd.read_sql("SELECT * FROM stock_data ORDER BY \"Date\" ASC", conn)
        # Convert date column to datetime objects
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

# Load the data
data_df = load_data()

if data_df is None:
    st.stop() # Halt execution if no data could be loaded

st.success(f"Successfully loaded {len(data_df)} records.")

# ==============================================================================
# Key Metrics
# ==============================================================================
st.header("Key Metrics")

if not filtered_df.empty:
    # Calculate metrics
    start_price = filtered_df['Close'].iloc[0]
    end_price = filtered_df['Close'].iloc[-1]
    price_change = end_price - start_price
    price_change_pct = (price_change / start_price) * 100

    avg_finbert_score = filtered_df['finbert_avg_score'].mean()
    total_articles = filtered_df['article_count'].sum()

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label=f"Closing Price ({selected_end_date.strftime('%Y-%m-%d')})",
            value=f"${end_price:,.2f}"
        )
    with col2:
        st.metric(
            label="Price Change (in range)",
            value=f"${price_change:,.2f}",
            delta=f"{price_change_pct:.2f}%"
        )
    with col3:
        st.metric(
            label="Avg. FinBERT Score",
            value=f"{avg_finbert_score:.3f}"
        )
    with col4:
        st.metric(
            label="Total News Articles",
            value=f"{int(total_articles)}"
        )
else:
    st.warning("No data available for the selected date range.")

# ==============================================================================
# Visualizations
# ==============================================================================
st.header("Charts")

if not filtered_df.empty:
    # --- Price and Volume Chart ---
    fig_price = go.Figure()

    # Add price line
    fig_price.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Close'],
        name='Close Price',
        line=dict(color='#1E88E5', width=2)
    ))

    # Add volume bars on a secondary y-axis
    fig_price.add_trace(go.Bar(
        x=filtered_df['Date'],
        y=filtered_df['Volume'],
        name='Volume',
        yaxis='y2',
        marker_color='#D6EAF8'
    ))

    fig_price.update_layout(
        title=f'<b>Price and Volume for {selected_ticker}</b>',
        template="plotly_white",
        yaxis=dict(title='Price (USD)'),
        yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99, bordercolor='lightgrey', borderwidth=1)
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --- Sentiment and Article Count Charts ---
    col1, col2 = st.columns(2)

    with col1:
        # Sentiment Scores Chart
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['vader_avg_score'], name='VADER Score', mode='lines', line_color='#2ECC71'))
        fig_sentiment.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['finbert_avg_score'], name='FinBERT Score', mode='lines', line_color='#F39C12'))
        fig_sentiment.update_layout(
            title=f'<b>Daily Average Sentiment Scores</b>',
            template="plotly_white",
            yaxis=dict(title='Average Sentiment Score'),
            legend=dict(x=0.01, y=0.99, bordercolor='lightgrey', borderwidth=1)
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        # Article Count Chart
        fig_articles = go.Figure()
        fig_articles.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['article_count'], name='Article Count', marker_color='#85C1E9'))
        fig_articles.update_layout(
            title=f'<b>Daily News Article Count</b>',
            template="plotly_white",
            yaxis=dict(title='Number of Articles')
        )
        st.plotly_chart(fig_articles, use_container_width=True)

else:
    st.warning("No data available to display charts for the selected date range.")

# ==============================================================================
# Raw Data View
# ==============================================================================
with st.expander("View Raw Data for Selection"):
    st.dataframe(filtered_df)

# ==============================================================================
# Sidebar Controls
# ==============================================================================
st.sidebar.header("Dashboard Controls")

# Ticker selection
tickers = sorted(data_df['Ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", tickers)

# Filter data based on selected ticker
ticker_df = data_df[data_df['Ticker'] == selected_ticker]

# Date range selection
min_date = ticker_df['Date'].min().date()
max_date = ticker_df['Date'].max().date()

selected_start_date = st.sidebar.date_input(
    "Start Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

selected_end_date = st.sidebar.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

if selected_start_date > selected_end_date:
    st.sidebar.error("Error: Start date must be before end date.")
    st.stop()

# Filter data based on date range
filtered_df = ticker_df[
    (ticker_df['Date'].dt.date >= selected_start_date) &
    (ticker_df['Date'].dt.date <= selected_end_date)
]

# About section in the sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This dashboard visualizes stock prices and financial news sentiment. "
    "The data pipeline is built with Python, and the dashboard is powered by Streamlit. "
    "Sentiment analysis is performed using VADER and FinBERT models."
)

# ==============================================================================
# Main Page Content
# ==============================================================================

# Filtered data will be used by the sections below
st.subheader(f"Displaying data for: {selected_ticker}")

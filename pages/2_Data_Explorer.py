import streamlit as st
import pandas as pd
import sys
import os

# Add src to path to import dashboard_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dashboard_utils import load_data

st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Explorer")
st.markdown("Explore the raw stock and sentiment data used in the dashboard.")

# --- Data Loading ---
data_df, source = load_data()
if data_df is None:
    st.error("Data could not be loaded. Please ensure the pipeline has run successfully.")
    st.stop()

# --- Interactive Data Table ---
st.markdown("### Processed Data")
st.info("Use the column headers to sort the data. Click the fullscreen icon to view more data at once.")

# Use st.dataframe for interactive features
st.dataframe(data_df)

# --- Data Filtering ---
st.markdown("### Filter Data")
st.warning("This is an example of advanced filtering. You can add more complex filters here.")

col1, col2, col3 = st.columns(3)
with col1:
    # Filter by Ticker
    tickers = ["All"] + sorted(data_df['Ticker'].unique())
    selected_ticker = st.selectbox("Filter by Ticker", tickers)
    if selected_ticker != "All":
        data_df = data_df[data_df['Ticker'] == selected_ticker]

with col2:
    # Filter by Sentiment Score
    sentiment_threshold = st.slider("Filter by FinBERT Score", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
    data_df = data_df[data_df['finbert_avg_score'] >= sentiment_threshold]

with col3:
    # Filter by Article Count
    min_articles = st.number_input("Minimum Daily Articles", min_value=0, value=0)
    data_df = data_df[data_df['article_count'] >= min_articles]

st.dataframe(data_df)

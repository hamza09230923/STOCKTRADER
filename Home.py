import streamlit as st
import sys
import os

# Add src to path to import dashboard_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from dashboard_utils import load_data

st.set_page_config(
    page_title="Stock & Sentiment Tracker",
    page_icon="🏠",
    layout="wide"
)

# --- Main Page Content ---
st.title("Welcome to the Stock & Sentiment Tracker!")
st.markdown("---")
st.markdown("""
This application provides a comprehensive platform to analyze stock market data and financial news sentiment.
Navigate through the different pages using the sidebar to explore the features.

### 📈 Dashboard
The main dashboard for visualizing stock prices, volume, and sentiment scores over time.
Select different stocks and date ranges to dive deep into the data.

### 📊 Data Explorer
View the raw, processed data in a tabular format. You can sort, filter, and search the data to find exactly what you're looking for.

### ℹ️ About
Learn more about the project, its data sources, the technologies used, and the developer behind it.

**To get started, select a page from the sidebar on the left.**
""")

# --- Sidebar ---
st.sidebar.title("Navigation")
st.sidebar.info("Select a page above to get started.")

# Load data and show connection status in the sidebar
with st.spinner("Loading data..."):
    data_df, source = load_data()

if source == "db":
    st.sidebar.success("Connected to Database")
elif source == "csv":
    st.sidebar.warning("Using local CSV file")
else:
    st.sidebar.error("Data loading failed")

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This dashboard visualizes stock prices and financial news sentiment. "
    "The data pipeline is built with Python, and the dashboard is powered by Streamlit."
)

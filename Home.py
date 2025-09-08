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
st.title("🏠 Welcome to the Stock & Sentiment Tracker!")
st.markdown("This application provides a comprehensive platform to analyze stock market data and financial news sentiment. Navigate through the different pages using the sidebar to explore the features.")
st.markdown("---")

# --- Feature Highlights ---
st.subheader("Key Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📈 Dashboard")
    st.markdown("Visualize stock prices, volume, and sentiment scores over time. Dive deep into the data with interactive charts and our ML-powered prediction model.")

with col2:
    st.markdown("#### 📊 Data Explorer")
    st.markdown("View the raw, processed data in a tabular format. Sort, filter, and even edit the data to find exactly what you're looking for.")

with col3:
    st.markdown("#### ℹ️ About")
    st.markdown("Learn more about the project, its data sources, the technologies used, and the developer behind it. Find links to the data sources and GitHub repo.")

st.markdown("---")
st.info("**To get started, select a page from the sidebar on the left.**")

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

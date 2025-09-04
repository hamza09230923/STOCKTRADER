import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About This Project")

st.markdown("""
This is a comprehensive, end-to-end data pipeline and visualization dashboard that tracks stock prices and financial news sentiment.
This project aims to explore the correlation between market sentiment and stock performance.
It also includes a machine learning pipeline to predict stock price movements based on historical data and sentiment scores.

### Key Features:
- **Automated Data Pipeline:** Fetches and processes data from Yahoo Finance and Finlight.me.
- **Interactive Dashboard:** Visualizes stock and sentiment data with Plotly and Streamlit.
- **ML Model Training:** Trains predictive models using XGBoost and Logistic Regression.
- **Scheduled Updates:** The data pipeline runs on a schedule to keep the data fresh.

This project was improved by Jules, an AI software engineer.
""")

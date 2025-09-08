import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About This Project")

st.markdown("""
This project is a comprehensive, end-to-end data pipeline and visualization dashboard that tracks stock prices and financial news sentiment. The goal is to explore the correlation between market sentiment and stock performance and to provide a platform for data-driven financial analysis.
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Key Features")
    st.markdown("""
    - **Automated Data Pipeline:** Fetches and processes data from Yahoo Finance and Finlight.me.
    - **Interactive Dashboard:** Visualizes stock and sentiment data with Plotly and Streamlit.
    - **ML Model Training:** Trains predictive models using XGBoost and Logistic Regression.
    - **ML-Powered Predictions:** Provides daily predictions on stock price movement.
    """)

with col2:
    st.subheader("🛠️ Technologies Used")
    st.markdown("""
    - **Python:** The core language for the entire project.
    - **Streamlit:** For building the interactive web dashboard.
    - **Plotly:** For creating rich, interactive visualizations.
    - **Pandas & NumPy:** For data manipulation and analysis.
    - **Scikit-learn & XGBoost:** For building and training machine learning models.
    - **PostgreSQL:** As the database for storing processed data.
    """)

st.markdown("---")

st.subheader("🔗 Important Links")
st.markdown("""
- **Data Sources:**
  - [Yahoo Finance](https://finance.yahoo.com/): For historical stock price data.
  - [Finlight.me API](https://finlight.me/): For financial news and sentiment analysis.
- **Source Code:**
  - [GitHub Repository](https://github.com/your-repo-link): Find the full source code for this project here.
""")

st.markdown("---")
st.info("This project was improved by Jules, an AI software engineer.")

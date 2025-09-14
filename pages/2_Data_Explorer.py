import streamlit as st
import pandas as pd
import sys
import os

# Add src to path to import dashboard_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dashboard_utils import load_data
from advanced_analysis import analyze_aspect_sentiment

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

# --- Data Filtering ---
st.markdown("### 🔍 Filter and Explore Data")
st.info("Use the controls below to filter the data. The table is editable, but changes will not be saved.")

# Create a copy of the dataframe to avoid modifying the cached version
filtered_df = data_df.copy()

col1, col2, col3 = st.columns(3)
with col1:
    # Filter by Ticker
    tickers = ["All"] + sorted(filtered_df['Ticker'].unique())
    selected_ticker = st.selectbox("Filter by Ticker", tickers)
    if selected_ticker != "All":
        filtered_df = filtered_df[filtered_df['Ticker'] == selected_ticker]

with col2:
    # Filter by Date Range
    min_date = filtered_df['Date'].min().date()
    max_date = filtered_df['Date'].max().date()
    selected_start_date, selected_end_date = st.date_input(
        "Filter by Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    filtered_df = filtered_df[
        (filtered_df['Date'].dt.date >= selected_start_date) &
        (filtered_df['Date'].dt.date <= selected_end_date)
    ]

with col3:
    # Filter by Sentiment Score
    sentiment_threshold = st.slider("Minimum FinBERT Score", min_value=-1.0, max_value=1.0, value=-1.0, step=0.1)
    filtered_df = filtered_df[filtered_df['finbert_avg_score'] >= sentiment_threshold]

# --- ABSA Analysis Expander ---
with st.expander("🔬 **Aspect-Based Sentiment Analysis (ABSA)**"):
    st.info("Analyze the sentiment towards specific tickers within the news headlines shown below. This provides a more granular view than the daily average sentiment.")

    # Use a unique key for the session state based on the filtered data's hash
    # This is a bit complex, but ensures we don't show old results for new filters
    filtered_hash = pd.util.hash_pandas_object(filtered_df).sum()
    absa_key = f'absa_results_{filtered_hash}'

    if st.button("Run ABSA on Filtered Data"):
        if 'headlines' in filtered_df.columns and not filtered_df['headlines'].str.strip().eq('').all():
            with st.spinner("Performing ABSA... This is computationally intensive and may take some time."):
                # Create the input dataframe and the list of aspects (tickers)
                absa_input_df = pd.DataFrame({'title': filtered_df['headlines'][filtered_df['headlines'].str.strip().ne('')]})
                aspects = filtered_df['Ticker'].unique().tolist()

                # Run the analysis
                absa_results_df = analyze_aspect_sentiment(absa_input_df, aspects)
                st.session_state[absa_key] = absa_results_df
        else:
            st.warning("No headlines available in the filtered data to analyze.")
            st.session_state[absa_key] = None

    # Display the results if they exist in the session state
    if absa_key in st.session_state:
        results = st.session_state[absa_key]
        if results is not None and not results.empty:
            st.subheader("ABSA Results")
            st.dataframe(results, use_container_width=True)
        elif results is None:
            st.info("No aspects found in the current selection of headlines.")

# --- Interactive Data Table ---
st.data_editor(
    filtered_df,
    num_rows="dynamic",
    use_container_width=True
)

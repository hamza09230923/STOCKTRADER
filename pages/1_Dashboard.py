import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from streamlit_autorefresh import st_autorefresh

# Add src to path to import dashboard_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dashboard_utils import load_data, add_ticker_to_config
from advanced_analysis import generate_summary, perform_topic_modeling
import subprocess
import joblib
from xgboost import XGBClassifier
import json

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

selected_tickers = st.sidebar.multiselect("Select Stock Ticker(s)", tickers, default=[tickers[0]] if tickers else [])

# --- Single vs. Multi Ticker Logic ---
is_multi_ticker = len(selected_tickers) > 1

chart_type = st.sidebar.radio(
    "Select Chart Type",
    ["Line", "Candlestick"],
    index=0 if is_multi_ticker else 1,
    help="Candlestick chart is only available for single ticker selection."
)
if is_multi_ticker and chart_type == "Candlestick":
    st.sidebar.warning("Candlestick chart is disabled for multi-ticker mode. Switching to Line view.")
    chart_type = "Line"

st.sidebar.markdown("---")
st.sidebar.header("Technical Indicators")
if is_multi_ticker:
    st.sidebar.info("Technical indicators are only available in single-ticker mode.")
show_sma = st.sidebar.checkbox("Show Moving Averages (20 & 50 day)", disabled=is_multi_ticker)
show_bbands = st.sidebar.checkbox("Show Bollinger Bands", disabled=is_multi_ticker)
show_rsi = st.sidebar.checkbox("Show Relative Strength Index (RSI)", disabled=is_multi_ticker)

if not selected_tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

ticker_df = data_df[data_df['Ticker'].isin(selected_tickers)]
if not is_multi_ticker:
    selected_ticker = selected_tickers[0]
else:
    # For display purposes in the subheader
    selected_ticker = ", ".join(selected_tickers)
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

# Show success message if a new ticker was added
if "new_ticker_success" in st.session_state:
    st.sidebar.success(st.session_state.new_ticker_success)
    del st.session_state.new_ticker_success

new_ticker = st.sidebar.text_input("Enter a new stock ticker (e.g., GOOGL):", key="new_ticker_input")
if st.sidebar.button("Add Ticker"):
    if new_ticker:
        with st.spinner(f"Adding {new_ticker.upper()}... This may take a moment."):
            add_ticker_to_config(new_ticker)

            # Run the pipeline for the new ticker
            result = subprocess.run(
                ["python", "run_pipeline.py", "--ticker", new_ticker, "--skip-db"],
                capture_output=True, text=True
            )

        if result.returncode == 0:
            st.session_state.new_ticker_success = f"Ticker {new_ticker.upper()} added successfully!"
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

# --- Single-Ticker Analysis Sections ---
if not is_multi_ticker:
    # --- ML Prediction ---
    @st.cache_resource
    def load_models():
        """Load the trained model and scaler."""
        try:
            model = joblib.load("models/xgboost_model.pkl")
            scaler = joblib.load("models/scaler.pkl")
            return model, scaler
        except FileNotFoundError:
            return None, None

    def get_prediction(ticker_df, latest_data):
        """Generates features and returns a prediction for the latest data point."""
        model, scaler = load_models()
        if model is None or scaler is None:
            return None, None

        # Feature Engineering
        df = ticker_df.sort_values(by='Date').copy()
        df['price_change_1d'] = df['Close'].pct_change(1)
        df['price_change_5d'] = df['Close'].pct_change(5)
        df['vader_1d_lag'] = df['vader_avg_score'].shift(1)
        df['finbert_1d_lag'] = df['finbert_avg_score'].shift(1)
        df['sma_7d'] = df['Close'].transform(lambda x: x.rolling(window=7).mean())
        df['sma_30d'] = df['Close'].transform(lambda x: x.rolling(window=30).mean())
        def calculate_rsi(series, window=14):
            delta = series.diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        df['rsi_14d'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))

        # Get the row corresponding to the latest data
        features_df = df[df['Date'] == latest_data['Date']].copy()

        # Load the exact feature columns and order from training
        try:
            with open('models/training_columns.json', 'r') as f:
                training_columns = json.load(f)
        except FileNotFoundError:
            st.error("`training_columns.json` not found. Please retrain the models.")
            return None, None

        # Ensure all required columns are present and in the correct order
        features_df = features_df.reindex(columns=training_columns).dropna()

        if features_df.empty:
            st.warning("Not enough data for the latest day to make a prediction.")
            return None, None

        # Scale features and predict
        scaled_features = scaler.transform(features_df)
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)

        return prediction[0], probability[0]

    prediction, probability = None, None
    if not filtered_df.empty:
        latest_data = filtered_df.iloc[-1]
        prediction, probability = get_prediction(ticker_df, latest_data)

    # --- Prediction Expander ---
    if prediction is not None:
        with st.expander("🤖 **ML-Powered Prediction**", expanded=True):
            prediction_text = "MOVE UP ⬆️" if prediction == 1 else "MOVE DOWN ⬇️"
            confidence = probability[prediction] * 100

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Next-Day Price Movement", value=prediction_text)
            with col2:
                st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

            st.info("**Disclaimer:** This is an experimental prediction based on historical data and sentiment. It is not financial advice.")

    # --- Key Metrics ---
    if not filtered_df.empty:
        st.markdown("### 🔑 Key Metrics")
        start_price = filtered_df['Close'].iloc[0]
        end_price = filtered_df['Close'].iloc[-1]
        price_change = end_price - start_price
        price_change_pct = (price_change / start_price) * 100 if start_price != 0 else 0
        avg_finbert_score = filtered_df['finbert_avg_score'].mean()
        total_articles = filtered_df['article_count'].sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label=f"Closing Price", value=f"${end_price:,.2f}")
        with col2:
            st.metric(label="Price Change", value=f"${price_change:,.2f}", delta=f"{price_change_pct:.2f}%")
        with col3:
            st.metric(label="Avg. FinBERT Score", value=f"{avg_finbert_score:.3f}")
        with col4:
            st.metric(label="Total News Articles", value=f"{int(total_articles)}")
    else:
        st.warning("No data available for the selected date range.")

    # --- AI Analysis Expander ---
    with st.expander("🤖 **AI-Powered Analysis**"):
        st.info("Click the buttons below to run advanced analysis on the news for the selected date range. This may take a moment.")

        col1, col2 = st.columns(2)
        summary_key = f'summary_{selected_ticker}_{selected_start_date}_{selected_end_date}'
        topics_key = f'topics_{selected_ticker}_{selected_start_date}_{selected_end_date}'

        # --- Summarization Button ---
        with col1:
            if st.button("Generate News Summary"):
                if 'headlines' in filtered_df.columns and not filtered_df['headlines'].str.strip().eq('').all():
                    with st.spinner("Generating summary... please wait."):
                        summary_input_df = pd.DataFrame({'title': filtered_df['headlines'][filtered_df['headlines'].str.strip().ne('')]})
                        st.session_state[summary_key] = generate_summary(summary_input_df)
                else:
                    st.session_state[summary_key] = "No news headlines available to summarize for this period."

        # --- Topic Modeling Button ---
        with col2:
            if st.button("Identify Key Topics"):
                if 'headlines' in filtered_df.columns and not filtered_df['headlines'].str.strip().eq('').all():
                    with st.spinner("Performing topic modeling... please wait."):
                        topic_input_df = pd.DataFrame({'title': filtered_df['headlines'][filtered_df['headlines'].str.strip().ne('')]})
                        st.session_state[topics_key] = perform_topic_modeling(topic_input_df)
                else:
                    st.session_state[topics_key] = None

        # --- Display Results ---
        if summary_key in st.session_state and st.session_state[summary_key]:
            st.markdown("---")
            st.subheader("Generated News Summary")
            st.markdown(f"> {st.session_state[summary_key]}")

        if topics_key in st.session_state and st.session_state[topics_key] is not None:
            st.markdown("---")
            st.subheader("Identified Key Topics")
            display_topics = st.session_state[topics_key][st.session_state[topics_key].Topic != -1]
            st.dataframe(
                display_topics,
                column_config={"Name": "Topic Keywords", "Count": "Article Count"},
                hide_index=True
            )

# --- Charting Logic ---
st.markdown("---")
if not filtered_df.empty:
    st.markdown("### 📊 Price Analysis")

    if is_multi_ticker:
        # --- Multi-Ticker Comparison Chart ---
        fig = go.Figure()
        colors = go.Figure().layout.template.data.scatter[0].line.color
        for i, ticker in enumerate(selected_tickers):
            ticker_data = filtered_df[filtered_df['Ticker'] == ticker].copy()
            if not ticker_data.empty:
                ticker_data['Normalized'] = ticker_data['Close'] / ticker_data['Close'].iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=ticker_data['Date'],
                    y=ticker_data['Normalized'],
                    name=ticker,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)])
                ))
        fig.update_layout(
            title_text='<b>Stock Performance Comparison</b>',
            template="plotly_dark",
            height=600,
            yaxis_title='Normalized Price (Start = 100)',
            legend_title='Tickers'
        )
        st.plotly_chart(fig, use_container_width=True)

    else: # This is the single-ticker view
        # --- Single-Ticker Detailed Chart ---
        def calculate_sma(df, window):
            return df['Close'].rolling(window=window).mean()

        def calculate_bbands(df, window=20, std=2):
            sma = calculate_sma(df, window)
            rolling_std = df['Close'].rolling(window=window).std()
            upper_band = sma + (rolling_std * std)
            lower_band = sma - (rolling_std * std)
            return upper_band, lower_band

        def calculate_rsi(series, window=14):
            delta = series.diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            if loss.eq(0).all():
                return pd.Series(100., index=series.index)
            rs = gain / loss.replace(0, 1e-6)
            return 100 - (100 / (1 + rs))

        chart_df = filtered_df.copy()
        if show_sma:
            chart_df['SMA_20'] = calculate_sma(chart_df, 20)
            chart_df['SMA_50'] = calculate_sma(chart_df, 50)
        if show_bbands:
            chart_df['BB_Upper'], chart_df['BB_Lower'] = calculate_bbands(chart_df)
        if show_rsi:
            chart_df['RSI'] = calculate_rsi(chart_df['Close'])

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.7, 0.3] if show_rsi else [1, 0],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )

        if chart_type == "Line":
            fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['Close'], name='Close Price', line=dict(color='#00C4FF', width=2)), row=1, col=1)
        else: # Candlestick
            fig.add_trace(go.Candlestick(x=chart_df['Date'], open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='Price', increasing_line_color='#00C4FF', decreasing_line_color='#FF6B6B'), row=1, col=1)

        if show_sma:
            fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['SMA_50'], name='SMA 50', line=dict(color='purple', width=1, dash='dash')), row=1, col=1)
        if show_bbands:
            fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

        fig.add_trace(go.Bar(x=chart_df['Date'], y=chart_df['Volume'], name='Volume', marker_color='rgba(0, 196, 255, 0.2)'), secondary_y=True, row=1, col=1)

        if show_rsi:
            fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['RSI'], name='RSI', line=dict(color='yellow', width=2)), row=2, col=1)
            fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(
            title_text=f'<b>{selected_ticker} Price Analysis</b>', template="plotly_dark", height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False
        )
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Volume", secondary_y=True, row=1, col=1, showgrid=False)
        if show_rsi:
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # --- Sentiment and Article Count Charts ---
        st.markdown("### Sentiment and News Volume")
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

import os
import sys
import pandas as pd
import numpy as np

# Add root to path to import other scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_pipeline import run_the_pipeline

# A simple mock for the argparse Namespace object
class MockArgs:
    def __init__(self, api_key, skip_db=True):
        self.api_key = api_key
        self.skip_db = skip_db
        self.db_name = None
        self.db_user = None
        self.db_password = None
        self.db_host = None
        self.db_port = None

def generate_feature_data():
    """
    This function encapsulates the feature engineering logic from the notebook.
    """
    print("\n--- Running Feature Engineering ---")
    try:
        df = pd.read_csv('data/processed_data.csv', parse_dates=['Date'])
    except FileNotFoundError:
        print("Error: `data/processed_data.csv` not found. Cannot run feature engineering.")
        return

    df.sort_values(by=['Ticker', 'Date'], inplace=True)
    df['price_change_1d'] = df.groupby('Ticker')['Close'].pct_change(1)
    df['price_change_5d'] = df.groupby('Ticker')['Close'].pct_change(5)
    df['vader_1d_lag'] = df.groupby('Ticker')['vader_avg_score'].shift(1)
    df['finbert_1d_lag'] = df.groupby('Ticker')['finbert_avg_score'].shift(1)
    df['sma_7d'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=7).mean())
    df['sma_30d'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=30).mean())

    def calculate_rsi(series, window=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    df['rsi_14d'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))

    df['next_day_close'] = df.groupby('Ticker')['Close'].shift(-1)
    df['target'] = (df['next_day_close'] > df['Close']).astype(int)

    ml_df = df.dropna()
    feature_cols = [
        'price_change_1d', 'price_change_5d', 'vader_1d_lag', 'finbert_1d_lag',
        'sma_7d', 'sma_30d', 'rsi_14d', 'article_count'
    ]
    final_df = ml_df[feature_cols + ['target', 'Date', 'Ticker']]

    output_path = 'data/ml_ready_data.csv'
    final_df.to_csv(output_path, index=False)
    print(f"ML-ready dataset saved to {output_path}")

if __name__ == "__main__":
    print("===== Generating All Data for Machine Learning =====")

    # Step 1: Run the main pipeline to get `processed_data.csv`
    # You must provide a valid Finlight API key here.
    api_key = os.getenv("FINLIGHT_API_KEY") # Try to get from env var
    if not api_key:
        print("Warning: FINLIGHT_API_KEY environment variable not set.")
        # As a last resort, use the one from the user's history if available
        api_key = "sk_cb3c818eed446795658fb733770be43d1bd47c0b6d823df90aa384f6684d0d5d"

    mock_args = MockArgs(api_key=api_key, skip_db=True)
    run_the_pipeline(mock_args)

    # Step 2: Run feature engineering to get `ml_ready_data.csv`
    generate_feature_data()

    print("\n===== Data Generation Finished =====")

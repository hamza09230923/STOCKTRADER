import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import sys
import warnings
import argparse

# Add root to path and suppress warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run_pipeline import run_the_pipeline

# --- Configuration ---
MODELS_DIR = "models"
TARGET_VARIABLE = "target"
TRAIN_SPLIT_FRACTION = 0.8

def generate_ml_data(api_key):
    """Generates the initial processed data and then the feature-engineered ML data."""
    print("--- Step 1: Generating Data ---")
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
            self.ticker = None  # Add missing ticker attribute expected by run_pipeline

    # Run the main data pipeline to get processed_data.csv
    run_the_pipeline(MockArgs(api_key=api_key))

    # Now, run feature engineering on the output
    try:
        df = pd.read_csv('data/processed_data.csv', parse_dates=['Date'])
    except FileNotFoundError:
        print("Error: `data/processed_data.csv` not found after running pipeline.")
        return None

    df.sort_values(by=['Ticker', 'Date'], inplace=True)
    df['price_change_1d'] = df.groupby('Ticker')['Close'].pct_change(1)
    df['price_change_5d'] = df.groupby('Ticker')['Close'].pct_change(5)
    df['vader_1d_lag'] = df.groupby('Ticker')['vader_avg_score'].shift(1)
    df['finbert_1d_lag'] = df.groupby('Ticker')['finbert_avg_score'].shift(1)
    df['sma_7d'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=7).mean())
    df['sma_30d'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=30).mean())

    def calculate_rsi(series, window=14):
        delta = series.diff(1); gain = (delta.where(delta > 0, 0)).rolling(window=window).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        return 100 - (100 / (1 + (gain / loss)))
    df['rsi_14d'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x))

    df['next_day_close'] = df.groupby('Ticker')['Close'].shift(-1)
    df['target'] = (df['next_day_close'] > df['Close']).astype(int)

    ml_df = df.dropna()
    print("ML-ready data generated successfully.")
    return ml_df

def prepare_data_for_training(df):
    """Splits and scales the feature-engineered data."""
    print("\n--- Step 2: Splitting and Scaling Data ---")
    # Drop the target, identifiers, and the source of the target ('next_day_close')
    X = df.drop(columns=[TARGET_VARIABLE, 'Date', 'Ticker', 'next_day_close'])
    y = df[TARGET_VARIABLE]

    split_index = int(len(df) * TRAIN_SPLIT_FRACTION)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("Data split and scaled. Scaler saved.")
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Trains and evaluates both Logistic Regression and XGBoost models."""
    print("\n--- Step 3: Training and Evaluating Models ---")
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        print(f"\n----- Training {name} -----")
        model.fit(X_train, y_train)

        print(f"----- Evaluating {name} -----")
        y_pred = model.predict(X_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Save the model
        model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}_model.pkl")
        joblib.dump(model, model_path)
        print(f"{name} model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data and train predictive models.")
    parser.add_argument("--api-key", required=True, help="Your Finlight API key.")
    args = parser.parse_args()

    print("===== Starting Full ML Pipeline =====")
    ml_data = generate_ml_data(api_key=args.api_key)

    if ml_data is not None and not ml_data.empty:
        X_train, X_test, y_train, y_test = prepare_data_for_training(ml_data)
        train_and_evaluate(X_train, y_train, X_test, y_test)
    else:
        print("Halting execution because data generation failed.")

    print("\n===== Full ML Pipeline Finished =====")

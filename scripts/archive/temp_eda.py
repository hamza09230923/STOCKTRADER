import pandas as pd
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

DATA_PATH = "data/processed_data.csv"

def explore_data(filepath):
    """
    Performs a basic exploratory data analysis on the processed dataset.
    """
    print(f"--- Exploratory Data Analysis for {filepath} ---")

    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        return

    print("\n[1] DataFrame Info:")
    print("--------------------")
    df.info()

    print("\n[2] Descriptive Statistics:")
    print("--------------------------")
    # Select only numeric columns for describe()
    numeric_cols = df.select_dtypes(include='number')
    print(numeric_cols.describe())

    print("\n[3] Correlation Matrix:")
    print("-----------------------")
    # Calculate correlations on numeric columns
    correlation_matrix = numeric_cols.corr()
    print(correlation_matrix)

    print("\n[4] Key Correlations with Closing Price:")
    print("-----------------------------------------")
    # Focus on correlations with the 'Close' price
    close_correlations = correlation_matrix['Close'].sort_values(ascending=False)
    print(close_correlations)

    print("\n--- EDA Finished ---")

if __name__ == "__main__":
    explore_data(DATA_PATH)

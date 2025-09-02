import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Define the list of stock tickers
TICKERS = ["AAPL", "TSLA", "NVDA", "JPM", "AMZN"]

# Define the time period for data extraction
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5*365)

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches historical stock price data for a list of tickers.
    """
    print(f"Fetching data for tickers: {', '.join(tickers)}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        stock_data = yf.download(tickers, start=start_date, end=end_date)
        if stock_data.empty:
            print("Warning: yfinance returned an empty DataFrame.")
            return None

        all_data = stock_data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
        print("Data fetched and processed successfully.")
        return all_data

    except Exception as e:
        print(f"An error occurred while fetching stock data: {e}")
        return None

def save_data_to_csv(data, filepath):
    """
    Saves a pandas DataFrame to a CSV file and verifies its creation.
    """
    if data is not None and not data.empty:
        print(f"DataFrame to be saved has {len(data)} rows.")
        dir_path = os.path.dirname(filepath)

        print(f"Ensuring directory '{dir_path}' exists...")
        os.makedirs(dir_path, exist_ok=True)

        print(f"Saving data to absolute path: {os.path.abspath(filepath)}...")
        data.to_csv(filepath, index=False)
        print("data.to_csv() command executed.")

        # --- VERIFICATION STEP ---
        print("\n--- Verifying file creation ---")
        if os.path.isfile(filepath):
            file_size = os.path.getsize(filepath)
            print(f"SUCCESS: Verified that '{filepath}' exists.")
            print(f"File size: {file_size} bytes.")
            if file_size == 0:
                print("Warning: The created file is empty.")
        else:
            print(f"FAILURE: Could not find file at '{os.path.abspath(filepath)}'.")
            if os.path.isdir(dir_path):
                print(f"Listing contents of '{dir_path}': {os.listdir(dir_path)}")
            else:
                print(f"Directory '{dir_path}' does not exist either.")
        print("--- Verification finished ---\n")

    else:
        print("No data to save.")

if __name__ == "__main__":
    print("--- Starting Stock Data Extraction ---")
    hist_data = fetch_stock_data(TICKERS, START_DATE, END_DATE)

    output_filepath = "data/stock_prices.csv"
    save_data_to_csv(hist_data, output_filepath)
    print("--- Stock Data Extraction Finished ---")

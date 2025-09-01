import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import psycopg2
from psycopg2 import sql

# --- Dependencies from other modules ---
import yfinance as yf
from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams

# Add src to path to import sentiment_analysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.sentiment_analysis import analyze_vader_sentiment, analyze_finbert_sentiment

# --- Constants ---
TICKERS = ["AAPL", "TSLA", "NVDA", "JPM", "AMZN"]
PROCESSED_CSV_PATH = "data/processed_data.csv"
DB_TABLE_NAME = "stock_data"

# ==============================================================================
# STEP 1: STOCK DATA EXTRACTION
# ==============================================================================
def fetch_stock_data(tickers):
    # ... (function body is the same as before)
    print("\n--- Step 1: Fetching Stock Data ---")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    print(f"Fetching data for: {', '.join(tickers)} from {start_date.date()} to {end_date.date()}")
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    if stock_data.empty:
        print("Error: yfinance returned an empty DataFrame.")
        return None
    all_data = stock_data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
    print("Stock data fetched successfully.")
    return all_data

# ==============================================================================
# STEP 2: NEWS DATA EXTRACTION
# ==============================================================================
def fetch_all_news(api, tickers):
    # ... (function body is the same as before)
    print("\n--- Step 2: Fetching News Data ---")
    all_articles = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    for ticker in tickers:
        try:
            params = GetArticlesParams(tickers=[ticker], from_=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'), page_size=100)
            response = api.articles.fetch_articles(params=params)
            if response and response.articles:
                for article in response.articles:
                    all_articles.append({'ticker': ticker.upper(), 'publish_date': article.publishDate, 'title': article.title, 'summary': article.summary, 'source': article.source, 'sentiment': article.sentiment, 'url': article.link})
                print(f"Found {len(response.articles)} articles for {ticker}.")
        except Exception as e:
            print(f"An error occurred fetching news for {ticker}: {e}")
    if not all_articles: return None
    print("News data fetched successfully.")
    return pd.DataFrame(all_articles)

# ==============================================================================
# STEP 3: DATA TRANSFORMATION
# ==============================================================================
def transform_data(stock_df, news_df):
    # ... (function body is the same as before)
    print("\n--- Step 3: Transforming Data ---")
    # 3a: Analyze Sentiment
    news_df['title'] = news_df['title'].astype(str)
    news_df['vader_score'] = news_df['title'].apply(analyze_vader_sentiment)
    finbert_results = news_df['title'].apply(lambda x: pd.Series(analyze_finbert_sentiment(x)))
    finbert_results.columns = ['finbert_label', 'finbert_score']
    news_with_sentiment = pd.concat([news_df, finbert_results], axis=1)
    # 3b: Aggregate Sentiment
    label_to_score = {'positive': 1, 'neutral': 0, 'negative': -1}
    news_with_sentiment['finbert_numeric_score'] = news_with_sentiment['finbert_label'].map(label_to_score)
    news_with_sentiment['finbert_weighted_score'] = news_with_sentiment['finbert_numeric_score'] * news_with_sentiment['finbert_score']
    news_with_sentiment['Date'] = pd.to_datetime(news_with_sentiment['publish_date']).dt.date
    agg_funcs = {'vader_score': 'mean', 'finbert_weighted_score': 'mean', 'title': 'count'}
    daily_sentiment = news_with_sentiment.groupby(['ticker', 'Date']).agg(agg_funcs).reset_index()
    daily_sentiment.rename(columns={'ticker': 'Ticker', 'title': 'article_count', 'vader_score': 'vader_avg_score', 'finbert_weighted_score': 'finbert_avg_score'}, inplace=True)
    # 3c: Merge Data
    stock_df['Date'] = pd.to_datetime(stock_df['Date'].dt.date)
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    final_df = pd.merge(stock_df, daily_sentiment, on=['Date', 'Ticker'], how='left')
    sentiment_cols = ['vader_avg_score', 'finbert_avg_score', 'article_count']
    final_df[sentiment_cols] = final_df[sentiment_cols].fillna(0)
    print("Data transformation complete.")
    return final_df

# ==============================================================================
# STEP 4: DATABASE OPERATIONS
# ==============================================================================
def get_db_connection(args):
    # ... (function body is the same as before)
    print("\n--- Step 4a: Connecting to Database ---")
    try:
        conn = psycopg2.connect(dbname=args.db_name, user=args.db_user, password=args.db_password, host=args.db_host, port=args.db_port)
        print("Database connection successful.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"CRITICAL: Could not connect to the database: {e}")
        return None

def setup_database_table(conn):
    # ... (function body is the same as before)
    print("\n--- Step 4b: Setting up Database Table ---")
    create_table_query = """
    CREATE TABLE stock_data (
        Date DATE NOT NULL, Ticker VARCHAR(10) NOT NULL, Close NUMERIC(15, 4) NOT NULL, High NUMERIC(15, 4) NOT NULL,
        Low NUMERIC(15, 4) NOT NULL, Open NUMERIC(15, 4) NOT NULL, Volume BIGINT NOT NULL, vader_avg_score FLOAT,
        finbert_avg_score FLOAT, article_count INTEGER, PRIMARY KEY (Date, Ticker)
    );"""
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {DB_TABLE_NAME};")
        cur.execute(create_table_query)
    print(f"Table '{DB_TABLE_NAME}' created successfully.")

def load_data_to_db(conn, df):
    # ... (function body is the same as before)
    print("\n--- Step 4c: Loading Data into Database ---")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    with conn.cursor() as cur:
        columns = ','.join([f'"{c}"' for c in df.columns])
        cur.copy_expert(sql=f"COPY {DB_TABLE_NAME} ({columns}) FROM STDIN WITH (FORMAT CSV)", file=buffer)
    print(f"Successfully loaded {len(df)} records.")

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
if __name__ == "__main__":
    print("===== Starting End-to-End Data Pipeline =====")
    parser = argparse.ArgumentParser(description="Run the full data pipeline for stock and news sentiment analysis.")
    parser.add_argument("--api-key", required=True, help="Your Finlight API key.")
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "stock_sentiment"), help="Database name")
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "postgres"), help="Database user")
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", "password"), help="Database password")
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"), help="Database host")
    parser.add_argument("--db-port", default=os.getenv("DB_PORT", "5432"), help="Database port")
    parser.add_argument("--skip-db", action="store_true", help="Skip all database operations.")
    args = parser.parse_args()

    # E-T-L Steps
    stock_df = fetch_stock_data(TICKERS)
    api = FinlightApi(ApiConfig(api_key=args.api_key))
    news_df = fetch_all_news(api, TICKERS)

    if stock_df is not None and news_df is not None:
        final_df = transform_data(stock_df, news_df)

        print(f"\n--- Saving processed data to {PROCESSED_CSV_PATH} for inspection ---")
        os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)
        final_df.to_csv(PROCESSED_CSV_PATH, index=False)
        print("CSV saved successfully.")

        if not args.skip_db:
            conn = None
            try:
                conn = get_db_connection(args)
                if conn:
                    setup_database_table(conn)
                    load_data_to_db(conn, final_df)
                    conn.commit()
                    print("\nDatabase operations completed successfully.")
            except Exception as e:
                print(f"\nAn error occurred during database operations: {e}")
                if conn: conn.rollback()
            finally:
                if conn: conn.close(); print("Database connection closed.")
        else:
            print("\n--skip-db flag was set. Skipping all database operations. --")

    else:
        print("\nPipeline halted due to errors in data extraction.")

    print("\n===== End-to-End Data Pipeline Finished =====")

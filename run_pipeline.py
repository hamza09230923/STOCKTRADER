import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
import io
import psycopg2
from psycopg2 import sql
import yfinance as yf
from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams
import time

# Import configuration
try:
    import config
except ImportError:
    print("CRITICAL: config.py not found. Please create it from config.py.example.")
    sys.exit(1)

# Add src to path to import sentiment_analysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.sentiment_analysis import analyze_vader_sentiment, analyze_finbert_sentiment

# ==============================================================================
# DATABASE SETUP AND TICKER MANAGEMENT
# ==============================================================================
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME, user=config.DB_USER, password=config.DB_PASSWORD,
            host=config.DB_HOST, port=config.DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"CRITICAL: Could not connect to the database: {e}")
        return None

def setup_database_tables(conn):
    """Ensures all necessary tables exist in the database."""
    print("\n--- Ensuring Database Tables Exist ---")
    with conn.cursor() as cur:
        # Main data table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{config.DB_TABLE_NAME}" (
                "Date" DATE NOT NULL, "Ticker" VARCHAR(10) NOT NULL,
                "Close" NUMERIC(15, 4), "High" NUMERIC(15, 4), "Low" NUMERIC(15, 4),
                "Open" NUMERIC(15, 4), "Volume" BIGINT,
                "vader_avg_score" FLOAT, "finbert_avg_score" FLOAT, "article_count" INTEGER,
                PRIMARY KEY ("Date", "Ticker")
            );
        """)
        # Table for tracked tickers
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tracked_tickers (
                ticker VARCHAR(10) PRIMARY KEY
            );
        """)
    print("Tables are ready.")

def get_tickers_to_process(conn):
    """
    Gets the list of tickers to process from the database.
    If the tracked_tickers table is empty, it seeds it from the config file.
    """
    print("\n--- Fetching Tickers to Process ---")
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM tracked_tickers;")
        tickers = [row[0] for row in cur.fetchall()]

        if not tickers:
            print("No tickers found in 'tracked_tickers' table. Seeding from config.py...")
            for ticker in config.TICKERS:
                cur.execute("INSERT INTO tracked_tickers (ticker) VALUES (%s) ON CONFLICT (ticker) DO NOTHING;", (ticker,))
            conn.commit()
            print(f"Seeded table with: {', '.join(config.TICKERS)}")
            return config.TICKERS

        print(f"Found tickers in database: {', '.join(tickers)}")
        return tickers

# ============================================================================== 
# DATA EXTRACTION
# ============================================================================== 
def fetch_stock_data(tickers, retries=3, delay=5):
    print("\n--- Step 1: Fetching Stock Data ---")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)
    print(f"Fetching data for: {', '.join(tickers)} from {start_date.date()} to {end_date.date()}")
    for attempt in range(retries):
        try:
            stock_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
            if stock_data.empty:
                raise ValueError("yfinance returned an empty DataFrame.")

            # Reformat the multi-level columns and stack
            all_data = []
            for ticker in tickers:
                if ticker in stock_data:
                    ticker_df = stock_data[ticker].copy()
                    ticker_df['Ticker'] = ticker
                    all_data.append(ticker_df.reset_index())

            if not all_data:
                print("Warning: yfinance did not return data for any of the requested tickers.")
                return None

            final_stock_df = pd.concat(all_data).rename(columns=str.title)
            print("Stock data fetched successfully.")
            return final_stock_df
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    print("CRITICAL: All attempts to fetch stock data failed.")
    return None

def fetch_all_news(api, tickers, retries=3, delay=5):
    print("\n--- Step 2: Fetching News Data ---")
    # Same as before...
    all_articles = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    for ticker in tickers:
        for attempt in range(retries):
            try:
                params = GetArticlesParams(tickers=[ticker], from_=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'), page_size=100)
                response = api.articles.fetch_articles(params=params)
                if response and response.articles:
                    for article in response.articles:
                        all_articles.append({'ticker': ticker.upper(), 'publish_date': article.publishDate, 'title': article.title})
                    print(f"Found {len(response.articles)} articles for {ticker}.")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retries} failed for {ticker}: {e}. Retrying...")
                time.sleep(delay)
        else:
            print(f"WARNING: All attempts to fetch news for {ticker} failed.")
    if not all_articles: return None
    return pd.DataFrame(all_articles)

# ============================================================================== 
# DATA TRANSFORMATION AND LOADING
# ============================================================================== 
def transform_data(stock_df, news_df):
    print("\n--- Step 3: Transforming Data ---")
    # Same as before...
    if news_df is None or news_df.empty:
        merged_df = stock_df.copy()
        merged_df[['vader_avg_score', 'finbert_avg_score', 'article_count']] = 0.0
        return merged_df

    news_df['title'] = news_df['title'].astype(str)
    news_df['vader_score'] = news_df['title'].apply(analyze_vader_sentiment)
    finbert_results = news_df['title'].apply(lambda x: pd.Series(analyze_finbert_sentiment(x)))
    finbert_results.columns = ['finbert_label', 'finbert_score']
    news_with_sentiment = pd.concat([news_df, finbert_results], axis=1)

    label_to_score = {'positive': 1, 'neutral': 0, 'negative': -1}
    news_with_sentiment['finbert_numeric_score'] = news_with_sentiment['finbert_label'].map(label_to_score)
    news_with_sentiment['finbert_weighted_score'] = news_with_sentiment['finbert_numeric_score'] * news_with_sentiment['finbert_score']
    news_with_sentiment['Date'] = pd.to_datetime(news_with_sentiment['publish_date']).dt.date

    agg_funcs = {'vader_score': 'mean', 'finbert_weighted_score': 'mean', 'title': 'count'}
    daily_sentiment = news_with_sentiment.groupby(['ticker', 'Date']).agg(agg_funcs).reset_index()
    daily_sentiment.rename(columns={'ticker': 'Ticker', 'title': 'article_count', 'vader_score': 'vader_avg_score', 'finbert_weighted_score': 'finbert_avg_score'}, inplace=True)

    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    final_df = pd.merge(stock_df, daily_sentiment, on=['Date', 'Ticker'], how='left')
    final_df[['vader_avg_score', 'finbert_avg_score', 'article_count']] = final_df[['vader_avg_score', 'finbert_avg_score', 'article_count']].fillna(0)
    return final_df


def upsert_data_to_db(conn, df):
    print("\n--- Step 4: Upserting Data into Database ---")
    # Same as before...
    temp_table_name = f"temp_{config.DB_TABLE_NAME}"
    with conn.cursor() as cur:
        cur.execute(f'CREATE TEMP TABLE "{temp_table_name}" (LIKE "{config.DB_TABLE_NAME}" INCLUDING DEFAULTS);')
        buffer = io.StringIO()
        # Ensure columns match the target table exactly
        df_for_db = df[['Date', 'Ticker', 'Close', 'High', 'Low', 'Open', 'Volume', 'vader_avg_score', 'finbert_avg_score', 'article_count']]
        df_for_db.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        columns = ','.join([f'"{c}"' for c in df_for_db.columns])
        cur.copy_expert(sql=f'COPY "{temp_table_name}" ({columns}) FROM STDIN WITH (FORMAT CSV)', file=buffer)

        update_cols = [f'"{col}" = EXCLUDED."{col}"' for col in df_for_db.columns if col not in ['Date', 'Ticker']]
        upsert_query = sql.SQL("""
            INSERT INTO "{target_table}" ({columns}) SELECT {columns} FROM "{temp_table}"
            ON CONFLICT ("Date", "Ticker") DO UPDATE SET {update_statements};
        """).format(target_table=sql.Identifier(config.DB_TABLE_NAME), columns=sql.SQL(', ').join(map(sql.Identifier, df_for_db.columns)),
                    temp_table=sql.Identifier(temp_table_name), update_statements=sql.SQL(', ').join(sql.SQL(s) for s in update_cols))
        cur.execute(upsert_query)
        cur.execute(f'DROP TABLE "{temp_table_name}";')
    print(f"Upserted {len(df_for_db)} records.")

# ============================================================================== 
# MAIN ORCHESTRATOR 
# ============================================================================== 
def run_the_pipeline(args):
    """Main ETL pipeline logic."""
    conn = get_db_connection()
    if not conn:
        sys.exit(1)

    try:
        setup_database_tables(conn)
        tickers_to_process = get_tickers_to_process(conn)

        if not tickers_to_process:
            print("No tickers to process. Exiting.")
            return

        stock_df = fetch_stock_data(tickers_to_process)

        if config.FINLIGHT_API_KEY == "YOUR_API_KEY_HERE":
            print("WARNING: Finlight API key is not set. Skipping news fetching.")
            news_df = pd.DataFrame()
        else:
            api = FinlightApi(ApiConfig(api_key=config.FINLIGHT_API_KEY))
            news_df = fetch_all_news(api, tickers_to_process)

        if stock_df is not None:
            final_df = transform_data(stock_df, news_df)

            print(f"\n--- Saving processed data to {config.PROCESSED_CSV_PATH} ---")
            os.makedirs(os.path.dirname(config.PROCESSED_CSV_PATH), exist_ok=True)
            final_df.to_csv(config.PROCESSED_CSV_PATH, index=False)
            print("CSV saved successfully.")

            if not args.skip_db:
                upsert_data_to_db(conn, final_df)
                conn.commit()
                print("\nDatabase operations completed successfully.")
        else:
            print("\nPipeline halted due to errors in data extraction.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during pipeline execution: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full data pipeline for stock and news sentiment analysis.")
    parser.add_argument("--skip-db", action="store_true", help="Skip all database operations.")
    args = parser.parse_args()

    print("===== Starting End-to-End Data Pipeline =====")
    run_the_pipeline(args)
    print("\n===== End-to-End Data Pipeline Finished =====")

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
# STEP 1: STOCK DATA EXTRACTION
# ============================================================================== 
def fetch_stock_data(tickers, retries=3, delay=5):
    print("\n--- Step 1: Fetching Stock Data ---")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)
    print(f"Fetching data for: {', '.join(tickers)} from {start_date.date()} to {end_date.date()}")
    for attempt in range(retries):
        try:
            stock_data = yf.download(tickers, start=start_date, end=end_date)
            if stock_data.empty:
                raise ValueError("yfinance returned an empty DataFrame.")
            all_data = stock_data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
            print("Stock data fetched successfully.")
            return all_data
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    print("CRITICAL: All attempts to fetch stock data failed.")
    return None

# ==============================================================================
# STEP 2: NEWS DATA EXTRACTION
# ==============================================================================
def fetch_all_news(api, tickers, retries=3, delay=5):
    print("\n--- Step 2: Fetching News Data ---")
    all_articles = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30) # Fetch news for the last 30 days
    for ticker in tickers:
        for attempt in range(retries):
            try:
                params = GetArticlesParams(
                    tickers=[ticker],
                    from_=start_date.strftime('%Y-%m-%d'),
                    to=end_date.strftime('%Y-%m-%d'),
                    page_size=100
                )
                response = api.articles.fetch_articles(params=params)
                if response and response.articles:
                    for article in response.articles:
                        all_articles.append({
                            'ticker': ticker.upper(),
                            'publish_date': article.publishDate,
                            'title': article.title,
                            'summary': article.summary,
                            'source': article.source,
                            'sentiment': article.sentiment,
                            'url': article.link
                        })
                    print(f"Found {len(response.articles)} articles for {ticker}.")
                break # Success, break the retry loop
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retries} failed for {ticker}: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
        else:
            print(f"WARNING: All attempts to fetch news for {ticker} failed.")
    if not all_articles:
        return None
    print("News data fetched successfully.")
    return pd.DataFrame(all_articles)

# ============================================================================== 
# STEP 3: DATA TRANSFORMATION
# ============================================================================== 
def transform_data(stock_df, news_df):
    print("\n--- Step 3: Transforming Data ---")
    if news_df is None or news_df.empty:
        print("No news data to transform. Skipping sentiment analysis.")
        merged_df = stock_df.copy()
        merged_df[['vader_avg_score', 'finbert_avg_score', 'article_count']] = 0.0
        return merged_df

    # Analyze sentiment
    news_df['title'] = news_df['title'].astype(str)
    news_df['vader_score'] = news_df['title'].apply(analyze_vader_sentiment)
    finbert_results = news_df['title'].apply(lambda x: pd.Series(analyze_finbert_sentiment(x)))
    finbert_results.columns = ['finbert_label', 'finbert_score']
    news_with_sentiment = pd.concat([news_df, finbert_results], axis=1)

    # Aggregate sentiment
    label_to_score = {'positive': 1, 'neutral': 0, 'negative': -1}
    news_with_sentiment['finbert_numeric_score'] = news_with_sentiment['finbert_label'].map(label_to_score)
    news_with_sentiment['finbert_weighted_score'] = (
        news_with_sentiment['finbert_numeric_score'] * news_with_sentiment['finbert_score']
    )
    news_with_sentiment['Date'] = pd.to_datetime(news_with_sentiment['publish_date']).dt.date
    agg_funcs = {
        'vader_score': 'mean',
        'finbert_weighted_score': 'mean',
        'title': 'count'
    }
    daily_sentiment = news_with_sentiment.groupby(['ticker', 'Date']).agg(agg_funcs).reset_index()
    daily_sentiment.rename(columns={
        'ticker': 'Ticker',
        'title': 'article_count',
        'vader_score': 'vader_avg_score',
        'finbert_weighted_score': 'finbert_avg_score'
    }, inplace=True)

    # Merge data
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
def get_db_connection():
    print("\n--- Step 4a: Connecting to Database ---")
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT
        )
        print("Database connection successful.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"CRITICAL: Could not connect to the database: {e}")
        return None

def setup_database_table(conn):
    print("\n--- Step 4b: Ensuring Database Table Exists ---")
    # This query creates the table if it doesn't exist, but does not modify it if it does.
    # This is much safer than DROP/CREATE.
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS "{config.DB_TABLE_NAME}" (
        "Date" DATE NOT NULL,
        "Ticker" VARCHAR(10) NOT NULL,
        "Close" NUMERIC(15, 4) NOT NULL,
        "High" NUMERIC(15, 4) NOT NULL,
        "Low" NUMERIC(15, 4) NOT NULL,
        "Open" NUMERIC(15, 4) NOT NULL,
        "Volume" BIGINT NOT NULL,
        "vader_avg_score" FLOAT,
        "finbert_avg_score" FLOAT,
        "article_count" INTEGER,
        PRIMARY KEY ("Date", "Ticker")
    );"""
    with conn.cursor() as cur:
        cur.execute(create_table_query)
    print(f"Table '{config.DB_TABLE_NAME}' is ready.")

def upsert_data_to_db(conn, df):
    print("\n--- Step 4c: Upserting Data into Database ---")

    # Define a temporary table
    temp_table_name = f"temp_{config.DB_TABLE_NAME}"

    with conn.cursor() as cur:
        # 1. Create a temporary table like the target table
        cur.execute(f'CREATE TEMP TABLE "{temp_table_name}" (LIKE "{config.DB_TABLE_NAME}" INCLUDING DEFAULTS);')

        # 2. Load data into the temporary table using copy_expert
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        columns = ','.join([f'"{c}"' for c in df.columns])
        cur.copy_expert(sql=f'COPY "{temp_table_name}" ({columns}) FROM STDIN WITH (FORMAT CSV)', file=buffer)
        print(f"Loaded {len(df)} records into the temporary table.")

        # 3. Perform the UPSERT from the temporary table to the main table
        # ON CONFLICT, update the existing row with the new data.
        update_cols = [f'"{col}" = EXCLUDED."{col}"' for col in df.columns if col not in ['Date', 'Ticker']]

        upsert_query = sql.SQL("""
            INSERT INTO "{target_table}" ({columns})
            SELECT {columns} FROM "{temp_table}"
            ON CONFLICT ("Date", "Ticker") DO UPDATE SET {update_statements};
        """).format(
            target_table=sql.Identifier(config.DB_TABLE_NAME),
            columns=sql.SQL(', ').join(map(sql.Identifier, df.columns)),
            temp_table=sql.Identifier(temp_table_name),
            update_statements=sql.SQL(', ').join(sql.SQL(s) for s in update_cols)
        )

        cur.execute(upsert_query)
        print("Upsert operation completed.")

        # 4. Drop the temporary table
        cur.execute(f'DROP TABLE "{temp_table_name}";')

# ============================================================================== 
# MAIN ORCHESTRATOR 
# ============================================================================== 
def run_the_pipeline(args):
    """Main ETL pipeline logic."""
    stock_df = fetch_stock_data(config.TICKERS)

    if config.FINLIGHT_API_KEY == "YOUR_API_KEY_HERE":
        print("WARNING: Finlight API key is not set in config.py. Skipping news fetching.")
        news_df = pd.DataFrame()
    else:
        api = FinlightApi(ApiConfig(api_key=config.FINLIGHT_API_KEY))
        news_df = fetch_all_news(api, config.TICKERS)

    if stock_df is not None:
        final_df = transform_data(stock_df, news_df)

        print(f"\n--- Saving processed data to {config.PROCESSED_CSV_PATH} for inspection ---")
        os.makedirs(os.path.dirname(config.PROCESSED_CSV_PATH), exist_ok=True)
        final_df.to_csv(config.PROCESSED_CSV_PATH, index=False)
        print("CSV saved successfully.")

        if not args.skip_db:
            conn = None
            try:
                conn = get_db_connection()
                if conn:
                    setup_database_table(conn)
                    upsert_data_to_db(conn, final_df)
                    conn.commit()
                    print("\nDatabase operations completed successfully.")
            except Exception as e:
                print(f"\nAn error occurred during database operations: {e}")
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    conn.close()
                    print("Database connection closed.")
        else:
            print("\n--skip-db flag was set. Skipping all database operations. --")
    else:
        print("\nPipeline halted due to errors in data extraction.")

def get_pipeline_args():
    """Sets up and parses command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="Run the full data pipeline for stock and news sentiment analysis.")
    parser.add_argument("--skip-db", action="store_true", help="Skip all database operations.")
    return parser

if __name__ == "__main__":
    print("===== Starting End-to-End Data Pipeline =====")
    parser = get_pipeline_args()
    args = parser.parse_args()
    run_the_pipeline(args)
    print("\n===== End-to-End Data Pipeline Finished =====")

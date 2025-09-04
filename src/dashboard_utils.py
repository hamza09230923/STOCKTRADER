import streamlit as st
import pandas as pd
import psycopg2
import yfinance as yf

# Import configuration
try:
    import config
except ImportError:
    st.error("CRITICAL: config.py not found. Please create it from config.py.example.")
    st.stop()

# --- Database Connection ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME, user=config.DB_USER, password=config.DB_PASSWORD,
            host=config.DB_HOST, port=config.DB_PORT, connect_timeout=3
        )
        return conn
    except psycopg2.OperationalError:
        return None

# --- Data Loading ---
@st.cache_data(ttl=300)
def load_data():
    """Loads the processed stock and sentiment data from the database or a CSV."""
    conn = get_db_connection()
    if conn:
        try:
            query = f'SELECT * FROM "{config.DB_TABLE_NAME}" ORDER BY "Date" ASC'
            df = pd.read_sql(query, conn)
            df['Date'] = pd.to_datetime(df['Date'])
            conn.close()
            return df, "db"
        except Exception as e:
            st.warning(f"Failed to load data from DB: {e}. Falling back to CSV.")
            conn.close()

    # Fallback to CSV
    try:
        df = pd.read_csv(config.PROCESSED_CSV_PATH, parse_dates=['Date'])
        return df, "csv"
    except FileNotFoundError:
        st.error(f"Fatal Error: Could not connect to DB and fallback file '{config.PROCESSED_CSV_PATH}' not found.")
        return None, "error"

# --- Ticker Management ---
@st.cache_data(ttl=60) # Cache for 1 minute
def get_tracked_tickers():
    """Fetches the list of tickers from the tracked_tickers table."""
    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed. Cannot fetch tickers.")
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM tracked_tickers ORDER BY ticker ASC;")
        tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers

def validate_ticker(ticker):
    """Checks if a ticker is valid using yfinance."""
    if not ticker or not isinstance(ticker, str):
        return False, "Ticker cannot be empty."

    # Check if the ticker already exists
    if ticker in get_tracked_tickers():
        return False, f"Ticker '{ticker}' is already being tracked."

    try:
        stock = yf.Ticker(ticker)
        if not stock.history(period="1d").empty:
            return True, f"Ticker '{ticker}' is valid."
        else:
            return False, f"Could not find any data for ticker '{ticker}'. It may be invalid or delisted."
    except Exception as e:
        return False, f"An error occurred while validating ticker '{ticker}': {e}"

def add_ticker(ticker):
    """Adds a new ticker to the tracked_tickers table."""
    is_valid, message = validate_ticker(ticker)
    if not is_valid:
        st.error(message)
        return False

    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed. Cannot add ticker.")
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO tracked_tickers (ticker) VALUES (%s) ON CONFLICT (ticker) DO NOTHING;", (ticker,))
        conn.commit()
        st.success(f"Ticker '{ticker}' added successfully. Data will be fetched on the next pipeline run.")
        # Bust the cache for get_tracked_tickers
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"An error occurred while adding the ticker: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def remove_ticker(ticker):
    """Removes a ticker from the tracked_tickers table."""
    conn = get_db_connection()
    if not conn:
        st.error("Database connection failed. Cannot remove ticker.")
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tracked_tickers WHERE ticker = %s;", (ticker,))
        conn.commit()
        st.success(f"Ticker '{ticker}' removed successfully.")
        # Bust the cache for get_tracked_tickers
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"An error occurred while removing the ticker: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

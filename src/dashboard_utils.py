import streamlit as st
import pandas as pd
import psycopg2

# Import configuration
try:
    import config
except ImportError:
    st.error("CRITICAL: config.py not found. Please create it from config.py.example.")
    st.stop()

@st.cache_data(ttl=300) # Cache for 5 minutes
def load_data():
    """
    Loads the processed stock and sentiment data, trying the database first
    and falling back to a local CSV.
    """
    try:
        conn = psycopg2.connect(
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            connect_timeout=3
        )
        query = f'SELECT * FROM "{config.DB_TABLE_NAME}" ORDER BY "Date" ASC'
        df = pd.read_sql(query, conn)
        df['Date'] = pd.to_datetime(df['Date'])
        conn.close()
        return df, "db"
    except psycopg2.OperationalError:
        st.warning("Could not connect to the database. Falling back to local CSV file.")
        try:
            df = pd.read_csv(config.PROCESSED_CSV_PATH, parse_dates=['Date'])
            return df, "csv"
        except FileNotFoundError:
            st.error(f"Fatal Error: Could not connect to the database AND the fallback file '{config.PROCESSED_CSV_PATH}' was not found.")
            return None, "error"

def add_ticker_to_config(ticker):
    """
    Adds a new ticker to the TICKERS list in the config.py file.
    """
    import re
    with open('config.py', 'r') as f:
        content = f.read()

    tickers_list_str_match = re.search(r'TICKERS = \[(.*?)\]', content)
    if tickers_list_str_match:
        tickers_list_str = tickers_list_str_match.group(1)
        tickers = [t.strip().strip("'\"") for t in tickers_list_str.split(',') if t.strip()]
    else:
        tickers = []

    if ticker.upper() not in tickers:
        tickers.append(ticker.upper())
        new_tickers_list_str = ', '.join([f'"{t}"' for t in tickers])

        if tickers_list_str_match:
            new_content = re.sub(r'TICKERS = \[.*?\]', f'TICKERS = [{new_tickers_list_str}]', content)
        else:
            # If TICKERS list not found, add it to the end of the file.
            new_content = content + f'\nTICKERS = [{new_tickers_list_str}]\n'

        with open('config.py', 'w') as f:
            f.write(new_content)

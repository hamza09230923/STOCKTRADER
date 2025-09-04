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

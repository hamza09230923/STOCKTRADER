# Stock Market + News Sentiment Tracker

A comprehensive, end-to-end data pipeline and visualization dashboard that tracks stock prices and financial news sentiment. This project aims to explore the correlation between market sentiment and stock performance.

## 🎯 Features

*   **Automated Data Pipeline (`run_pipeline.py`):**
    *   Fetches 5 years of historical stock data from Yahoo Finance.
    *   Fetches recent financial news from the Finlight.me API.
    *   Performs sentiment analysis on news headlines using both VADER and FinBERT.
    *   Transforms and merges the data into a single, clean dataset.
    *   Loads the final data into a PostgreSQL database.

*   **Interactive Dashboard (`dashboard.py`):**
    *   A web-based dashboard built with Streamlit and Plotly.
    *   Visualizes stock price, trading volume, and sentiment scores over time.
    *   Provides interactive controls to select stocks and date ranges.
    *   Features a professional and clean user interface.

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**
    It is highly recommended to use a Python virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## 🚀 How to Run

There are two primary ways to run the project and access the data:

### 1. Running the Data Pipeline

The `run_pipeline.py` script executes the entire ETL (Extract, Transform, Load) process. It will generate a `processed_data.csv` file in the `data/` directory and, if configured, load the data into a PostgreSQL database.

**Prerequisites:**
*   You need a **Finlight API Key**. You can get a free one from [Finlight.me](https://finlight.me/).
*   For database loading, you need a running **PostgreSQL server**.

**Execution:**

Run the script from your terminal. You must provide your Finlight API key. Database credentials can also be provided as arguments.

```bash
python run_pipeline.py --api-key "your_finlight_api_key" --db-name "stock_sentiment" --db-user "your_user" --db-password "your_password"
```

**Running without a Database:**
If you don't have a database set up, you can run the data processing part of the pipeline and just generate the CSV file. Use the `--skip-db` flag:

```bash
python run_pipeline.py --api-key "your_finlight_api_key" --skip-db
```
This will create `data/processed_data.csv`, which the dashboard can use as a fallback.

### 2. Running the Interactive Dashboard

The `dashboard.py` script launches a web application to visualize the data.

**Prerequisites:**
*   You must have the data available. Either run the pipeline first (see above) to create `data/processed_data.csv`, or have a populated PostgreSQL database that the dashboard can connect to.

**Execution:**

Run the following command in your terminal:

```bash
streamlit run dashboard.py
```

The dashboard will automatically try to connect to the PostgreSQL database using default credentials (`user=postgres`, `db=stock_sentiment`, etc.). If it fails, it will fall back to using the `data/processed_data.csv` file.

## 📁 Project Structure

```
.
├── .streamlit/
│   └── config.toml     # Streamlit theme configuration
├── data/               # Output directory for CSV files (created on run)
├── src/
│   └── sentiment_analysis.py # Core module for VADER and FinBERT analysis
├── dashboard.py        # Main application file for the Streamlit dashboard
├── README.md           # This file
├── requirements.txt    # Project dependencies
└── run_pipeline.py     # Master script for the entire ETL data pipeline
```
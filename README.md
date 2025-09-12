# Stock Market + News Sentiment Tracker

A comprehensive, end-to-end data pipeline and multi-page visualization dashboard that tracks stock prices and financial news sentiment. This project aims to explore the correlation between market sentiment and stock performance, and includes a machine learning pipeline to predict stock price movements.

![Dashboard Screenshot](https://via.placeholder.com/800x400.png?text=Dashboard+Screenshot+Here)

---

## ✨ Tech Stack

*   **Data Pipeline:** Python, Pandas, Yahoo Finance API (yfinance), Finlight.me API
*   **Sentiment Analysis:** VADER, FinBERT (Hugging Face Transformers)
*   **Database:** PostgreSQL
*   **Dashboard:** Streamlit, Plotly
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Deployment:** Heroku, Streamlit Community Cloud

---

## 🎯 Features

*   **Automated Data Pipeline (`run_pipeline.py`):**
    *   Fetches 5 years of historical stock data from Yahoo Finance.
    *   Fetches recent financial news from the Finlight.me API.
    *   Performs sentiment analysis on news headlines using both VADER and FinBERT.
    *   Transforms and merges the data into a single, clean dataset.
    *   Loads the final data into a PostgreSQL database using a robust upsert mechanism.

*   **Interactive Multi-Page Dashboard:**
    *   **Home:** A welcoming landing page that provides an overview of the app's features.
    *   **Dashboard:** Visualizes stock price, trading volume, and sentiment scores over time with interactive charts. Includes ML-powered price movement predictions.
    *   **Data Explorer:** Allows users to view, sort, and filter the raw, processed data.
    *   **About:** Provides information about the project, data sources, and technologies used.

*   **ML Model Training Pipeline (`scripts/train_models.py`):**
    *   Performs feature engineering to create a dataset for predicting stock price movements.
    *   Trains both a Logistic Regression and an XGBoost model.
    *   Saves the trained models for future use.

*   **Automated Scheduler (`scheduler.py`):**
    *   Runs the data pipeline automatically at a configurable interval to keep the data fresh.

---

## 📁 Project Structure

```
.
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── models/                 # Saved machine learning models
├── notebooks/              # Jupyter notebooks for analysis
├── pages/                  # Streamlit pages
│   ├── 1_Dashboard.py
│   ├── 2_Data_Explorer.py
│   └── 3_About.py
├── scripts/                # Helper and training scripts
├── src/                    # Source code for dashboard and sentiment analysis
├── Home.py                 # Main application file for Streamlit
├── requirements.txt        # Project dependencies
├── run_pipeline.py         # Master script for the ETL data pipeline
└── ...
```

---

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    *   The `psycopg2` library requires PostgreSQL client libraries. On Debian/Ubuntu, you can install them with:
        ```bash
        sudo apt-get update && sudo apt-get install libpq-dev
        ```
    *   Install the Python packages:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Configure the application:**
    This project uses a `config.py` file for configuration, but it is highly recommended to use environment variables, especially for production and deployment.
    *   Create a `.env` file in the root directory (this is ignored by `.gitignore`).
    *   Add your secrets to the `.env` file. Refer to `config.py.example` for the required variable names (`FINLIGHT_API_KEY`, `DB_NAME`, etc.). The application will automatically load these variables.

---

## 🚀 How to Run

### 1. Running the Data Pipeline Manually
The `run_pipeline.py` script executes the entire ETL process. It will generate a `processed_data.csv` file and load the data into the configured PostgreSQL database.
```bash
python run_pipeline.py
```
To skip database operations and only generate the CSV:
```bash
python run_pipeline.py --skip-db
```

### 2. Running the Interactive Dashboard
The `Home.py` script launches the web application. Make sure you have run the pipeline at least once or have a `processed_data.csv` file available.
```bash
streamlit run Home.py
```

### 3. Running the Pipeline on a Schedule
The `scheduler.py` script runs the data pipeline at a regular interval.
```bash
python scheduler.py --interval-hours 4
```

### 4. Training the ML Models
The `scripts/train_models.py` script runs the full ML pipeline. You must provide your Finlight API key.
```bash
python scripts/train_models.py --api-key "your_finlight_api_key"
```

---

## ☁️ Deployment

This application is designed for cloud deployment. Below are instructions for Streamlit Community Cloud and Heroku.

### Prerequisites
*   A public GitHub repository for your project.
*   A cloud-hosted PostgreSQL database (e.g., from AWS RDS, Google Cloud SQL, or Heroku).
*   Accounts with your chosen hosting provider(s).

### Environment Variables
Before deploying, set the following environment variables on your hosting platform.
*   `FINLIGHT_API_KEY`
*   `DB_NAME`
*   `DB_USER`
*   `DB_PASSWORD`
*   `DB_HOST`
*   `DB_PORT`

### Option 1: Deploying to Streamlit Community Cloud (Dashboard Only)
This is the easiest way to deploy the dashboard. You will need to run the data pipeline separately to keep the data fresh.
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click "New app" and connect your GitHub account.
3.  Select the repository, branch, and set the "Main file path" to `Home.py`.
4.  In "Advanced settings," add your environment variables as secrets.
5.  Click "Deploy!".

### Option 2: Deploying to Heroku (Dashboard and Scheduler)
Heroku can run both the dashboard and the scheduled pipeline worker.
1.  Create a `Procfile` in your root directory with the following content:
    ```
    web: sh setup.sh && streamlit run Home.py
    worker: python scheduler.py
    ```
2.  Create a `packages.txt` file and add `libpq-dev` to it. Heroku will use this to install system-level dependencies.
3.  In your Heroku app's "Settings" tab, add the required environment variables in the "Config Vars" section.
4.  In the "Resources" tab, enable the `worker` dyno to run the scheduler. This may incur costs.
5.  **Alternative:** Use the "Heroku Scheduler" add-on to run `python run_pipeline.py` on a schedule, which can be more cost-effective than a persistent worker dyno.
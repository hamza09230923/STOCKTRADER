# Stock Market + News Sentiment Tracker

A comprehensive, end-to-end data pipeline and visualization dashboard that tracks stock prices and financial news sentiment. This project aims to explore the correlation between market sentiment and stock performance. It also includes a machine learning pipeline to predict stock price movements based on historical data and sentiment scores.

## 🎯 Features

*   **Automated Data Pipeline (`run_pipeline.py`):**
    *   Fetches 5 years of historical stock data from Yahoo Finance.
    *   Fetches recent financial news from the Finlight.me API.
    *   Performs sentiment analysis on news headlines using both VADER and FinBERT.
    *   Transforms and merges the data into a single, clean dataset.
    *   Loads the final data into a PostgreSQL database using a robust upsert mechanism.

*   **Interactive Dashboard (`dashboard.py`):**
    *   A web-based dashboard built with Streamlit and Plotly.
    *   Visualizes stock price, trading volume, and sentiment scores over time.
    *   Provides interactive controls to select stocks and date ranges.
    *   Features a professional and clean user interface.

*   **ML Model Training Pipeline (`scripts/train_models.py`):**
    *   Performs feature engineering to create a dataset for predicting stock price movements.
    *   Trains both a Logistic Regression and an XGBoost model.
    *   Saves the trained models for future use.

*   **Automated Scheduler (`scheduler.py`):**
    *   Runs the data pipeline automatically at a configurable interval.

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

3.  **Configure the application:**
    *   Rename the `config.py.example` file to `config.py`.
    *   Open `config.py` and add your Finlight API key and database credentials. It is highly recommended to use environment variables for these secrets.

## 🚀 How to Run

### 1. Running the Data Pipeline Manually

The `run_pipeline.py` script executes the entire ETL (Extract, Transform, Load) process. It will generate a `processed_data.csv` file in the `data/` directory and load the data into the configured PostgreSQL database.

```bash
python run_pipeline.py
```
You can also run it without database operations:
```bash
python run_pipeline.py --skip-db
```

### 2. Running the Interactive Dashboard

The `dashboard.py` script launches the web application. Make sure you have run the pipeline at least once to generate the necessary data.

```bash
streamlit run dashboard.py
```

### 3. Running the Pipeline on a Schedule

The `scheduler.py` script runs the data pipeline at a regular interval.

```bash
python scheduler.py --interval-hours 4
```

### 4. Training the ML Models

The `scripts/train_models.py` script runs the full ML pipeline, from data generation to model training. You must provide your Finlight API key.

```bash
python scripts/train_models.py --api-key "your_finlight_api_key"
```

## 📁 Project Structure

```
.
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── config.py               # Main configuration file (ignored by git)
├── config.py.example       # Example configuration file
├── data/                   # Output directory for CSV files (created on run)
│   └── .gitkeep
├── models/                 # Saved machine learning models
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   └── xgboost_model.pkl
├── notebooks/              # Jupyter notebooks for analysis
│   └── 01-EDA-and-Feature-Engineering.ipynb
├── scripts/
│   ├── archive/            # Old/unused scripts
│   ├── train_models.py     # Script to train predictive models
│   └── ...
├── src/
│   └── sentiment_analysis.py # Core module for VADER and FinBERT analysis
├── dashboard.py            # Main application file for the Streamlit dashboard
├── README.md               # This file
├── requirements.txt        # Project dependencies
├── run_pipeline.py         # Master script for the ETL data pipeline
└── scheduler.py            # Script to run the pipeline on a schedule
```

## 部署

This application is designed to be deployed to a cloud platform. Below are instructions for two popular options: Streamlit Community Cloud (for the dashboard) and Heroku (for both the dashboard and the scheduled data pipeline).

### Prerequisites

1.  **GitHub Repository:** Your project must be in a public GitHub repository.
2.  **Cloud-Hosted PostgreSQL Database:** For a live application, you need a database that can be accessed from the internet. Services like [Amazon RDS](https://aws.amazon.com/rds/), [Google Cloud SQL](https://cloud.google.com/sql), or [Heroku Postgres](https://www.heroku.com/postgres) are good options.
3.  **Platform Accounts:** You will need an account with [Streamlit Community Cloud](https://streamlit.io/cloud) and/or [Heroku](https://www.heroku.com/).

### Environment Variables

Before deploying, you must set the following environment variables on your chosen hosting platform. These correspond to the settings in `config.py`.

*   `FINLIGHT_API_KEY`: Your API key for the Finlight.me service.
*   `DB_NAME`: The name of your PostgreSQL database.
*   `DB_USER`: The username for your database.
*   `DB_PASSWORD`: The password for your database.
*   `DB_HOST`: The hostname or IP address of your database server.
*   `DB_PORT`: The port for your database (usually `5432`).

### Option 1: Deploying to Streamlit Community Cloud (Dashboard Only)

Streamlit Community Cloud is the easiest way to deploy the dashboard. Note that this will only deploy the user-facing application, not the data pipeline scheduler. You will need to run the pipeline separately to keep the data fresh.

1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click "New app" and connect your GitHub account.
3.  Select the repository and branch for your project.
4.  Set the "Main file path" to `Home.py`.
5.  In the "Advanced settings" section, add all the required environment variables (secrets).
6.  Click "Deploy!".

### Option 2: Deploying to Heroku (Dashboard and Scheduler)

Heroku allows you to deploy both the dashboard and the scheduled pipeline.

#### Deploying the Dashboard (Web Dyno)

1.  Create a new app on Heroku.
2.  Connect your GitHub repository to the Heroku app and enable automatic deploys if desired.
3.  In the app's "Settings" tab, go to the "Config Vars" section and add all the environment variables listed above.
4.  Heroku will automatically detect the `Procfile` and start the `web` process to run your dashboard.

#### Deploying the Scheduler (Worker Dyno)

The `scheduler.py` script needs to be run continuously as a worker process.

1.  First, you need to modify the `Procfile` to include a worker process. Change the `Procfile` to:
    ```
    web: sh setup.sh && streamlit run Home.py
    worker: python scheduler.py
    ```
2.  Commit and push this change to your repository.
3.  In your Heroku app's "Resources" tab, you should see the `web` and `worker` processes.
4.  Enable the `worker` dyno. This will run the scheduler continuously. Note that this may incur costs depending on your Heroku plan.

**Alternative for Scheduler (Heroku Scheduler):**

If you don't want a continuously running worker, you can use the Heroku Scheduler add-on to run the pipeline periodically.

1.  In your Heroku app's "Resources" tab, find the "Add-ons" section and add "Heroku Scheduler".
2.  Open the Heroku Scheduler dashboard.
3.  Create a new job. Set the schedule (e.g., "Every hour") and for the command, enter: `python run_pipeline.py`.
4.  This will run the data pipeline on your chosen schedule without needing a persistent worker dyno.
# Stock Market + News Sentiment Tracker

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, end-to-end data pipeline and multi-page visualization dashboard that tracks stock prices and financial news sentiment. This project aims to explore the correlation between market sentiment and stock performance, and includes a machine learning pipeline to predict stock price movements.

---
---

## 🎯 Features

*   **Automated Data Pipeline (`run_pipeline.py`):**
    *   Fetches 5 years of historical stock data from Yahoo Finance.
    *   Fetches recent financial news and social media sentiment from the Finlight.me API, Reddit, and Twitter.
    *   Performs sentiment analysis on news headlines and posts using both VADER and FinBERT.
    *   Transforms and merges the data into a single, clean dataset.
    *   Loads the final data into a PostgreSQL database using a robust upsert mechanism.

*   **Interactive Multi-Page Dashboard:**
    *   **Home:** A welcoming landing page that provides an overview of the app's features.
    *   **Dashboard:** Visualizes stock price, trading volume, and sentiment scores over time with interactive charts. Includes ML-powered price movement predictions.
    *   **Data Explorer:** Allows users to view, sort, and filter the raw, processed data.
    *   **About:** Provides information about the project, data sources, and technologies used.

*   **Sentiment-Based Backtesting Engine:**
    *   A new page to backtest trading strategies based on sentiment scores.
    *   Allows users to select a stock, date range, and tune strategy parameters.
    *   Displays detailed performance metrics and statistics for the backtest.

*   **ML Model Training Pipeline (`scripts/train_models.py`):**
    *   Performs feature engineering to create a dataset for predicting stock price movements.
    *   Trains both a Logistic Regression and an XGBoost model.
    *   Saves the trained models for future use.

*   **Automated Scheduler (`scheduler.py`):**
    *   Runs the data pipeline automatically at a configurable interval to keep the data fresh.

---

## ✨ Tech Stack

*   **Data Pipeline:** Python, Pandas, Yahoo Finance API (yfinance), Finlight.me API, Reddit API, Twitter API
*   **Sentiment Analysis:** VADER, FinBERT (Hugging Face Transformers)
*   **Database:** PostgreSQL
*   **Dashboard:** Streamlit, Plotly
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Backtesting:** backtesting.py
*   **Deployment:** Heroku, Streamlit Community Cloud

---

## 📁 Project Structure

<details>
<summary>Click to view the project structure</summary>

```
.
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── models/                 # Saved machine learning models
├── notebooks/              # Jupyter notebooks for analysis
├── pages/                  # Streamlit pages
│   ├── 1_Dashboard.py
│   ├── 2_Data_Explorer.py
│   ├── 3_About.py
│   └── 4_Backtester.py
├── scripts/                # Helper and training scripts
├── src/                    # Source code for dashboard and sentiment analysis
├── Home.py                 # Main application file for Streamlit
├── requirements.txt        # Project dependencies
├── run_pipeline.py         # Master script for the ETL data pipeline
└── ...
```

</details>

---

## ⚙️ Setup and Configuration

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
The `psycopg2` library requires PostgreSQL client libraries. On Debian/Ubuntu, you can install them with:
```bash
sudo apt-get update && sudo apt-get install libpq-dev
```
Then, install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
This project uses a `.env` file to manage secrets and configuration variables.
1.  **Create `.env` file:** Make a copy of `config.py.example` and rename it to `.env`.
2.  **Add your credentials:** Fill in the required API keys and database credentials in the `.env` file. The application will automatically load these variables.

---

## 📊 Data Sources & API Keys

This project aggregates data from multiple sources. You will need to acquire free API keys for the following services:

| Service       | Data Provided        | Instructions                                                                                                                                                                                                                                                                                       |
|---------------|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Yahoo Finance** | Historical stock data (OHLCV) | No API key needed. Data is accessed via the `yfinance` library.                                                                                                                                                                                                                           |
| **Finlight.me**   | Financial news articles | Get a free API key at [finlight.me](https://finlight.me/).                                                                                                                                                                                                                                         |
| **Reddit**        | Social media sentiment | Go to [Reddit's App Preferences](https://www.reddit.com/prefs/apps), create a "script" app, and get your `Client ID` and `Client Secret`.                                                                                                                                                        |
| **Twitter (X)**   | Social media sentiment | Apply for a [Twitter Developer Account](https://developer.twitter.com/en/apply-for-access), create an App in your developer portal, and generate a `Bearer Token` for the v2 API. |

---

## 🚀 How to Run

### Running the Data Pipeline
The `run_pipeline.py` script executes the entire ETL process. It fetches data, performs sentiment analysis, and loads it into your database.

**To run the pipeline and load data into the database:**
```bash
python run_pipeline.py
```

**To run the pipeline and only generate a local CSV file (skips database operations):**
```bash
python run_pipeline.py --skip-db
```

### Running the Interactive Dashboard
The `Home.py` script launches the Streamlit web application. Make sure you have run the pipeline at least once or have a `processed_data.csv` file available.
```bash
streamlit run Home.py
```

### Running the Pipeline on a Schedule
The `scheduler.py` script runs the data pipeline at a regular interval to keep the data fresh.
```bash
python scheduler.py --interval-hours 4
```

### Training the ML Models
The `scripts/train_models.py` script runs the full ML training pipeline.
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

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or want to fix a bug, please feel free to open an issue or submit a pull request.

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature-name`
3.  **Make your changes** and commit them with a clear message.
4.  **Push your changes** to your fork.
5.  **Submit a pull request** to the main repository.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
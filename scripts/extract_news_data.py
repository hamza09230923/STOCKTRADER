import os
import pandas as pd
from datetime import datetime, timedelta
import argparse
from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams

# --- Configuration ---
TICKERS = ["AAPL", "TSLA", "NVDA", "JPM", "AMZN"]
OUTPUT_FILEPATH = "data/news_articles.csv"

def fetch_all_news(api, tickers):
    """
    Fetches news articles for a list of tickers for the past 30 days.
    """
    print("Fetching news articles for the last 30 days...")
    all_articles = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
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
                    article_dict = {
                        'ticker': ticker,
                        'publish_date': article.publishDate,
                        'title': article.title,
                        'summary': article.summary,
                        'source': article.source,
                        'sentiment': article.sentiment,
                        'url': article.link
                    }
                    all_articles.append(article_dict)
                print(f"Found {len(response.articles)} articles for {ticker}.")
            else:
                print(f"No articles found for {ticker}.")

        except Exception as e:
            print(f"An error occurred while fetching news for {ticker}: {e}")

    if not all_articles:
        print("No articles were fetched in total.")
        return None

    return pd.DataFrame(all_articles)

if __name__ == "__main__":
    print("--- Starting News Data Extraction ---")

    parser = argparse.ArgumentParser(description="Fetch financial news using the Finlight API.")
    parser.add_argument("--api-key", required=True, help="Your Finlight API key.")
    args = parser.parse_args()

    # Correctly initialize the API client
    config = ApiConfig(api_key=args.api_key)
    api = FinlightApi(config)

    news_df = fetch_all_news(api, TICKERS)

    if news_df is not None and not news_df.empty:
        dir_path = os.path.dirname(OUTPUT_FILEPATH)
        os.makedirs(dir_path, exist_ok=True)
        print(f"\nSaving {len(news_df)} articles to {OUTPUT_FILEPATH}...")
        news_df.to_csv(OUTPUT_FILEPATH, index=False)
        print("News data saved successfully.")

    print("--- News Data Extraction Finished ---")

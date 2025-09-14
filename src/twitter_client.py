import tweepy
import pandas as pd
from datetime import datetime
import config

def fetch_twitter_data(tickers, limit_per_ticker=100):
    """
    Fetches recent tweets from Twitter for a list of tickers.

    Args:
        tickers (list): A list of stock tickers to search for.
        limit_per_ticker (int): The maximum number of tweets to fetch for each ticker.

    Returns:
        pandas.DataFrame: A DataFrame containing the fetched data.
    """
    if not config.TWITTER_BEARER_TOKEN or config.TWITTER_BEARER_TOKEN == "YOUR_TWITTER_BEARER_TOKEN":
        print("WARNING: Twitter Bearer Token is not configured. Skipping Twitter data fetch.")
        return pd.DataFrame()

    try:
        client = tweepy.Client(bearer_token=config.TWITTER_BEARER_TOKEN)
        print("Successfully authenticated with Twitter API.")
    except Exception as e:
        print(f"CRITICAL: Failed to authenticate with Twitter API: {e}")
        return pd.DataFrame()

    all_tweets = []
    for ticker in tickers:
        try:
            # Construct the query. Search for the ticker symbol, ensuring it's not part of a cashtag
            # and that it's in English. We also exclude retweets.
            query = f'"{ticker}" lang:en -is:retweet'

            print(f"Fetching tweets for {ticker}...")
            response = client.search_recent_tweets(
                query,
                max_results=limit_per_ticker,
                tweet_fields=["created_at", "public_metrics", "author_id"]
            )

            if response.data:
                for tweet in response.data:
                    all_tweets.append({
                        'ticker': ticker.upper(),
                        'publish_date': tweet.created_at,
                        'title': tweet.text,
                        'summary': tweet.text,
                        'source': 'Twitter',
                        'sentiment': None,  # To be filled later
                        'url': f"https://twitter.com/{tweet.author_id}/status/{tweet.id}"
                    })
        except Exception as e:
            print(f"WARNING: Could not fetch tweets for {ticker}: {e}")
            continue

    if not all_tweets:
        print("No relevant tweets found on Twitter.")
        return pd.DataFrame()

    print(f"Fetched {len(all_tweets)} relevant tweets from Twitter.")
    return pd.DataFrame(all_tweets)

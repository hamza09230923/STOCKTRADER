import praw
import pandas as pd
from datetime import datetime
import config

def fetch_reddit_data(tickers, subreddits, limit=100):
    """
    Fetches posts from specified subreddits for a list of tickers.

    Args:
        tickers (list): A list of stock tickers to search for.
        subreddits (list): A list of subreddits to search in.
        limit (int): The maximum number of posts to fetch from each subreddit.

    Returns:
        pandas.DataFrame: A DataFrame containing the fetched data, with columns:
                          'ticker', 'publish_date', 'title', 'summary', 'source',
                          'sentiment', and 'url'.
    """
    if not all([config.REDDIT_CLIENT_ID, config.REDDIT_CLIENT_SECRET, config.REDDIT_USER_AGENT]):
        print("WARNING: Reddit API credentials are not fully configured. Skipping Reddit data fetch.")
        return pd.DataFrame()

    try:
        reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
        )
        print("Successfully authenticated with Reddit API.")
    except Exception as e:
        print(f"CRITICAL: Failed to authenticate with Reddit API: {e}")
        return pd.DataFrame()

    all_posts = []
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            print(f"Fetching posts from r/{subreddit_name}...")
            for post in subreddit.hot(limit=limit):
                for ticker in tickers:
                    if ticker.lower() in post.title.lower() or ticker.lower() in post.selftext.lower():
                        all_posts.append({
                            'ticker': ticker.upper(),
                            'publish_date': datetime.fromtimestamp(post.created_utc),
                            'title': post.title,
                            'summary': post.selftext[:200],  # Truncate for summary
                            'source': f"Reddit r/{subreddit_name}",
                            'sentiment': None,  # To be filled later
                            'url': post.url
                        })
        except Exception as e:
            print(f"WARNING: Could not fetch data from r/{subreddit_name}: {e}")
            continue

    if not all_posts:
        print("No relevant posts found on Reddit.")
        return pd.DataFrame()

    print(f"Fetched {len(all_posts)} relevant posts from Reddit.")
    return pd.DataFrame(all_posts)

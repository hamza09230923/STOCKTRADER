from transformers import pipeline
import pandas as pd

# --- Text Summarization Setup ---
# Initialize the summarization pipeline once to avoid reloading the model.
# Using a distilled version for a good balance of performance and size.
print("Initializing Text Summarization pipeline (this may take a moment)...")
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print("Summarization pipeline initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize Summarization pipeline: {e}")
    summarizer = None

def generate_summary(articles_df: pd.DataFrame) -> str:
    """
    Generates a summary from a DataFrame of news articles.

    Args:
        articles_df (pd.DataFrame): A DataFrame with a 'title' column containing
                                    the news headlines/articles to summarize.

    Returns:
        str: A single, concise summary of all the articles. Returns an error
             message if the summarizer is not available or if there are no articles.
    """
    if summarizer is None:
        return "Error: The summarization model is not available."

    if articles_df.empty or 'title' not in articles_df.columns:
        return "No articles provided to summarize."

    # Concatenate all article titles/texts into a single block of text.
    # Using a separator to help the model distinguish between articles.
    full_text = ". ".join(articles_df['title'].astype(str).tolist())

    if not full_text.strip():
        return "The provided articles are empty."

    # The model has a max input length. We must truncate the text to avoid errors.
    # The default max length for this model is 1024 tokens.
    max_length = 1024
    if len(full_text) > max_length * 4: # A rough heuristic for character to token conversion
        print(f"Warning: Input text is very long, truncating to ~{max_length} tokens for summarization.")
        full_text = full_text[:max_length * 4]

    try:
        # Generate the summary. We specify min and max length for the output.
        summary_result = summarizer(full_text, max_length=150, min_length=30, do_sample=False)
        return summary_result[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error: Could not generate summary for the provided text."

from bertopic import BERTopic

# --- Topic Modeling Setup ---
# Using a pre-trained sentence transformer model for embeddings.
# This model is lightweight and effective for general-purpose text.
embedding_model = "all-MiniLM-L6-v2"
topic_model = BERTopic(embedding_model=embedding_model, verbose=False)

def perform_topic_modeling(articles_df: pd.DataFrame):
    """
    Performs topic modeling on a list of news headlines.

    Args:
        articles_df (pd.DataFrame): A DataFrame with a 'title' column containing
                                    the news headlines to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing the identified topics, their sizes,
                      and their top representative words. Returns None if an
                      error occurs or if there are not enough articles.
    """
    if articles_df.empty or 'title' not in articles_df.columns:
        print("Warning: No articles provided for topic modeling.")
        return None

    # BERTopic requires a list of strings.
    headlines = articles_df['title'].astype(str).tolist()

    # Topic modeling requires a sufficient number of documents to be effective.
    if len(headlines) < 10:
        print("Warning: Not enough documents to perform meaningful topic modeling.")
        return None

    try:
        # Fit the model and transform the headlines to get topic assignments.
        topics, _ = topic_model.fit_transform(headlines)

        # Get the topic info as a DataFrame.
        topic_info = topic_model.get_topic_info()
        return topic_info

    except Exception as e:
        print(f"Error during topic modeling: {e}")
        return None

# --- Aspect-Based Sentiment Analysis (ABSA) Setup ---
print("Initializing ABSA pipeline (this may take a moment)...")
try:
    absa_classifier = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
    print("ABSA pipeline initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize ABSA pipeline: {e}")
    absa_classifier = None

def analyze_aspect_sentiment(articles_df: pd.DataFrame, aspects: list):
    """
    Performs aspect-based sentiment analysis on news headlines.

    Args:
        articles_df (pd.DataFrame): A DataFrame with a 'title' column.
        aspects (list): A list of aspects (e.g., stock tickers) to look for.

    Returns:
        pd.DataFrame: A DataFrame with columns ['title', 'aspect', 'sentiment'].
                      Returns None if the model is not available or no aspects are found.
    """
    if absa_classifier is None:
        print("Error: The ABSA model is not available.")
        return None

    if articles_df.empty or 'title' not in articles_df.columns:
        print("Warning: No articles provided for ABSA.")
        return None

    results = []
    headlines = articles_df['title'].astype(str).tolist()

    for text in headlines:
        found_aspects = [aspect for aspect in aspects if aspect.lower() in text.lower()]
        if not found_aspects:
            continue

        # Analyze for the first found aspect
        aspect = found_aspects[0]
        try:
            result = absa_classifier(text, text_pair=aspect)
            # The result is a list of dicts, e.g., [{'label': 'Positive', 'score': 0.99}]
            # We just need the label.
            sentiment = result[0]['label']
            results.append({'title': text, 'aspect': aspect, 'sentiment': sentiment})
        except Exception as e:
            print(f"Error during ABSA for text: '{text}' with aspect '{aspect}': {e}")
            continue

    if not results:
        return None

    return pd.DataFrame(results)

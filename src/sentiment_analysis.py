import nltk
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# --- VADER Setup ---
# Ensure the VADER lexicon is downloaded before use.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon for sentiment analysis...")
    nltk.download('vader_lexicon')
    print("Download complete.")

# Initialize VADER sentiment analyzer as a global object for efficiency.
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_vader_sentiment(text):
    """
    Analyzes the sentiment of a given text using VADER.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The compound sentiment score from VADER, ranging from -1 (most negative)
               to +1 (most positive). Returns 0.0 for invalid input.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0  # Return a neutral score for empty or non-string input.

    # Get the polarity scores.
    scores = vader_analyzer.polarity_scores(text)
    # Return the compound score.
    return scores['compound']


# --- FinBERT Setup ---
# Initialize the FinBERT pipeline. This is done once at the module level to avoid
# reloading the heavy model on every function call. This will download the model
# from HuggingFace the first time it's executed.
print("Initializing FinBERT sentiment analysis pipeline (this may take a moment)...")
try:
    finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
    print("FinBERT pipeline initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize FinBERT pipeline: {e}")
    finbert_pipeline = None

def analyze_finbert_sentiment(text):
    """
    Analyzes the sentiment of a given text using the FinBERT model.

    Args:
        text (str): The text to analyze.

    Returns:
        tuple: A tuple containing the sentiment label (e.g., 'positive') and the
               confidence score. Returns (None, None) if analysis fails.
    """
    if finbert_pipeline is None:
        print("FinBERT pipeline is not available. Cannot perform analysis.")
        return None, None

    if not isinstance(text, str) or not text.strip():
        return 'neutral', 1.0  # Assume neutral for empty or non-string input.

    try:
        # The pipeline returns a list containing a dictionary, e.g., [{'label': 'positive', 'score': 0.98}]
        results = finbert_pipeline(text)
        result = results[0]
        return result['label'], result['score']
    except Exception as e:
        # Catch potential errors from the pipeline, e.g., for very long or unusual text.
        print(f"Could not analyze text with FinBERT due to an error: {e}")
        return None, None


# --- Example Usage ---
if __name__ == '__main__':
    print("\n--- Testing Sentiment Analysis Module ---")

    test_data = [
        "Stocks are soaring to new highs after the positive earnings report!",
        "The market crash was devastating for investors and led to huge losses.",
        "The company announced a neutral outlook for the next quarter.",
        "" # Empty string test case
    ]

    results = []

    for text in test_data:
        vader_score = analyze_vader_sentiment(text)
        finbert_label, finbert_score = analyze_finbert_sentiment(text)

        results.append({
            "Text": text[:50] + '...',
            "VADER Score": vader_score,
            "FinBERT Label": finbert_label,
            "FinBERT Score": finbert_score
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string())
    print("\n--- Module Test Finished ---")

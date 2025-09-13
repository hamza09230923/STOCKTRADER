import pandas as pd
from backtesting import Backtest, Strategy

class SentimentStrategy(Strategy):
    """
    A trading strategy that buys or sells based on a sentiment score.
    """
    # --- Strategy Parameters ---
    # These can be optimized by the backtesting framework.
    buy_sentiment_threshold = 0.5
    sell_sentiment_threshold = -0.2

    def init(self):
        """
        Initialize the strategy.
        This is called once at the beginning of the backtest.
        """
        # The `self.data` object gives access to the data passed to the backtest.
        # We assume a 'sentiment_score' column exists in the data.
        self.sentiment = self.I(lambda x: x, self.data.sentiment_score)

    def next(self):
        """
        Define the trading logic for the next data point (e.g., the next day).
        This is called for each data point in the dataset.
        """
        # If we have no position and sentiment is strongly positive, go long.
        if not self.position and self.sentiment[-1] > self.buy_sentiment_threshold:
            self.buy()

        # If we have a position and sentiment turns negative, close the position.
        elif self.position and self.sentiment[-1] < self.sell_sentiment_threshold:
            self.position.close()


def run_backtest(data: pd.DataFrame, strategy: Strategy, cash: int = 10000, commission: float = 0.002):
    """
    Runs a backtest on the given data using the specified strategy.

    Args:
        data (pd.DataFrame): The OHLCV data, must include a 'sentiment_score' column.
        strategy (Strategy): The strategy class to use for the backtest.
        cash (int, optional): The initial cash amount. Defaults to 10000.
        commission (float, optional): The commission rate for each trade. Defaults to 0.002.

    Returns:
        A pandas Series with the backtest statistics.
    """
    if 'sentiment_score' not in data.columns:
        raise ValueError("Input data must contain a 'sentiment_score' column.")

    # The backtesting library expects specific column names.
    # We will rename the columns if they are different.
    # For now, we assume the data is already in the correct format:
    # 'Open', 'High', 'Low', 'Close', 'Volume'

    bt = Backtest(data, strategy, cash=cash, commission=commission)
    stats = bt.run()

    # The bt.plot() function opens a browser window, which is not ideal for Streamlit.
    # We will return the stats object and let the Streamlit page handle the plotting if needed.
    # For now, we return the stats series. A future improvement could be to generate a plot object.

    return stats

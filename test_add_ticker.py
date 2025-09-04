import sys
sys.path.append('src')
from dashboard_utils import add_ticker_to_config

add_ticker_to_config('MSFT')
print("Ticker 'MSFT' added to config.")

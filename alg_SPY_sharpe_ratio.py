import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

# Alpaca API keys
API_KEY = 'PLACEHOLDER_API_KEY'
API_SECRET = 'PLACEHOLDER_SECRET_KEY'
BASE_URL = 'https://paper-api.alpaca.markets'  # Restricting to paper trading environment

# Creating an Alpaca API instance
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Define constants
SYMBOL = "SPY"
RISK_FREE_RATE = 0.0001  # Setting risk-free rate, assume 0.01% daily
WINDOW_LENGTH = 30  # Setting rolling window of 30 days for returns
SHARPE_THRESHOLD = 1.1  # Sharpe Ratio threshold

# Function to calculate the Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    avg_return = np.mean(returns)
    volatility = np.std(returns)
    sharpe_ratio = (avg_return - risk_free_rate) / volatility
    return sharpe_ratio

# Function to get historical SPY data from Alpaca
def get_historical_data(symbol, window_length):
    end_time = pd.Timestamp.now(tz='America/New_York')
    start_time = end_time - timedelta(days=window_length + 10)
    
    barset = api.get_barset(symbol, 'day', start=start_time.isoformat(), end=end_time.isoformat())
    bars = barset[symbol]
    
    data = pd.DataFrame({
        'time': [bar.t for bar in bars],
        'close': [bar.c for bar in bars]
    })
    
    return data

# Function to execute the trading logic based on Sharpe Ratio
def trade_based_on_sharpe():
    # Get SPY historical data for the past 30 days
    data = get_historical_data(SYMBOL, WINDOW_LENGTH)
    
    if len(data) < WINDOW_LENGTH:
        print("Not enough data to calculate Sharpe Ratio.")
        return
    
    # Calculate daily returns
    data['return'] = data['close'].pct_change().dropna()
    
    # Calculate Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(data['return'].dropna())
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Get the current SPY position
    position = api.get_position(SYMBOL) if SYMBOL in [p.symbol for p in api.list_positions()] else None
    current_price = api.get_last_trade(SYMBOL).price
    
    if sharpe_ratio > SHARPE_THRESHOLD:
        if position is None:
            # Buy SPY if the Sharpe Ratio exceeds the threshold and we don't already hold a position
            cash_balance = float(api.get_account().cash)
            qty = int(cash_balance // current_price)  # Buy as much SPY as we can with available cash
            if qty > 0:
                api.submit_order(
                    symbol=SYMBOL,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                print(f"Bought {qty} shares of SPY at {current_price}")
    else:
        if position:
            # Sell SPY if the Sharpe Ratio is below the threshold and we have a position
            api.submit_order(
                symbol=SYMBOL,
                qty=position.qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            print(f"Sold all {position.qty} shares of SPY at {current_price}")

# Main loop to run the bot daily
while True:
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Run the bot daily at 30 minutes after the market opens. Note the times (9:30 AM - 4:00 PM Eastern)
    if current_time == "09:30:00":
        trade_based_on_sharpe()
    
    # Sleep for a minute to avoid overloading the API
    time.sleep(60)

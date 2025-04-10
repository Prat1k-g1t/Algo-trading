# COMMENT: Intraday trading strategy using the GARCH model concentrating on a single stock

# TODO:
# 1. Load simulated daily and 5-minute data 
# 2. Define function to fit GARCH model and predict 1-day ahead volatility in a rolling window
# 3. Calculate prediction premium and form a daily signal from it
# 4. Merge with intraday data and calculate intraday indicators to form the intraday signal
# 5. Generate the position entry and hold until the end of the day
# 6. Calculate final strategy returns
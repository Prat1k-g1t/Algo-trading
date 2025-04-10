# How unsupervised learning is applied in trading:
#     1. Clustering
#     2. Dimensionality reduction
#     3. Anomaly detection
#     4. Market regime detection
#     5. Portfolio optimization

# TODO:
    # 1. Download SP500 stock data
    # 2. Calculate different technical indicators and features for each stock 
    #    (Garman-Klass Volatility, RSI, Bollinger bands, ATR, MACD, Dollar volume)
    # 3. Aggregate on monthly level and filter for each month only top 150 most liquid stocks
    # 4. Calculate montly returns for different time-horizons ti add to features
    # 5. Download Fama-French Factors and calculate rolling factor betas for each stock
    # 6. Investigate the use of Carhart four-factor model for the portfolio
    # 7. For each month fit a K-means clustering model to group similar assets based on their features
    # 8. For each month select assets based on the cluster and form a portfolio based on 
    #    Efficient Frontier max sharpe ratio portfolio optimization
    # 9. Visualize the portfolio returns and compare to SP500 returns

# Limitation:
# We will use most recent SP500 data, leading to survivorship bias in this list. Use survivorship free data
# What is survivorship? It is the possibility that a shock which is part of the test data in the portfolio has left the SP500.
# Therefore, always use survivorship free data. Always backtest with updated data in the portfolio

from statsmodels.regression.rolling import RollingOLS
import pandas as pd
import pandas_ta
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import datetime as dt
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')

SP500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

SP500['Symbol'] = SP500['Symbol'].str.replace('.', '-')

symbols_list = SP500['Symbol'].unique().tolist()

end_date = '2025-03-31'
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)

df = yf.download(tickers=symbols_list, start=start_date, end=end_date).stack()

df.index.names = ['date', 'ticker']

df.columns = df.column.str.lower()

df['garman_klass_vol'] = 0.5*((np.log(df['high'])-np.log(df['low']))**2)-(2*np.log(2)-1)*(np.log(df['adj close'])-np.log(df['open']))**2


df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])

df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])

df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    
    return atr.sub(atr.mean()).div(atr.std())


def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20)

    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)


df['dollar volume'] = (df['adj close']*df['volume'])/1e6

last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 
                                                         'high', 'low', 'close']]

data = pd.concat([df.unstack('ticker')['dollar volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                  df.unstack()[last_cols].resample('M').last().stack('ticker')], axis=1).dropna()

data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12).mean().stack())

data['dollar_vol_rank'] = (data.groupby('data')['dollar volume'].rank(ascending=False))

data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

def calculate_returns(df):

    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                            .pct_change(lag)
                            .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                    upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
    return df


data = data.group(level=1, group_keys=False).apply(calculate_returns).dropna()


factor_data = web.DataReader('F-F_Research_Data_5_Factors_2X3',
               'famafrench',
               start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()

factor_data = factor_data.resample('M').Last().div(100)

factor_data.index.name = 'date'

factor_data = factor_data.join(data['return_1m']).sort_index()

observations = factor_data.groupby(level=1).size()

valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks)]

betas = (factor_data.groupby(level=1,
                    group_keys=False).apply(lambda x: RollingOLS(endog=x['return_1m'],
                           exog=sm.add_constant(x.drop('return_1m', axis=1)),
                           window=min(24, x.shape[0]),
                           min_nobs=len(x.columns)+1).fit(params_only=True).params.drop('const', axis=1)))



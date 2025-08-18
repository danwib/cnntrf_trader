import pandas as pd
from pandas_datareader.stooq import StooqDailyReader
from .base import MarketDataProvider
class StooqProvider(MarketDataProvider):
    def fetch_bars(self,symbol,start,end,interval):
        if interval!='1d': raise ValueError('daily only')
        df=StooqDailyReader(symbols=symbol, start=start, end=end).read()[symbol]
        if df is None or df.empty: return pd.DataFrame(columns=['open','high','low','close','volume'])
        df=df.rename(columns=str.lower).rename_axis('timestamp')
        df.index=pd.to_datetime(df.index).tz_localize('UTC')
        return df[['open','high','low','close','volume']]

import pandas as pd, yfinance as yf
from .base import MarketDataProvider
class YFinanceProvider(MarketDataProvider):
    def fetch_bars(self,symbol,start,end,interval):
        tf={'1min':'1m','5min':'5m','15min':'15m','1h':'60m','1d':'1d'}[interval]
        df=yf.download(symbol, start=start, end=end, interval=tf, progress=False)
        if df.empty: return pd.DataFrame(columns=['open','high','low','close','volume'])
        df=df.rename(columns=str.lower)
        df.index=pd.to_datetime(df.index, utc=True)
        return df[['open','high','low','close','volume']]

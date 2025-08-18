import os, requests, pandas as pd
from .base import MarketDataProvider
class AlphaVantageProvider(MarketDataProvider):
    def __init__(self): self.key=os.getenv('ALPHAVANTAGE_API_KEY')
    def fetch_bars(self,symbol,start,end,interval):
        return pd.DataFrame(columns=['open','high','low','close','volume'])

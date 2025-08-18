import os, requests, pandas as pd
from .base import MarketDataProvider
class AlpacaProvider(MarketDataProvider):
    def __init__(self): self.key=os.getenv('ALPACA_KEY_ID'); self.secret=os.getenv('ALPACA_SECRET_KEY')
    def fetch_bars(self,symbol,start,end,interval):
        return pd.DataFrame(columns=['open','high','low','close','volume'])

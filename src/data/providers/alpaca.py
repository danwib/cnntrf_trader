# src/data/providers/alpaca.py
import os
import pandas as pd
import alpaca_trade_api as tradeapi

class AlpacaProvider:
    def __init__(self):
        key_id = os.getenv("ALPACA_KEY_ID")
        secret = os.getenv("ALPACA_SECRET_KEY")
        base_url = "https://paper-api.alpaca.markets"
        if not key_id or not secret:
            raise RuntimeError("Set ALPACA_KEY_ID and ALPACA_SECRET_KEY")
        self.api = tradeapi.REST(key_id, secret, base_url, api_version="v2")

    def fetch_bars(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        # Map our intervals to Alpaca resolutions
        interval_map = {"1min": "1Min", "5min": "5Min", "15min": "15Min", "1h": "1H", "1d": "1D"}
        tf = interval_map.get(interval)
        if tf is None:
            raise ValueError(f"Unsupported interval {interval}")

        bars = self.api.get_bars(symbol, tf, start=start, end=end).df
        if bars.empty:
            return pd.DataFrame()

        # Ensure uniform OHLCV columns
        df = bars[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(bars.index)
        return df


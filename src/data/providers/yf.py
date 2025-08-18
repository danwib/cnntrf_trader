# src/data/providers/yf.py
from __future__ import annotations
import pandas as pd
import yfinance as yf

class YFinanceProvider:
    """
    Free, no key. Notes:
    - 1m is limited to recent ~30 days.
    - 5m/15m/60m have longer history (60m commonly returns ~2 years).
    """
    _MAP = {"1min": "1m", "5min": "5m", "15min": "15m", "1h": "60m", "1d": "1d"}

    def fetch_bars(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        tf = self._MAP[interval]
        df = yf.download(symbol, start=start, end=end, interval=tf, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)
        df = df.rename(columns=str.lower)
        # Some intervals come as timezone-naive; normalize to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        out = df[["open","high","low","close","volume"]].astype(float).sort_index()
        return out

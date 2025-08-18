# src/data/fetch.py
from __future__ import annotations
import os, time
import pandas as pd

from .cache import load_cached, save_cache
from .providers.yf import YFinanceProvider
from .providers.alpha_vantage import AlphaVantageProvider
from .utils_timeseries import restrict_rth, resample_ohlcv

_YF = YFinanceProvider()
_AV = None  # lazily init only if needed

def get_bars(symbol: str, start: str, end: str, interval: str, rth_only: bool) -> pd.DataFrame:
    # 1) cache
    cached = load_cached(symbol, interval, start, end)
    if cached is not None and not cached.empty:
        df = cached
    else:
        # 2) try yfinance first (broad history, free)
        df = _YF.fetch_bars(symbol, start, end, interval)
        if df.empty or (interval in ("15min","1h") and len(df) < 50):
            # 3) fallback to Alpha Vantage
            global _AV
            if _AV is None:
                _AV = AlphaVantageProvider(api_key=os.getenv("ALPHAVANTAGE_API_KEY"))
            df = _AV.fetch_bars(symbol, start, end, interval)
        # cache result (even empty, to avoid hammering)
        if not df.empty:
            save_cache(symbol, interval, start, end, df)

    if df.empty:
        return df

    if rth_only and interval in ("1min","5min","15min","1h"):
        df = restrict_rth(df)

    # Make sure bar spacing is regular for model features; resample to exact rule if needed
    rule_map = {"1min":"1min","5min":"5min","15min":"15min","1h":"1H","1d":"1D"}
    df = resample_ohlcv(df, rule_map.get(interval, interval))
    return df

# src/data/fetch.py
from __future__ import annotations
import os
import pandas as pd

from .cache import load_cached, save_cache
from .providers.yf import YFinanceProvider
from .providers.alpha_vantage import AlphaVantageProvider
from .providers.alpaca import AlpacaProvider
from .utils_timeseries import restrict_rth, resample_ohlcv

_YF = YFinanceProvider()
_AV = None
_APCA = None

def get_bars(symbol: str, start: str, end: str, interval: str, rth_only: bool) -> pd.DataFrame:
    """
    Preferred source order:
      cache → Alpaca (if keys) → yfinance → AlphaVantage (if key)
    For daily interval, we still allow yfinance first (broad/cheap), then AV.
    """
    # 1) cache
    cached = load_cached(symbol, interval, start, end)
    if cached is not None and not cached.empty:
        df = cached
    else:
        df = pd.DataFrame()
        intraday = interval in ("1min", "5min", "15min", "1h")

        try_order = []
        if intraday:
            try_order = ["alpaca", "yf", "av"]   # <-- prefer Alpaca for intraday
        else:
            try_order = ["yf", "av"]

        global _APCA, _AV
        if "alpaca" in try_order and _APCA is None and os.getenv("ALPACA_KEY_ID") and os.getenv("ALPACA_SECRET_KEY"):
            _APCA = AlpacaProvider()
        if "av" in try_order and _AV is None and os.getenv("ALPHAVANTAGE_API_KEY"):
            _AV = AlphaVantageProvider()

        for src in try_order:
            if src == "alpaca" and _APCA is not None:
                df = _APCA.fetch_bars(symbol, start, end, interval)
            elif src == "yf":
                df = _YF.fetch_bars(symbol, start, end, interval)
            elif src == "av" and _AV is not None:
                df = _AV.fetch_bars(symbol, start, end, interval)
            if df is not None and not df.empty:
                break

        if df is not None and not df.empty:
            save_cache(symbol, interval, start, end, df)

    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype(float)

    if rth_only and interval in ("1min", "5min", "15min", "1h"):
        df = restrict_rth(df)

    rule_map = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "1H", "1d": "1D"}
    return resample_ohlcv(df, rule_map.get(interval, interval))


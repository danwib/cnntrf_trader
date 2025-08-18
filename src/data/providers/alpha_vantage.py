# src/data/providers/alpha_vantage.py
import os
import time
from typing import Literal, Dict

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

BarSize = Literal["1min", "5min", "15min", "1h", "1d"]


class RateLimitError(Exception):
    pass


class AlphaVantageProvider:
    """
    Alpha Vantage free-tier friendly fetcher.
    Notes:
      - Free tier limits: 5 requests/min, 500/day.
      - Intraday 'outputsize=full' returns limited history on free; good for prototyping.
      - Timestamps returned in US/Eastern; we convert to UTC.
    Env:
      - ALPHAVANTAGE_API_KEY
    """

    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None, polite_sleep_sec: float = 12.0):
        self.key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.key:
            raise ValueError("ALPHAVANTAGE_API_KEY not set.")
        self.polite_sleep_sec = polite_sleep_sec

    @staticmethod
    def _intraday_key_map(tf: str) -> Dict[str, str]:
        return {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "1h": "60min",
        }

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, RateLimitError)),
    )
    def _get_json(self, params: Dict) -> Dict:
        r = requests.get(self.BASE, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if "Note" in j or "Information" in j:
            # Hit rate limit or missing permission
            raise RateLimitError(j.get("Note") or j.get("Information"))
        if "Error Message" in j:
            raise requests.RequestException(j["Error Message"])
        return j

    def fetch_bars(self, symbol: str, start: str, end: str, interval: BarSize) -> pd.DataFrame:
        """
        Return UTC-indexed DataFrame with columns: open, high, low, close, volume.
        """
        if interval == "1d":
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.key,
            }
            j = self._get_json(params)
            key = "Time Series (Daily)"
            data = j.get(key, {})
            df = pd.DataFrame.from_dict(data, orient="index")
            if df.empty:
                return _empty_df()
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.rename(
                columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "6. volume": "volume",
                }
            )
            df = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
            # polite sleep to respect 5 req/min
            time.sleep(self.polite_sleep_sec)
            return _slice_df(df, start, end)

        # Intraday
        tf_map = self._intraday_key_map("dummy")  # just to access type hints
        tf_map = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "60min"}
        av_tf = tf_map[interval]

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": av_tf,
            "outputsize": "full",
            "apikey": self.key,
        }
        j = self._get_json(params)
        key = f"Time Series ({av_tf})"
        data = j.get(key, {})
        df = pd.DataFrame.from_dict(data, orient="index")
        if df.empty:
            return _empty_df()
        # AlphaVantage intraday timestamps are in America/New_York (documented); localize & convert to UTC
        df.index = pd.to_datetime(df.index).tz_localize("America/New_York").tz_convert("UTC")
        df = df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )
        df = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
        time.sleep(self.polite_sleep_sec)  # rate-limit friendly
        return _slice_df(df, start, end)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype(float)


def _slice_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    return df[(df.index >= s) & (df.index <= e)]

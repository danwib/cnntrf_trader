# src/data/providers/alpaca.py
import os
import requests
import pandas as pd

class AlpacaProvider:
    """
    REST-only provider (no alpaca-trade-api SDK).
    Uses: https://data.alpaca.markets/v2/stocks/{symbol}/bars
    Env:
      ALPACA_KEY_ID, ALPACA_SECRET_KEY
    """

    BASE = "https://data.alpaca.markets/v2"

    def __init__(self, key_id: str | None = None, secret_key: str | None = None):
        self.key = key_id or os.getenv("ALPACA_KEY_ID")
        self.secret = secret_key or os.getenv("ALPACA_SECRET_KEY")
        if not (self.key and self.secret):
            raise RuntimeError("Set ALPACA_KEY_ID and ALPACA_SECRET_KEY")
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        })

    def fetch_bars(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        tf = {"1min":"1Min","5min":"5Min","15min":"15Min","1h":"1Hour","1d":"1Day"}[interval]
        url = f"{self.BASE}/stocks/{symbol}/bars"

        params = {
            "timeframe": tf,
            "start": start,  # ISO 8601
            "end": end,
            "limit": 10000,
            "adjustment": "raw",   # or "all" if you want splits/div adjustments
            "feed": "sip" if os.getenv("ALPACA_USE_SIP") == "1" else "iex",  # free = iex
        }

        all_rows = []
        page_token = None

        while True:
            if page_token:
                params["page_token"] = page_token
            r = self.session.get(url, params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
            bars = j.get("bars", [])
            if not bars:
                break
            all_rows.extend(bars)
            page_token = j.get("next_page_token")
            if not page_token:
                break

        if not all_rows:
            return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)

        df = pd.DataFrame(all_rows)
        # columns: t (timestamp), o,h,l,c,v, n (trade count), vw (vwap)
        df["timestamp"] = pd.to_datetime(df["t"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        return df[["open","high","low","close","volume"]].astype(float)


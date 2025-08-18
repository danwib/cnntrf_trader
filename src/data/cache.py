# src/data/cache.py
import os
from pathlib import Path
import pandas as pd

CACHE_ROOT = Path(os.getenv("DATA_CACHE_DIR", "data/cache")).resolve()
CACHE_VERBOSE = os.getenv("CACHE_VERBOSE") == "1"

def cache_path(symbol: str, interval: str, start: str, end: str) -> Path:
    safe = f"{symbol}_{interval}_{start}_{end}".replace(":", "-")
    p = CACHE_ROOT / interval / symbol / f"{safe}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_cached(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame | None:
    p = cache_path(symbol, interval, start, end)
    if p.exists():
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        if CACHE_VERBOSE:
            print(f"[cache] HIT  -> {p}")
        return df
    if CACHE_VERBOSE:
        print(f"[cache] MISS -> {cache_path(symbol, interval, start, end)}")
    return None

def save_cache(symbol: str, interval: str, start: str, end: str, df: pd.DataFrame) -> None:
    p = cache_path(symbol, interval, start, end)
    df.to_parquet(p)
    if CACHE_VERBOSE:
        print(f"[cache] SAVE -> {p}")


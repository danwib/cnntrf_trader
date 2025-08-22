# src/data/utils_timeseries.py
from __future__ import annotations

from typing import Iterable, List, Optional
import numpy as np
import pandas as pd

NY_TZ = "America/New_York"


def _ensure_utc_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DatetimeIndex is tz-aware UTC and sorted ascending.
    If index is tz-naive, assume it's UTC (change if your source differs).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    idx = df.index
    if idx.tz is None:
        # assume UTC if naive (adjust if your provider is different)
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    if not idx.is_monotonic_increasing:
        df = df.sort_index()
        idx = df.index.tz_convert("UTC")
    df = df.copy()
    df.index = idx
    return df


def restrict_rth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep Regular Trading Hours (NYSE): 09:30–16:00 America/New_York.
    Assumes df.index is (or will be made) tz-aware UTC; returns filtered UTC-indexed frame.
    """
    if df.empty:
        return df
    df = _ensure_utc_sorted(df)
    idx_local = df.index.tz_convert(NY_TZ)
    # If your timestamps are bar END times (typical), keeping <= 16:00 is OK.
    mask = (
        idx_local.time >= pd.Timestamp("09:30", tz=NY_TZ).time()
    ) & (
        idx_local.time <= pd.Timestamp("16:00", tz=NY_TZ).time()
    )
    return df.loc[mask]


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Downsample OHLCV with *period-end* stamps and right-closed windows: (t-dT, t].
    This avoids ambiguous inclusion of future bars when later aligning to a base frame.
    """
    if df.empty:
        return df
    df = _ensure_utc_sorted(df)

    # Normalize common rule aliases
    rule_map = {"1min": "1T", "5min": "5T", "15min": "15T", "1h": "1H", "1d": "1D"}
    r = rule_map.get(rule, rule)

    out = (
        df.resample(r, label="right", closed="right")
          .agg({
              "open":  "first",
              "high":  "max",
              "low":   "min",
              "close": "last",
              "volume":"sum",
          })
          .dropna()
    )
    # Ensure still UTC + sorted
    out = _ensure_utc_sorted(out)
    return out


def merge_asof_back_to_base(
    base: pd.DataFrame,
    hi: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    """
    Safe alignment of higher-timeframe features back to the base timeline.
    Each base timestamp receives the most recent *completed* higher-TF value <= t.

    Requirements:
      - base.index and hi.index are UTC tz-aware, sorted
      - 'cols' exist in hi
    """
    if base.empty or hi.empty:
        return base.copy()
    base = _ensure_utc_sorted(base)
    hi   = _ensure_utc_sorted(hi)

    merged = pd.merge_asof(
        left=base.sort_index(),
        right=hi[cols].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",   # <- crucial to avoid peeking forward
        allow_exact_matches=True
    )
    return merged


def future_log_return(df: pd.DataFrame, horizon_bars: int, price_col: str = "close") -> pd.Series:
    """
    y_t = log(P_{t+h} / P_t), aligned to time t.
    Purely vectorized, causal, and leaves the last `h` rows as NaN.
    """
    if df.empty:
        return pd.Series(dtype="float64", index=df.index)

    df = _ensure_utc_sorted(df)
    p = df[price_col].astype("float64")
    y = np.log(p.shift(-horizon_bars) / p)
    # Keep the same index and dtype
    y.name = f"logret_fut_{horizon_bars}"
    return y

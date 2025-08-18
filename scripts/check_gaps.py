import pandas as pd
from datetime import time

NY_TZ = "America/New_York"

def expected_rth_index(dates, tz=NY_TZ):
    """
    Build expected 15-min bar start times for each trading date (09:30..15:45 local).
    """
    idx_parts = []
    for d in dates:
        start = pd.Timestamp.combine(pd.Timestamp(d).date(), time(9, 30)).tz_localize(tz)
        end   = pd.Timestamp.combine(pd.Timestamp(d).date(), time(15, 45)).tz_localize(tz)
        # 09:30, 09:45, ..., 15:45  => 26 bars
        idx_parts.append(pd.date_range(start, end, freq="15min", tz=tz))
    if not idx_parts:
        return pd.DatetimeIndex([], tz=tz)
    return idx_parts[0].append(idx_parts[1:])

def check_gaps_rth(df: pd.DataFrame, freq="15min"):
    """
    Gap check for RTH only: compares against per-day grids in NY time.
    Assumes df.index is tz-aware UTC and each row is a bar start time.
    """
    if df.empty:
        print("Empty DataFrame.")
        return pd.DatetimeIndex([], tz="UTC"), pd.DatetimeIndex([], tz="UTC")

    # Ensure sorted and tz-aware UTC
    df = df.sort_index()
    assert df.index.tz is not None, "Index must be tz-aware"
    assert str(df.index.tz) in ("UTC", "UTC+00:00"), "Index should be UTC"

    # Convert to NY to construct daily RTH grids
    idx_ny = df.index.tz_convert(NY_TZ)
    trading_dates = pd.Index(idx_ny.date).unique().tolist()

    exp_ny = expected_rth_index(trading_dates, tz=NY_TZ)
    exp_utc = exp_ny.tz_convert("UTC")

    # Missing and duplicates relative to RTH grid
    missing = exp_utc.difference(df.index)
    dupes = df.index[df.index.duplicated()]

    print(f"Trading days: {len(trading_dates)}")
    print(f"Expected RTH bars (26/day): {len(exp_utc)}")
    print(f"Actual bars:                {len(df)}")
    print(f"Missing bars (RTH only):    {len(missing)}")
    print(f"Duplicate bars:             {len(dupes)}")

    # Show a few examples
    if len(missing) > 0:
        print("First 10 missing:", missing[:10].tolist())

    return missing, dupes

if __name__ == "__main__":
    # Point this at one of your cached files
    path = "data/cache/15min/MSFT/MSFT_15min_2023-08-20_2025-08-18.parquet"
    df = pd.read_parquet(path)
    missing, dupes = check_gaps_rth(df, freq="15min")


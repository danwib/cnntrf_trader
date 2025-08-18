# src/data/utils_timeseries.py
import pandas as pd

NY_TZ = "America/New_York"

def restrict_rth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep Regular Trading Hours (NYSE): 09:30–16:00 America/New_York.
    Assumes df.index is tz-aware UTC; returns filtered UTC-indexed frame.
    """
    if df.empty:
        return df
    idx_local = df.index.tz_convert(NY_TZ)
    mask = (idx_local.time >= pd.Timestamp("09:30", tz=NY_TZ).time()) & (
        idx_local.time <= pd.Timestamp("16:00", tz=NY_TZ).time()
    )
    return df.loc[mask]

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample to a pandas rule, e.g. "15min" -> "15min", "1h" -> "1H", "1d" -> "1D".
    Keeps OHLCV semantics.
    """
    if df.empty:
        return df
    rule_map = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "1h", "1d": "1d"}
    r = rule_map.get(rule, rule)
    return (
        df.resample(r, origin="start")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )

def future_log_return(df: pd.DataFrame, horizon_bars: int, price_col: str = "close") -> pd.Series:
    """
    Log-return over 'horizon_bars' into the future: log(P_{t+h}/P_t)
    Assumes regular sampling; returns aligned with current row index (NaN at tail).
    """
    return (pd.Series(df[price_col]).shift(-horizon_bars).pipe(lambda s: (s / df[price_col]).apply(lambda x: 0 if pd.isna(x) else x)).apply(lambda r: pd.NA if r == 0 else r)).astype("float64").apply(lambda r: pd.NA if pd.isna(r) else float(pd.np.log(r)))  # noqa

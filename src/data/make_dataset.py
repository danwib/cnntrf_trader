# src/data/make_dataset.py
import argparse
import json
import os
from typing import List

from dotenv import load_dotenv

import numpy as np
import pandas as pd

from .feature_pipeline import engineer_basic_features
from .utils_timeseries import resample_ohlcv
from .fetch import get_bars  # unified fetch: cache → Alpaca → yfinance → Alpha Vantage

# Load variables from .env into os.environ (optional convenience)
load_dotenv()

def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def future_log_return(df: pd.DataFrame, horizon_bars: int, price_col: str = "close") -> pd.Series:
    return (np.log(df[price_col].shift(-horizon_bars)) - np.log(df[price_col]))

def interval_to_seconds(interval: str) -> int:
    return {
        "1min": 60,
        "5min": 300,
        "15min": 900,
        "1h": 3600,
        "1d": 86400,
    }[interval]

def main():
    ap = argparse.ArgumentParser(description="Build features/labels from market data (prefers Alpaca).")
    ap.add_argument("--symbols", required=True, help="Comma-separated, e.g., AAPL,MSFT,SPY")
    ap.add_argument("--start", required=True, help="UTC ISO date (e.g., 2024-01-01)")
    ap.add_argument("--end", required=True, help="UTC ISO date (e.g., 2024-12-31)")
    ap.add_argument("--base-interval", default="15min", choices=["1min", "5min", "15min", "1h", "1d"])
    ap.add_argument("--agg-intervals", default="15min,1h", help="e.g., 15min,1h")
    ap.add_argument("--label-horizons", default="4,8", help="Horizon in bars at BASE interval (e.g., 4=1h if base=15m)")
    ap.add_argument("--rth-only", action="store_true")
    ap.add_argument("--out-features", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--out-symbol-ids", required=True)     # per-row symbol ids
    ap.add_argument("--out-times", required=True)          # NEW: per-row UTC epoch seconds
    ap.add_argument("--out-meta", required=True)
    args = ap.parse_args()

    symbols = parse_csv_list(args.symbols)
    agg_intervals = parse_csv_list(args.agg_intervals)
    horizons = [int(x) for x in parse_csv_list(args.label_horizons)]

    # Simple banner so you can see which sources are wired
    has_apca = bool(os.getenv("ALPACA_KEY_ID") and os.getenv("ALPACA_SECRET_KEY"))
    has_av   = bool(os.getenv("ALPHAVANTAGE_API_KEY"))
    print(f"Data sources available → Alpaca: {has_apca} | AlphaVantage fallback: {has_av}")

    # Stable symbol→id mapping based on the order provided
    sym2id = {sym: i for i, sym in enumerate(symbols)}

    all_rows = []
    for sym in symbols:
        print(f"[{sym}] fetching {args.base_interval} (cache → Alpaca → yfinance → AlphaVantage)...")
        df = get_bars(sym, args.start, args.end, args.base_interval, rth_only=args.rth_only)
        if df.empty:
            print(f"[{sym}] no data.")
            continue

        # Build multi-scale frames (base + agg)
        frames = {"base": df}
        for rule in agg_intervals:
            if rule == args.base_interval:
                frames[rule] = df.copy()
            else:
                frames[rule] = resample_ohlcv(df, rule)

        # Engineer features per interval, suffix columns by scale, align on base index
        feats = []
        for k, fdf in frames.items():
            fe = engineer_basic_features(fdf)[
                ["open", "high", "low", "close", "volume", "ret_1", "logret_1", "rv_15", "rv_60", "rsi_14", "vol_z"]
            ].add_suffix(f"@{k}")
            feats.append(fe)
        base_idx = frames["base"].index
        feat_df = pd.concat([fe.reindex(base_idx) for fe in feats], axis=1).dropna()
        # Keep column order stable:
        feat_df = feat_df[sorted(feat_df.columns)]

        # Create labels on base interval (future log returns)
        label_df = pd.DataFrame(index=feat_df.index)
        for h in horizons:
            label_df[f"y_{h}"] = future_log_return(frames["base"].reindex(feat_df.index), h)
        label_df = label_df.dropna()

        combined = feat_df.join(label_df, how="inner")
        combined["symbol"] = sym  # keep symbol to derive symbol_ids later
        all_rows.append(combined)

    if not all_rows:
        raise SystemExit("No data collected—check keys/symbols/date range.")

    # Concatenate all symbols; rows are time-aligned within each symbol and then combined
    data = pd.concat(all_rows).sort_index()

    # Select features and pick the first horizon as main y (you can later train multi-target)
    feature_cols = [c for c in data.columns if "@" in c and not c.startswith("y_")]
    target_cols = [c for c in data.columns if c.startswith("y_")]

    X = data[feature_cols].to_numpy(dtype="float32")
    y = data[target_cols[0]].to_numpy(dtype="float32")
    symbol_ids = data["symbol"].map(sym2id).to_numpy(dtype=np.int32)

    # Per-row timestamps (UTC epoch seconds) aligned to X/y rows
    # Use int64 seconds to be robust; index is tz-aware UTC from providers.
    row_times = (data.index.astype("int64") // 1_000_000_000).astype("int64")

    # Compute summary ts for delta filtering
    global_max_ts = int(row_times.max()) if len(row_times) else None
    per_symbol_max_ts = (
        data["symbol"]
        .groupby(data["symbol"])
        .apply(lambda s: int((data.loc[s.index].index.astype("int64").max() // 1_000_000_000)))
        .to_dict()
    )

    bar_seconds = interval_to_seconds(args.base_interval)

    # Save artifacts (does not touch the provider cache)
    os.makedirs(os.path.dirname(args.out_features), exist_ok=True)
    np.save(args.out_features, X)
    np.save(args.out_labels, y)
    np.save(args.out_symbol_ids, symbol_ids)
    np.save(args.out_times, row_times)

    # Write metadata (include sym2id mapping and ts summaries)
    with open(args.out_meta, "w") as f:
        json.dump(
            {
                "symbols": symbols,
                "sym2id": sym2id,
                "base_interval": args.base_interval,
                "agg_intervals": agg_intervals,
                "label_horizons": horizons,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "rth_only": bool(args.rth_only),
                "global_max_ts": global_max_ts,
                "per_symbol_max_ts": per_symbol_max_ts,
                "bar_seconds": bar_seconds,
            },
            f,
            indent=2,
        )

    print(
        "Wrote:\n"
        f"  X            -> {args.out_features}\n"
        f"  y            -> {args.out_labels}\n"
        f"  symbol_ids   -> {args.out_symbol_ids}\n"
        f"  times        -> {args.out_times}\n"
        f"  meta         -> {args.out_meta}"
    )

if __name__ == "__main__":
    main()

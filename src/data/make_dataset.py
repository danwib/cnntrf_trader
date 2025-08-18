# src/data/make_dataset.py
import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd

from .feature_pipeline import engineer_basic_features
from .providers.alpha_vantage import AlphaVantageProvider
from .utils_timeseries import restrict_rth, resample_ohlcv

def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def future_log_return(df: pd.DataFrame, horizon_bars: int, price_col: str = "close") -> pd.Series:
    import numpy as np
    return (np.log(df[price_col].shift(-horizon_bars)) - np.log(df[price_col]))

def main():
    ap = argparse.ArgumentParser(description="Build features/labels from Alpha Vantage data.")
    ap.add_argument("--symbols", required=True, help="Comma-separated, e.g., AAPL,MSFT,SPY")
    ap.add_argument("--start", required=True, help="UTC ISO date (e.g., 2024-01-01)")
    ap.add_argument("--end", required=True, help="UTC ISO date (e.g., 2024-12-31)")
    ap.add_argument("--base-interval", default="15min", choices=["1min", "5min", "15min", "1h", "1d"])
    ap.add_argument("--agg-intervals", default="15min,1h", help="e.g., 15min,1h")
    ap.add_argument("--label-horizons", default="4,8", help="Horizon in bars at BASE interval (e.g., 4=1h if base=15m)")
    ap.add_argument("--rth-only", action="store_true")
    ap.add_argument("--out-features", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--out-meta", required=True)
    args = ap.parse_args()

    if not os.getenv("ALPHAVANTAGE_API_KEY"):
        raise SystemExit("Set ALPHAVANTAGE_API_KEY before running.")

    provider = AlphaVantageProvider()
    symbols = parse_csv_list(args.symbols)
    agg_intervals = parse_csv_list(args.agg_intervals)
    horizons = [int(x) for x in parse_csv_list(args.label_horizons)]

    all_rows = []
    for sym in symbols:
        print(f"[{sym}] fetching {args.base_interval} from Alpha Vantage...")
        df = provider.fetch_bars(sym, args.start, args.end, args.base_interval)
        if df.empty:
            print(f"[{sym}] no data.")
            continue
        if args.rth_only and args.base_interval in ("1min", "5min", "15min", "1h"):
            df = restrict_rth(df)

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

        # Create labels on base interval (future log returns)
        label_df = pd.DataFrame(index=feat_df.index)
        for h in horizons:
            label_df[f"y_{h}"] = future_log_return(frames["base"].reindex(feat_df.index), h)
        label_df = label_df.dropna()

        combined = feat_df.join(label_df, how="inner")
        combined["symbol"] = sym
        all_rows.append(combined)

    if not all_rows:
        raise SystemExit("No data collected—check symbols/date range/API limits.")

    data = pd.concat(all_rows).sort_index()

    # Select features and pick the first horizon as main y (you can later train multi-target)
    feature_cols = [c for c in data.columns if "@" in c and not c.startswith("y_")]
    target_cols = [c for c in data.columns if c.startswith("y_")]
    X = data[feature_cols].to_numpy(dtype="float32")
    y = data[target_cols[0]].to_numpy(dtype="float32")

    os.makedirs(os.path.dirname(args.out_features), exist_ok=True)
    np.save(args.out_features, X)
    np.save(args.out_labels, y)
    with open(args.out_meta, "w") as f:
        json.dump(
            {
                "symbols": symbols,
                "base_interval": args.base_interval,
                "agg_intervals": agg_intervals,
                "label_horizons": horizons,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "rth_only": bool(args.rth_only),
            },
            f,
            indent=2,
        )

    print(f"Wrote:\n  X -> {args.out_features}\n  y -> {args.out_labels}\n  meta -> {args.out_meta}")

if __name__ == "__main__":
    main()

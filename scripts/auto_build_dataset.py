# scripts/auto_build_dataset.py
# Automate: (A) 2-year master dataset builds (versioned) and (B) recent "delta" updates using saved scalers.
# Adds: delta overlap purge, and an "extended master" cumulative dataset that appends deltas (no refits).

from __future__ import annotations
import argparse, subprocess, sys, os, json, shutil, datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import joblib

# --------- Config: default 50 liquid US tickers ----------
DEFAULT_50 = (
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AVGO,AMD,GOOG,"
    "JPM,JNJ,PG,XOM,UNH,V,MA,HD,COST,PEP,"
    "BAC,WMT,ABBV,KO,LLY,MRK,ORCL,ADBE,NFLX,CRM,"
    "CSCO,CMCSA,INTC,PFE,DIS,TMO,CVX,MCD,ABNB,TXN,"
    "QCOM,NKE,CAT,IBM,GE,GS,BKNG,T,VZ,UPS"
)

ROOT = Path(__file__).resolve().parents[1]  # repo root (assumes scripts/ under repo)
ARTIFACTS = ROOT / "artifacts"
DATASETS = ARTIFACTS / "datasets"
UPDATES = DATASETS / "updates"
EXTENDED = DATASETS / "extended_master"     # cumulative, scaled, no refits

def sh(cmd: list[str], cwd: Path | None = None):
    print("→", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        sys.exit(r.returncode)

def iso_date(d: dt.date | dt.datetime) -> str:
    return d.strftime("%Y-%m-%d")

def ensure_empty_dir(p: Path):
    if p.exists() and any(p.iterdir()):
        raise SystemExit(f"[ABORT] Output dir already has files: {p}")
    p.mkdir(parents=True, exist_ok=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r") as f:
        return json.load(f)

def build_raw(symbols: str, start: str, end: str, out_dir: Path) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Calls make_dataset.py to produce raw artifacts into out_dir/tmp_raw:
      X.npy, y.npy, symbol_ids.npy, times.npy, meta.json
    """
    tmp = out_dir / "tmp_raw"
    ensure_dir(tmp)
    x_path   = tmp / "X.npy"
    y_path   = tmp / "y.npy"
    sid_path = tmp / "symbol_ids.npy"
    t_path   = tmp / "times.npy"
    meta_path= tmp / "meta.json"

    cmd = [
        sys.executable, "-m", "src.data.make_dataset",
        "--symbols", symbols,
        "--start", start, "--end", end,
        "--base-interval", "15min",
        "--agg-intervals", "15min,1h",
        "--label-horizons", "4,8",
        "--rth-only",
        "--out-features", str(x_path),
        "--out-labels", str(y_path),
        "--out-symbol-ids", str(sid_path),
        "--out-times", str(t_path),
        "--out-meta", str(meta_path),
    ]
    sh(cmd, cwd=ROOT)
    return x_path, y_path, sid_path, t_path, meta_path

# ---------------- helpers for splitting ---------------- #

def _bars_per_trading_day(meta: Dict) -> int:
    """
    Estimate bars per *trading* day from meta.
    If RTH 15m: 26; else derive from bar_seconds and RTH flag.
    """
    bar_seconds = int(meta.get("bar_seconds", 900))
    rth_only = bool(meta.get("rth_only", True))
    if rth_only:
        trading_seconds = int(6.5 * 3600)  # 09:30–16:00
    else:
        trading_seconds = 24 * 3600
    bpd = max(1, int(round(trading_seconds / bar_seconds)))
    return bpd

def _row_slices_with_purge(N: int, num_symbols: int, n_val_rows: int, n_test_rows: int,
                           purge_bars: int) -> Tuple[slice, slice, slice]:
    """
    Compute contiguous train/val/test slices from the tail, inserting a purge gap
    of 'purge_bars' base bars per symbol between train→val and val→test.
    We work in *rows*; one base bar across all symbols ≈ num_symbols rows.
    """
    purge_rows = purge_bars * num_symbols

    # Build from the tail to guarantee the requested test size
    test_end = N
    test_start = max(0, test_end - n_test_rows)

    # Purge before test
    val_end = max(0, test_start - purge_rows)
    val_start = max(0, val_end - n_val_rows)

    # Purge before val
    train_end = max(0, val_start - purge_rows)
    train_start = 0

    # Safety: if train becomes empty/negative, shrink purges first, then val
    if train_end <= train_start:
        # Try removing purges
        train_end = max(0, val_start)
        if train_end <= train_start:
            # Move boundary earlier by shrinking val
            deficit = (train_start + 1) - train_end
            val_start = max(0, val_start - deficit)
            train_end = max(0, val_start)

    tr_sl = slice(train_start, train_end)
    va_sl = slice(val_start, val_end)
    te_sl = slice(test_start, test_end)
    return tr_sl, va_sl, te_sl

def _approx_rows_for_days(test_days: int, val_days: int, meta: Dict, symids: np.ndarray) -> Tuple[int, int, int]:
    """
    Given desired trading days for val/test, approximate row counts using bars_per_day * num_symbols.
    """
    N = int(symids.shape[0])
    num_symbols = int(np.max(symids)) + 1 if symids.size > 0 else 1
    bpd = _bars_per_trading_day(meta)
    n_test_rows = max(1, int(test_days) * bpd * num_symbols) if test_days > 0 else 0
    n_val_rows  = max(0, int(val_days)  * bpd * num_symbols) if val_days  > 0 else 0

    # Cap to dataset size
    n_test_rows = min(n_test_rows, N)
    n_val_rows  = min(n_val_rows, max(0, N - n_test_rows))
    n_train_rows= max(0, N - n_val_rows - n_test_rows)
    return n_train_rows, n_val_rows, n_test_rows

# ---------------- MASTER BUILD ---------------- #

def split_and_scale_master(raw_X: Path, raw_y: Path, raw_sid: Path, raw_t: Path, raw_meta: Path, out_dir: Path,
                           train_frac=0.7, val_frac=0.15, robust=True, per_symbol=True,
                           test_days: int = 0, val_days: int = 0, seq_len: int = 64, max_horizon: int = 8):
    """
    Fit scaler(s) on TRAIN ONLY and save train/val/test + scaler(s) + meta.
    If test_days>0 (or val_days>0), use day-length holdouts with purge gaps;
    otherwise fall back to ratio-based split.
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler

    X = np.load(raw_X); y = np.load(raw_y); symids = np.load(raw_sid); times = np.load(raw_t)
    meta = load_meta(raw_meta)
    N = len(X)
    num_symbols = int(np.max(symids)) + 1 if symids.size > 0 else 1

    Scaler = RobustScaler if robust else StandardScaler

    ensure_dir(out_dir)

    use_days_based = (test_days > 0 or val_days > 0)

    if use_days_based:
        # Approximate rows by trading days, then insert purge gaps in *rows*
        n_train_rows, n_val_rows, n_test_rows = _approx_rows_for_days(test_days, val_days, meta, symids)
        purge_bars = max(0, (int(seq_len) - 1) + int(max_horizon))
        tr_sl, va_sl, te_sl = _row_slices_with_purge(
            N=N, num_symbols=num_symbols,
            n_val_rows=n_val_rows, n_test_rows=n_test_rows,
            purge_bars=purge_bars
        )
        split_info = {
            "mode": "days",
            "val_days": int(val_days),
            "test_days": int(test_days),
            "bars_per_day": _bars_per_trading_day(meta),
            "purge_bars": purge_bars,
            "purge_rows": purge_bars * num_symbols,
            "train_rows": int(tr_sl.stop - tr_sl.start),
            "val_rows": int(va_sl.stop - va_sl.start),
            "test_rows": int(te_sl.stop - te_sl.start),
        }
    else:
        # Ratio split (legacy)
        n_train = int(N * train_frac)
        n_val   = int(N * (train_frac + val_frac))
        tr_sl = slice(0, n_train); va_sl = slice(n_train, n_val); te_sl = slice(n_val, N)
        split_info = {
            "mode": "ratio",
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
            "train_rows": int(n_train),
            "val_rows": int(n_val - n_train),
            "test_rows": int(N - n_val),
            "purge_bars": 0,
            "purge_rows": 0,
        }

    # Scale (fit on TRAIN only)
    if per_symbol:
        scalers = {}
        X_tr = np.empty_like(X[tr_sl]); X_va = np.empty_like(X[va_sl]); X_te = np.empty_like(X[te_sl])
        for sid in np.unique(symids):
            m_tr = (symids[tr_sl] == sid)
            m_va = (symids[va_sl] == sid)
            m_te = (symids[te_sl] == sid)
            scaler = Scaler()
            if m_tr.sum() == 0:
                scaler.fit(X[tr_sl])  # rare, but keep pipeline moving
            else:
                scaler.fit(X[tr_sl][m_tr])
            scalers[int(sid)] = scaler
            if m_tr.any(): X_tr[m_tr] = scaler.transform(X[tr_sl][m_tr])
            if m_va.any(): X_va[m_va] = scaler.transform(X[va_sl][m_va])
            if m_te.any(): X_te[m_te] = scaler.transform(X[te_sl][m_te])

        np.savez_compressed(out_dir / "train.npz", X=X_tr, y=y[tr_sl], sym_id=symids[tr_sl])
        np.savez_compressed(out_dir / "val.npz",   X=X_va, y=y[va_sl], sym_id=symids[va_sl])
        np.savez_compressed(out_dir / "test.npz",  X=X_te, y=y[te_sl], sym_id=symids[te_sl])
        joblib.dump(scalers, out_dir / "scalers.joblib")
        scale_scope = "per-symbol"
    else:
        scaler = Scaler()
        X_tr = scaler.fit_transform(X[tr_sl])
        X_va = scaler.transform(X[va_sl])
        X_te = scaler.transform(X[te_sl])

        np.savez_compressed(out_dir / "train.npz", X=X_tr, y=y[tr_sl], sym_id=symids[tr_sl])
        np.savez_compressed(out_dir / "val.npz",   X=X_va, y=y[va_sl], sym_id=symids[va_sl])
        np.savez_compressed(out_dir / "test.npz",  X=X_te, y=y[te_sl], sym_id=symids[te_sl])
        joblib.dump(scaler, out_dir / "scaler.joblib")
        scale_scope = "global"

    # Update meta
    meta.update({
        "N": int(N),
        "n_train": int((tr_sl.stop or 0) - (tr_sl.start or 0)),
        "n_val":   int((va_sl.stop or 0) - (va_sl.start or 0)),
        "n_test":  int((te_sl.stop or 0) - (te_sl.start or 0)),
        "scale_scope": scale_scope,
        "robust": bool(robust),
        "split": split_info,
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[MASTER] wrote dataset to {out_dir}")
    print(f"[SPLIT] {split_info}")

def init_extended_master_from_version(version_dir: Path):
    """
    Initialize the cumulative 'extended master' from a master version.
    Creates EXTENDED/cum.npz with all (train+val+test) rows and EXTENDED/META pointing to version/scalers.
    """
    ensure_dir(EXTENDED)

    ztr = np.load(version_dir / "train.npz")
    zva = np.load(version_dir / "val.npz")
    zte = np.load(version_dir / "test.npz")

    X = np.concatenate([ztr["X"], zva["X"], zte["X"]], axis=0)
    y = np.concatenate([ztr["y"], zva["y"], zte["y"]], axis=0)
    sid = np.concatenate([ztr["sym_id"], zva["sym_id"], zte["sym_id"]], axis=0)

    # No timestamps in splits; fill zeros. Deltas carry real ts and we dedupe by (sid, ts).
    ts = np.zeros(len(X), dtype=np.int64)

    np.savez_compressed(EXTENDED / "cum.npz", X=X, y=y, sym_id=sid, ts=ts)
    (EXTENDED / "MASTER_VERSION.txt").write_text(version_dir.name)
    print(f"[EXTENDED] initialized from {version_dir.name} with {len(X)} rows")

# ---------------- DELTA (FILTER + TRANSFORM + APPEND) ---------------- #

def transform_with_existing_scalers_and_filter(raw_X: Path, raw_y: Path, raw_sid: Path, raw_t: Path,
                                               raw_meta: Path, master_dir: Path, out_dir: Path,
                                               seq_len: int, max_horizon: int):
    """
    Reuse master scaler(s), drop overlapping rows using purge margin, save update.npz (with ts).
    """
    X = np.load(raw_X); y = np.load(raw_y); symids = np.load(raw_sid); times = np.load(raw_t)
    master_meta = load_meta(master_dir / "meta.json")

    bar_seconds = master_meta.get("bar_seconds", 900)  # default 15min
    purge_bars = (seq_len - 1) + max_horizon
    if "global_max_ts" not in master_meta:
        raise SystemExit("[ERR] master meta missing 'global_max_ts'. Update make_dataset to write it.")
    cutoff_ts = int(master_meta["global_max_ts"]) - purge_bars * bar_seconds

    keep = times > cutoff_ts
    if keep.sum() == 0:
        raise SystemExit("[DELTA] No new rows after cutoff; nothing to update.")

    X, y, symids, times = X[keep], y[keep], symids[keep], times[keep]

    per_symbol = (master_meta.get("scale_scope") == "per-symbol")
    ensure_dir(out_dir)

    if per_symbol:
        scalers = joblib.load(master_dir / "scalers.joblib")
        X_s = np.empty_like(X)
        for sid in np.unique(symids):
            m = (symids == sid)
            s = scalers.get(int(sid))
            if s is None:
                print(f"[WARN] symbol id {sid} not in master scalers; leaving unscaled.")
                X_s[m] = X[m]
            else:
                X_s[m] = s.transform(X[m])
    else:
        scaler = joblib.load(master_dir / "scaler.joblib")
        X_s = scaler.transform(X)

    np.savez_compressed(out_dir / "update.npz", X=X_s, y=y, sym_id=symids, ts=times)
    shutil.copy2(raw_meta, out_dir / "meta.raw.json")
    (out_dir / "CUT.txt").write_text(
        f"cutoff_ts={cutoff_ts} purge_bars={purge_bars} bar_seconds={bar_seconds}\n"
    )
    print(f"[DELTA] kept {len(X)} rows after cutoff={cutoff_ts}")
    return out_dir / "update.npz"

def append_delta_to_extended_master(update_npz: Path):
    """
    Append update.npz to EXTENDED/cum.npz, with de-dup by (sym_id, ts).
    If EXTENDED not initialized yet, abort with a helpful hint.
    """
    if not (EXTENDED / "cum.npz").exists():
        raise SystemExit("[ERR] Extended master not initialized. Build master first; it will create extended master.")

    zc = np.load(EXTENDED / "cum.npz")
    Xc, yc, sic, tsc = zc["X"], zc["y"], zc["sym_id"], zc["ts"]

    zu = np.load(update_npz)
    Xu, yu, siu, tsu = zu["X"], zu["y"], zu["sym_id"], zu["ts"]

    # De-dup by (sym_id, ts). Old rows may have ts=0; keep all ts>0 from deltas.
    if len(tsc) > 0:
        existing = set(zip(sic.tolist(), tsc.tolist()))
        mask_new = np.array([(int(si), int(t)) not in existing for si, t in zip(siu, tsu)], dtype=bool)
    else:
        mask_new = np.ones(len(Xu), dtype=bool)

    if mask_new.sum() == 0:
        print("[EXTENDED] No truly new rows after de-dup. Skipping append.")
        return

    Xc2  = np.concatenate([Xc,  Xu[mask_new]], axis=0)
    yc2  = np.concatenate([yc,  yu[mask_new]], axis=0)
    sic2 = np.concatenate([sic, siu[mask_new]], axis=0)
    tsc2 = np.concatenate([tsc, tsu[mask_new]], axis=0)

    np.savez_compressed(EXTENDED / "cum.npz", X=Xc2, y=yc2, sym_id=sic2, ts=tsc2)
    print(f"[EXTENDED] appended {mask_new.sum()} new rows; total now {len(Xc2)}")

# ---------------- CLI ---------------- #

def main():
    ap = argparse.ArgumentParser(description="Automate master builds (origin) + delta updates (filtered) + extended master append.")
    ap.add_argument("--mode", choices=["master", "delta"], required=True,
                    help="master: build 2y dataset + fit scalers + init extended; delta: recent slice -> transform -> append to extended")
    ap.add_argument("--symbols", default=DEFAULT_50, help="Comma-separated tickers (defaults to curated 50).")
    ap.add_argument("--days-back", type=int, default=730, help="Lookback for master builds (default ~2y).")
    ap.add_argument("--delta-days", type=int, default=30, help="Recent window for delta updates.")
    ap.add_argument("--robust", action="store_true", help="Use RobustScaler (default True for master).")
    ap.add_argument("--global-scale", action="store_true", help="Use global scaler instead of per-symbol (master only).")
    ap.add_argument("--seq-len", type=int, default=64, help="Model sequence length (used for purge gaps).")
    ap.add_argument("--max-horizon", type=int, default=8, help="Max label horizon (bars) for purge gaps.")
    # NEW: day-length holdout for master split
    ap.add_argument("--test-days", type=int, default=0, help="If >0, use last N trading days for TEST (with purge).")
    ap.add_argument("--val-days",  type=int, default=0, help="If >0, use previous M trading days for VAL (with purge).")

    args = ap.parse_args()

    today = dt.date.today()
    if args.mode == "master":
        start = iso_date(today - dt.timedelta(days=args.days_back))
        end   = iso_date(today)
        version = f"v{end}"
        out_version = DATASETS / version
        ensure_empty_dir(out_version)

        # 1) Build raw features/labels for ~lookback window
        raw_X, raw_y, raw_sid, raw_t, raw_meta = build_raw(args.symbols, start, end, out_version)

        # 2) Split + scale (fit on train) and write version dir
        split_and_scale_master(
            raw_X, raw_y, raw_sid, raw_t, raw_meta, out_version,
            robust=(args.robust or True),           # robust default True
            per_symbol=(not args.global_scale),
            test_days=args.test_days,
            val_days=args.val_days,
            seq_len=args.seq_len,
            max_horizon=args.max_horizon,
        )

        # 3) Update "master" pointer
        (DATASETS / "master").unlink(missing_ok=True)
        (DATASETS / "master").symlink_to(out_version.name)  # relative symlink
        (DATASETS / "MASTER").write_text(version)           # plain text pointer

        # 4) Initialize extended master from this version
        init_extended_master_from_version(out_version)

        print(f"[MASTER] Now pointing master -> {version}")

    else:  # DELTA
        master_pointer = (DATASETS / "master")
        if not master_pointer.exists():
            raise SystemExit("[ERR] No master dataset found. Build one first: --mode master")

        # Resolve master dir (handle symlink)
        master_dir = (DATASETS / os.readlink(master_pointer)) if master_pointer.is_symlink() else master_pointer
        if not master_dir.exists():
            raise SystemExit(f"[ERR] Master directory not found: {master_dir}")

        start = iso_date(today - dt.timedelta(days=args.delta_days))
        end   = iso_date(today)
        update_dir = UPDATES / f"update_{end}"
        ensure_empty_dir(update_dir)

        # 1) Build raw features/labels for recent window (same symbols)
        raw_X, raw_y, raw_sid, raw_t, raw_meta = build_raw(args.symbols, start, end, update_dir)

        # 2) Transform with saved scalers (no fitting) + purge overlap; save update.npz (with ts)
        update_npz = transform_with_existing_scalers_and_filter(
            raw_X, raw_y, raw_sid, raw_t, raw_meta, master_dir, update_dir,
            seq_len=args.seq_len, max_horizon=args.max_horizon
        )

        # 3) Append to extended master (with simple de-dup by (sym_id, ts))
        append_delta_to_extended_master(update_npz)

        print(f"[DELTA] Completed delta update ({start}..{end}) against master {master_dir.name} and appended to extended master")

if __name__ == "__main__":
    main()

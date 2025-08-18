# scripts/auto_build_dataset.py
# Automate: (A) 2-year master dataset builds (versioned) and (B) recent "delta" updates using saved scalers.
# Uses your existing make_dataset.py to produce raw features, then scales/splits (or transforms only for deltas).

from __future__ import annotations
import argparse, subprocess, sys, os, json, shutil, datetime as dt
from pathlib import Path

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

def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r") as f:
        return json.load(f)

def build_raw(symbols: str, start: str, end: str, out_dir: Path):
    """Calls your make_dataset.py to produce X.npy, y.npy, symbol_ids.npy, meta.json into out_dir/tmp."""
    tmp = out_dir / "tmp_raw"
    tmp.mkdir(parents=True, exist_ok=True)
    x_path = tmp / "X.npy"
    y_path = tmp / "y.npy"
    sid_path = tmp / "symbol_ids.npy"
    meta_path = tmp / "meta.json"

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
        "--out-meta", str(meta_path),
    ]
    sh(cmd, cwd=ROOT)
    return x_path, y_path, sid_path, meta_path

def split_and_scale_master(raw_X: Path, raw_y: Path, raw_sid: Path, raw_meta: Path, out_dir: Path,
                           train_frac=0.7, val_frac=0.15, robust=True, per_symbol=True):
    """Fit scalers on train split and save train/val/test + scalers + meta."""
    import json
    from sklearn.preprocessing import StandardScaler, RobustScaler

    X = np.load(raw_X)
    y = np.load(raw_y)
    symids = np.load(raw_sid)
    meta = load_meta(raw_meta)

    N = len(X)
    n_train = int(N * train_frac)
    n_val = int(N * (train_frac + val_frac))
    tr_sl = slice(0, n_train); va_sl = slice(n_train, n_val); te_sl = slice(n_val, N)

    Scaler = RobustScaler if robust else StandardScaler

    out_dir.mkdir(parents=True, exist_ok=True)

    if per_symbol:
        scalers = {}
        X_tr = np.empty_like(X[tr_sl]); X_va = np.empty_like(X[va_sl]); X_te = np.empty_like(X[te_sl])
        for sid in np.unique(symids):
            tr_mask = (symids[tr_sl] == sid)
            va_mask = (symids[va_sl] == sid)
            te_mask = (symids[te_sl] == sid)
            scaler = Scaler()
            # Handle rare case of zero train rows for a symbol by falling back to global fit
            if tr_mask.sum() == 0:
                scaler.fit(X[tr_sl])
            else:
                scaler.fit(X[tr_sl][tr_mask])
            scalers[int(sid)] = scaler
            if tr_mask.any(): X_tr[tr_mask] = scaler.transform(X[tr_sl][tr_mask])
            if va_mask.any(): X_va[va_mask] = scaler.transform(X[va_sl][va_mask])
            if te_mask.any(): X_te[te_mask] = scaler.transform(X[te_sl][te_mask])

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

    meta.update({
        "N": int(N), "n_train": int(n_train), "n_val": int(n_val), "n_test": int(N - n_val),
        "scale_scope": scale_scope, "robust": bool(robust),
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[MASTER] wrote dataset to {out_dir}")

def transform_with_existing_scalers(raw_X: Path, raw_y: Path, raw_sid: Path, raw_meta: Path,
                                    master_dir: Path, out_dir: Path):
    """
    Reuse saved scalers from master to transform only (no fitting),
    and save a single 'update.npz' for resume training.
    """
    X = np.load(raw_X); y = np.load(raw_y); symids = np.load(raw_sid)
    master_meta = load_meta(master_dir / "meta.json")

    per_symbol = (master_meta.get("scale_scope") == "per-symbol")
    if per_symbol:
        scalers = joblib.load(master_dir / "scalers.joblib")
    else:
        scaler = joblib.load(master_dir / "scaler.joblib")

    out_dir.mkdir(parents=True, exist_ok=True)
    if per_symbol:
        X_s = np.empty_like(X)
        for sid in np.unique(symids):
            mask = (symids == sid)
            if not mask.any(): continue
            s = scalers.get(int(sid))
            if s is None:
                # Fallback: if a brand-new symbol sneaks in, use a global surrogate
                print(f"[WARN] symbol id {sid} not in master scalers; applying global fallback.")
                # Fit a surrogate on the whole master train would be ideal; here we just identity-transform
                X_s[mask] = X[mask]
            else:
                X_s[mask] = s.transform(X[mask])
    else:
        X_s = scaler.transform(X)

    np.savez_compressed(out_dir / "update.npz", X=X_s, y=y, sym_id=symids)
    # copy meta (helpful for train scripts)
    shutil.copy2(raw_meta, out_dir / "meta.raw.json")
    print(f"[DELTA] wrote update to {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Automate 2-year master builds and recent delta updates.")
    ap.add_argument("--mode", choices=["master", "delta"], required=True,
                    help="master = build 2y dataset + fit scalers; delta = recent slice using existing scalers")
    ap.add_argument("--symbols", default=DEFAULT_50, help="Comma-separated tickers (defaults to curated 50).")
    ap.add_argument("--days-back", type=int, default=730, help="Lookback for master builds (default ~2y).")
    ap.add_argument("--delta-days", type=int, default=30, help="Recent window for delta updates.")
    ap.add_argument("--robust", action="store_true", help="Use RobustScaler (default True for master).")
    ap.add_argument("--global-scale", action="store_true", help="Use global scaler instead of per-symbol (master only).")
    args = ap.parse_args()

    today = dt.date.today()
    if args.mode == "master":
        start = iso_date(today - dt.timedelta(days=args.days_back))
        end = iso_date(today)
        version = f"v{end}"
        out_version = DATASETS / version
        ensure_empty_dir(out_version)

        # 1) build raw features/labels for 2y window
        raw_X, raw_y, raw_sid, raw_meta = build_raw(args.symbols, start, end, out_version)

        # 2) split + scale (fit on train) and write version dir
        split_and_scale_master(
            raw_X, raw_y, raw_sid, raw_meta, out_version,
            robust=(args.robust or True),  # robust default True
            per_symbol=(not args.global_scale)
        )

        # 3) update "master" pointer
        (DATASETS / "master").unlink(missing_ok=True)
        (DATASETS / "master").symlink_to(out_version.name)  # relative symlink
        (DATASETS / "MASTER").write_text(version)  # also write a plain text pointer

        print(f"[MASTER] Now pointing master -> {version}")

    else:  # delta
        master_pointer = (DATASETS / "master")
        if not master_pointer.exists():
            raise SystemExit("[ERR] No master dataset found. Build one first: --mode master")

        # Resolve master dir (handle symlink)
        master_dir = (DATASETS / os.readlink(master_pointer)) if master_pointer.is_symlink() else master_pointer
        if not master_dir.exists():
            raise SystemExit(f"[ERR] Master directory not found: {master_dir}")

        start = iso_date(today - dt.timedelta(days=args.delta_days))
        end = iso_date(today)
        update_dir = UPDATES / f"update_{end}"
        ensure_empty_dir(update_dir)

        # 1) build raw features/labels for recent window (same symbols)
        raw_X, raw_y, raw_sid, raw_meta = build_raw(args.symbols, start, end, update_dir)

        # 2) transform with saved scalers (no fitting) into single update.npz
        transform_with_existing_scalers(raw_X, raw_y, raw_sid, raw_meta, master_dir, update_dir)

        print(f"[DELTA] Completed delta update ({start}..{end}) against master {master_dir.name}")

if __name__ == "__main__":
    main()

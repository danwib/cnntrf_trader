# scripts/auto_build_dataset.py
# Automate: (A) 2-year master dataset builds (versioned) and (B) recent "delta" updates using saved scalers.
# Adds: delta overlap purge, and an "extended master" cumulative dataset that appends deltas (no refits).

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

def build_raw(symbols: str, start: str, end: str, out_dir: Path):
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
        "--out-times", str(t_path),              # << requires make_dataset to support this
        "--out-meta", str(meta_path),
    ]
    sh(cmd, cwd=ROOT)
    return x_path, y_path, sid_path, t_path, meta_path

# ---------------- MASTER BUILD ---------------- #

def split_and_scale_master(raw_X: Path, raw_y: Path, raw_sid: Path, raw_meta: Path, out_dir: Path,
                           train_frac=0.7, val_frac=0.15, robust=True, per_symbol=True):
    """
    Fit scaler(s) on train split and save train/val/test + scaler(s) + meta.
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler

    X = np.load(raw_X); y = np.load(raw_y); symids = np.load(raw_sid)
    meta = load_meta(raw_meta)

    N = len(X)
    n_train = int(N * train_frac)
    n_val   = int(N * (train_frac + val_frac))
    tr_sl = slice(0, n_train); va_sl = slice(n_train, n_val); te_sl = slice(n_val, N)

    Scaler = RobustScaler if robust else StandardScaler

    ensure_dir(out_dir)

    if per_symbol:
        scalers = {}
        X_tr = np.empty_like(X[tr_sl]); X_va = np.empty_like(X[va_sl]); X_te = np.empty_like(X[te_sl])
        for sid in np.unique(symids):
            tr_mask = (symids[tr_sl] == sid)
            va_mask = (symids[va_sl] == sid)
            te_mask = (symids[te_sl] == sid)
            scaler = Scaler()
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

def init_extended_master_from_version(version_dir: Path):
    """
    Initialize the cumulative 'extended master' from a master version.
    Creates EXTENDED/cum.npz with all (train+val+test) rows and EXTENDED/META pointing to version/scalers.
    """
    ensure_dir(EXTENDED)

    # Load splits and concat
    ztr = np.load(version_dir / "train.npz")
    zva = np.load(version_dir / "val.npz")
    zte = np.load(version_dir / "test.npz")

    X = np.concatenate([ztr["X"], zva["X"], zte["X"]], axis=0)
    y = np.concatenate([ztr["y"], zva["y"], zte["y"]], axis=0)
    sid = np.concatenate([ztr["sym_id"], zva["sym_id"], zte["sym_id"]], axis=0)

    # We don't have per-row timestamps in splits; add a placeholder ts of zeros.
    # (Deltas we append WILL include 'ts', and we'll dedupe by (sid, ts) where ts>0.)
    ts = np.zeros(len(X), dtype=np.int64)

    np.savez_compressed(EXTENDED / "cum.npz", X=X, y=y, sym_id=sid, ts=ts)
    # Remember which version/scalers this extended master ties to
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

    # Compute cutoff to purge context overlap
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

    # De-duplication by (sym_id, ts). Extended master may have old rows with ts=0; we keep all ts>0.
    if len(tsc) > 0:
        existing = set(zip(sic.tolist(), tsc.tolist()))
        mask_new = np.array([(int(si), int(t)) not in existing for si, t in zip(siu, tsu)], dtype=bool)
    else:
        mask_new = np.ones(len(Xu), dtype=bool)

    if mask_new.sum() == 0:
        print("[EXTENDED] No truly new rows after de-dup. Skipping append.")
        return

    Xc2 = np.concatenate([Xc, Xu[mask_new]], axis=0)
    yc2 = np.concatenate([yc, yu[mask_new]], axis=0)
    sic2= np.concatenate([sic, siu[mask_new]], axis=0)
    tsc2= np.concatenate([tsc, tsu[mask_new]], axis=0)

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
    ap.add_argument("--seq-len", type=int, default=64, help="Model sequence length for delta purge.")
    ap.add_argument("--max-horizon", type=int, default=8, help="Max label horizon (bars) for delta purge.")
    args = ap.parse_args()

    today = dt.date.today()
    if args.mode == "master":
        start = iso_date(today - dt.timedelta(days=args.days_back))
        end   = iso_date(today)
        version = f"v{end}"
        out_version = DATASETS / version
        ensure_empty_dir(out_version)

        # 1) Build raw features/labels for ~2y window
        raw_X, raw_y, raw_sid, raw_t, raw_meta = build_raw(args.symbols, start, end, out_version)

        # 2) Split + scale (fit on train) and write version dir
        split_and_scale_master(
            raw_X, raw_y, raw_sid, raw_meta, out_version,
            robust=(args.robust or True),  # robust default True
            per_symbol=(not args.global_scale)
        )

        # 3) Update "master" pointer
        (DATASETS / "master").unlink(missing_ok=True)
        (DATASETS / "master").symlink_to(out_version.name)  # relative symlink
        (DATASETS / "MASTER").write_text(version)           # plain text pointer

        # 4) Initialize extended master from this version (one-time per new master)
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

# scripts/split_scale.py
import argparse, json, joblib, numpy as np, pathlib as p
from sklearn.preprocessing import StandardScaler, RobustScaler

def split_idx(N, train_frac=0.7, val_frac=0.15):
    n_train = int(N * train_frac)
    n_val = int(N * (train_frac + val_frac))
    return slice(0, n_train), slice(n_train, n_val), slice(n_val, N)

def fit_transform_global(X_train, X_val, X_test, robust=False):
    Scaler = RobustScaler if robust else StandardScaler
    scaler = Scaler()
    X_train_s = scaler.fit_transform(X_train)
    return X_train_s, scaler.transform(X_val), scaler.transform(X_test), scaler

def fit_transform_per_symbol(X, sym_ids, train_sl, val_sl, test_sl, robust=False):
    Scaler = RobustScaler if robust else StandardScaler
    scalers = {}
    X_train_s = np.empty_like(X[train_sl]); X_val_s = np.empty_like(X[val_sl]); X_test_s = np.empty_like(X[test_sl])
    # process each symbol independently
    for sid in np.unique(sym_ids):
        tr_mask = (sym_ids[train_sl] == sid)
        va_mask = (sym_ids[val_sl] == sid)
        te_mask = (sym_ids[test_sl] == sid)
        scaler = Scaler()
        if tr_mask.sum() == 0:
            # fallback: use global stats if no train rows for this sid
            scaler = Scaler().fit(X[train_sl])
        else:
            scaler.fit(X[train_sl][tr_mask])
        scalers[int(sid)] = scaler
        if tr_mask.any(): X_train_s[tr_mask] = scaler.transform(X[train_sl][tr_mask])
        if va_mask.any(): X_val_s[va_mask]   = scaler.transform(X[val_sl][va_mask])
        if te_mask.any(): X_test_s[te_mask]  = scaler.transform(X[test_sl][te_mask])
    return X_train_s, X_val_s, X_test_s, scalers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", default="artifacts/X.npy")
    ap.add_argument("--y", default="artifacts/y.npy")
    ap.add_argument("--symids", default="artifacts/symbol_ids.npy")
    ap.add_argument("--meta", default="artifacts/meta.json")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--outdir", default="artifacts/datasets/v1")
    ap.add_argument("--robust", action="store_true", help="RobustScaler instead of StandardScaler")
    ap.add_argument("--scale-scope", choices=["global","per-symbol"], default="per-symbol")
    args = ap.parse_args()

    out = p.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    X = np.load(args.X); y = np.load(args.y); sym_ids = np.load(args.symids)
    N = len(X)
    tr_sl, va_sl, te_sl = split_idx(N, args.train_frac, args.val_frac)

    if args.scale_scoop == "global":  # typo guard, in case user copies older script
        args.scale_scope = "global"

    if args.scale_scope == "global":
        X_tr, X_va, X_te, scaler = fit_transform_global(X[tr_sl], X[va_sl], X[te_sl], robust=args.robust)
        joblib.dump(scaler, out / "scaler.joblib")
    else:
        X_tr, X_va, X_te, scalers = fit_transform_per_symbol(X, sym_ids, tr_sl, va_sl, te_sl, robust=args.robust)
        joblib.dump(scalers, out / "scalers.joblib")  # dict: {sym_id: fitted Scaler}

    # save splits (include symbol ids for the model’s embedding)
    np.savez_compressed(out / "train.npz", X=X_tr, y=y[tr_sl], sym_id=sym_ids[tr_sl])
    np.savez_compressed(out / "val.npz",   X=X_va, y=y[va_sl], sym_id=sym_ids[va_sl])
    np.savez_compressed(out / "test.npz",  X=X_te, y=y[te_sl], sym_id=sym_ids[te_sl])

    # pass-through meta + split sizes
    meta = json.load(open(args.meta))
    meta.update({"N": int(N), "n_train": int(N*args.train_frac), "n_val": int(N*(args.train_frac+args.val_frac)),
                 "scale_scope": args.scale_scope, "robust": bool(args.robust)})
    json.dump(meta, open(out / "meta.json", "w"), indent=2)

    print("Wrote:", out)

if __name__ == "__main__":
    main()

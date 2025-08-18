# scripts/split_scale.py
import argparse, json, joblib, numpy as np, pathlib as p
from sklearn.preprocessing import StandardScaler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", default="artifacts/X.npy")
    ap.add_argument("--y", default="artifacts/y.npy")
    ap.add_argument("--meta", default="artifacts/meta.json")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--outdir", default="artifacts/datasets/v1")
    ap.add_argument("--robust", action="store_true", help="use RobustScaler instead of StandardScaler")
    args = ap.parse_args()

    X = np.load(args.X)
    y = np.load(args.y)

    N = len(X)
    n_train = int(N * args.train_frac)
    n_val   = int(N * (args.train_frac + args.val_frac))

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
    X_test,  y_test  = X[n_val:], y[n_val:]

    Scaler = ( __import__("sklearn.preprocessing", fromlist=["RobustScaler"]).preprocessing.RobustScaler
               if args.robust else StandardScaler )
    scaler = Scaler()
    scaler.fit(X_train)                  # <-- fit on train only

    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    out = p.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "train.npz", X=X_train_s, y=y_train)
    np.savez_compressed(out / "val.npz",   X=X_val_s,   y=y_val)
    np.savez_compressed(out / "test.npz",  X=X_test_s,  y=y_test)

    joblib.dump(scaler, out / "scaler.joblib")

    # pass through meta (plus split sizes)
    meta = json.load(open(args.meta))
    meta.update({"N": N, "n_train": n_train, "n_val": n_val, "n_test": N - n_val})
    json.dump(meta, open(out / "meta.json", "w"), indent=2)

    print("Wrote:", out)

if __name__ == "__main__":
    main()

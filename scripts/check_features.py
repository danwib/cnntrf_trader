import numpy as np, json

X = np.load("artifacts/X.npy")
y = np.load("artifacts/y.npy")
with open("artifacts/meta.json") as f:
    meta = json.load(f)

print("Shapes:", X.shape, y.shape)
print("Feature count:", len(meta["feature_cols"]))
print("Target cols:", meta["target_cols"][:5], "...")

# Basic stats
means = X.mean(axis=0)
stds = X.std(axis=0)
print("Mean range:", means.min(), "→", means.max())
print("Std range:", stds.min(), "→", stds.max())

print("y stats → min:", y.min(), "max:", y.max(), "mean:", y.mean(), "std:", y.std())

# sanity check on RSI feature if present
rsi_cols = [i for i,c in enumerate(meta["feature_cols"]) if "rsi" in c]
if rsi_cols:
    print("RSI sample:", X[:10, rsi_cols[0]])

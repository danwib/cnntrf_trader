# CNN + Transformer Time-Series Starter (with Data Providers)

A research repo for **short-horizon equity return forecasting** on intraday bars.  
It includes data fetching & caching, feature engineering, windowed datasets, and a **CNN + Transformer** model with a clean training/evaluation pipeline.

## Highlights
- **Data providers & cache:** Alpaca (IEX free feed) with fallbacks (yfinance, AlphaVantage), local Parquet cache.
- **Dataset builder:** engineered features across multiple intervals, alignment, and future-return labels.
- **Automation:** build a **versioned master** dataset and **delta** updates (overlap-purged, de-duplicated).
- **Models:** modern **CNN + PatchTST-style Transformer** (default) + baseline model for A/B.
- **Training loop:** AMP, AdamW, early stopping, checkpoints, JSON/CSV logs.
- **Eval:** MSE/MAE/dir-acc + extras; ready to extend with cost-aware thresholding & PnL.
- **Reproducible artifacts:** versioned datasets and timestamped run folders.

---

## Repo structure (key paths)

```
.
├─ src/
│  ├─ data/
│  │  ├─ make_dataset.py              # builds X/y/symbol_ids/times/meta
│  │  ├─ fetch.py                     # unified fetch (cache → Alpaca → yfinance → AV)
│  │  ├─ providers/
│  │  │  ├─ alpaca.py                 # REST client (no alpaca-trade-api dep)
│  │  │  └─ alpha_vantage.py
│  │  ├─ feature_pipeline.py          # engineer_basic_features()
│  │  └─ utils_timeseries.py
│  ├─ models/
│  │  ├─ cnn_patchtst.py              # CNN + PatchTST-style Transformer (default)
│  │  ├─ modules/                     # stem/patcher/transformer/mixer modules
│  │  └─ cnn_transformer.py           # (optional) baseline model for A/B
│  ├─ training/
│  │  ├─ train.py                     # training loop + checkpoints
│  │  ├─ eval.py                      # evaluation on test split
│  │  └─ data_module.py               # WindowedDataset + build_dataloaders()
│  └─ inference/
│     └─ (stubs / to be extended)     # decision rules, live loop, etc.
├─ scripts/
│  ├─ auto_build_dataset.py           # master & delta automation (+ extended master append)
│  └─ split_scale.py                  # (older splitter; superseded by automation)
├─ artifacts/
│  ├─ datasets/
│  │  ├─ master -> vYYYY-MM-DD/       # symlink pointer
│  │  ├─ vYYYY-MM-DD/                 # versioned master dataset (train/val/test)
│  │  ├─ updates/update_YYYY-MM-DD/   # delta slices (scaled)
│  │  └─ extended_master/cum.npz      # cumulative, scaled, with (sym_id, ts)
│  └─ runs/                           # timestamped training runs (+ checkpoints, metrics)
├─ data/
│  └─ cache/                          # parquet cache (ignored by git)
├─ .env.example                       # API keys for providers
├─ requirements.txt
└─ README.md
```

---

## Quickstart

### 0) Environment
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1) Configure data providers
Copy `.env.example` → `.env` and set keys (only Alpaca is typically required):
```
ALPACA_KEY_ID=...
ALPACA_SECRET_KEY=...
ALPHAVANTAGE_API_KEY=...      # optional fallback
```

### 2) Build a master dataset (scaled, split by time)
Build ~2 years for 50 liquid US tickers (default list inside the script). This creates a versioned dataset under `artifacts/datasets/vYYYY-MM-DD/` and points `artifacts/datasets/master` to it.

```bash
python -m scripts.auto_build_dataset --mode master
```

**Outputs in `artifacts/datasets/master/`:**
- `train.npz`, `val.npz`, `test.npz` (each: `X`, `y`, `sym_id`)
- `scalers.joblib` (per-symbol RobustScaler) or `scaler.joblib` (global)
- `meta.json` (schema, feature names, label horizons, intervals, etc.)

### 3) (Optional) Pull recent deltas & extend the master
Purges overlap to prevent leakage; appends unique rows into `extended_master/cum.npz`.

```bash
python -m scripts.auto_build_dataset --mode delta --seq-len 64 --max-horizon 8
```

---

## Training

The default architecture is a **CNN + PatchTST-style Transformer** (PatchTST hybrid). Keep the baseline around for comparisons if you like.

### GPU tips
- Use `--use-amp` on NVIDIA GPUs (A10/3090/A100).  
- Recommended: `torch.set_float32_matmul_precision('high')`, `cudnn.benchmark=True` (already handled in code).

### Smoke test (CPU or GPU)
```bash
python -m src.training.train   --arch patchtst   --seq-len 64 --stride 1   --batch-size 128 --max-epochs 2 --use-amp --num-workers 2   --cnn-hidden 128 --cnn-blocks 3 --cnn-kernels 5,5,3 --cnn-dilations 1,2,4   --patch-size 8 --patch-stride 4   --d-model 128 --n-heads 4 --num-layers 3 --dropout 0.1   --channel-mixer mlp --emb-dim 12   --loss huber --huber-delta 0.01 --lr 3e-4 --weight-decay 0.01 --grad-clip 1.0
```

### Full run (single GPU)
```bash
python -m src.training.train   --arch patchtst   --seq-len 64 --stride 1   --batch-size 256 --max-epochs 20 --use-amp --num-workers 4   --cnn-hidden 128 --cnn-blocks 3 --cnn-kernels 5,5,3 --cnn-dilations 1,2,4   --patch-size 8 --patch-stride 4   --d-model 128 --n-heads 4 --num-layers 3 --dropout 0.1   --channel-mixer mlp --emb-dim 12   --loss huber --huber-delta 0.01 --lr 3e-4 --weight-decay 0.01 --grad-clip 1.0
```

**A/B with baseline** (if you keep `src/models/cnn_transformer.py`):
```bash
python -m src.training.train --arch baseline  --seq-len 64 ...
```

### Checkpoints & run folder
On each run the trainer creates:
```
artifacts/runs/<YYYYMMDD-HHMMSS>/
  ├─ checkpoints/
  │   └─ best.pt                  # model_state, optimizer_state, model_cfg, train_args, epoch, val_loss
  ├─ configs/
  │   ├─ model_config.json
  │   └─ train_config.json
  └─ metrics_test.json            # written after final test eval
```

To resume (useful for Spot instances):
```bash
python -m src.training.train ... --resume artifacts/runs/<ts>/checkpoints/best.pt
```

---

## Evaluation

Evaluate a checkpoint on the **test** split; the script rebuilds the model from the saved config:

```bash
python -m src.training.eval   --checkpoint artifacts/runs/<ts>/checkpoints/best.pt   --datasets-root artifacts/datasets   --batch-size 512 --num-workers 4
```

It prints and writes JSON with:
- `mse`, `mae`, **directional accuracy** (`dir_acc`)
- **hit_rate_top_decile** (by |y|), and **pearson_r**.

> **Next step (optional):** add **cost-aware thresholding & PnL** on the validation set to pick a decision threshold τ, then report out-of-sample PnL on test. A minimal `decision.py` helper is easy to add later.

---

## Data & features (what the model sees)

- **Base interval:** 15-minute bars (typical), `bar_seconds=900`.  
- **Engineered features** per interval:  
  `open, high, low, close, volume, ret_1, logret_1, rv_15, rv_60, rsi_14, vol_z`  
  Suffixes indicate scale, e.g., `close@base`, `rsi_14@1h`.  
- **Labels:** future log returns at selected horizons (e.g., `y_4` = +4 bars).  
- **Scaled artifacts (master split):**  
  - `X`: `(N, F)` float32 — **already scaled**  
  - `y`: `(N,)` float32 — target (first horizon, e.g., `y_4`)  
  - `sym_id`: `(N,)` int32 — maps to `meta["sym2id"]`

### Windowing for training
`WindowedDataset` reconstructs per-symbol sequences and yields:
- `x`: `(B, C, L)` channel-first windows  
- `sym_id`: `(B,)` for symbol embeddings  
- `y`: `(B,)` target aligned to the **last step** in the window

It **never crosses symbol boundaries**.

---

## Model overview (default)

**CNN + PatchTST-style Transformer**

1. **CNN stem (TCN-like):** depthwise-separable Conv1d blocks with dilations `[1,2,4]`, GLU gating, GroupNorm → local temporal patterns.
2. **Channel mixer:** lightweight cross-channel mixing (`mlp` 1×1 conv or `ivar` iTransformer-style).  
3. **Patch tokenizer:** Conv1d over time (`patch_size`, `patch_stride`) → tokens `(B, T, D)`.
4. **Time Transformer:** causal Transformer encoder (2–4 layers) over tokens.
5. **Pooling + head:** last token (causal) → linear regression head to predict future log return.

Symbol embeddings (`--emb-dim`) are **tiled as extra channels** before the CNN stem.

---

## Common flags (trainer)

- `--arch {patchtst,baseline}`  
- `--seq-len`, `--stride`  
- CNN: `--cnn-hidden`, `--cnn-blocks`, `--cnn-kernels 5,5,3`, `--cnn-dilations 1,2,4`, `--cnn-dropout`  
- Patching: `--patch-size`, `--patch-stride`  
- Transformer: `--d-model`, `--n-heads`, `--num-layers`, `--dropout`  
- Mixer: `--channel-mixer {mlp,ivar,none}`  
- Embeddings: `--emb-dim`  
- Optim: `--lr`, `--weight-decay`, `--grad-clip`, `--use-amp`, `--max-epochs`, `--patience`  
- Loss: `--loss {huber,mse}`, `--huber-delta`

---

## Reproducibility & benchmarks

- Trainer saves both **training config** and **model config** to the run folder.  
- Early stopping on **validation loss**; best checkpoint auto-saved.  
- For reliability, we recommend:
  - Train **2–3 seeds** and report mean ± stdev.
  - Walk-forward splits (e.g., retrain on successive windows, test on the next month/quarter).

---

## AWS / GPU notes (optional)

- **Single-GPU** (A10/L4/3090) is sufficient to train the default model.  
- Keep datasets/checkpoints on **S3**; use an **EBS gp3** volume for the VM.  
- **Spot instances** are cost-effective; use `--resume` with frequent checkpoints.

---

## Troubleshooting

- **“No module named src…”** → ensure `src/` has `__init__.py` files (`src`, `src/models`, `src/training`, `src/models/modules`).  
- **Data not found** → run `python -m scripts.auto_build_dataset --mode master` first; ensure `artifacts/datasets/master` points to the latest version directory.  
- **CPU run with `--use-amp`** → AMP is auto-disabled on CPU; it’s fine to leave the flag on for cross-device scripts.  
- **Throughput** → increase `--batch-size`, set `--num-workers` near half your CPU threads, and keep `--use-amp` on for NVIDIA GPUs.

---

## Roadmap (next steps)
- **Decision rules & cost-aware thresholding** (select τ on validation; report out-of-sample PnL).  
- **Dual-head model** (regression + classification) with probability calibration.  
- **Live loop & execution** via Alpaca (brackets/limits), risk controls, and monitoring.  
- **More features**: time-of-day/day-of-week embeddings, volatility regime, microstructure proxies.

---

> **Note:** This repository is for research/education. It does **not** constitute financial advice. Use at your own risk.

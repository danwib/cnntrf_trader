
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.models.cnn_transformer import CNNTransformer, CNNTransformerConfig


def load_checkpoint(ckpt_path: str | Path, device: torch.device) -> CNNTransformer:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = CNNTransformerConfig(**ckpt["model_cfg"])
    model = CNNTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_meta(meta_path: str | Path) -> Dict:
    with open(meta_path, "r") as f:
        return json.load(f)


def build_latest_sequence(
    datasets_root: str | Path,
    symbol: str,
    seq_len: int,
) -> tuple[np.ndarray, int]:
    """
    Attempts to build the latest (F, L) sequence for a given symbol from the extended master dataset.
    Requires artifacts/datasets/extended_master/cum.npz and artifacts/datasets/master/meta.json
    """
    root = Path(datasets_root).resolve()
    meta = load_meta(root / "meta.json")
    sym2id = meta["sym2id"]
    if symbol not in sym2id:
        raise ValueError(f"Symbol {symbol} not in sym2id")
    sid = int(sym2id[symbol])

    # Prefer extended master with timestamps (cum.npz), else fallback to master train+val+test concatenation order
    ext = Path("artifacts/datasets/extended_master/cum.npz")
    if ext.exists():
        arr = np.load(ext)
        X = arr["X"]; y = arr["y"]; s = arr["sym_id"]
        idxs = np.where(s == sid)[0]
        if len(idxs) < seq_len:
            raise ValueError(f"Not enough rows for {symbol} in extended master ({len(idxs)} < {seq_len})")
        # take last contiguous seq_len rows for this symbol
        idxs = idxs[-seq_len:]
        x = X[idxs].T  # (F, L)
        return x.astype(np.float32), sid

    # Fallback: concatenate splits from master (order is historical then increasing)
    chunks = []
    for split in ["train", "val", "test"]:
        p = root / f"{split}.npz"
        if p.exists():
            arr = np.load(p)
            mask = arr["sym_id"] == sid
            chunks.append(arr["X"][mask])
    if not chunks:
        raise FileNotFoundError("No data found to build sequence. Expected extended_master/cum.npz or master splits.")
    Xcat = np.vstack(chunks)
    if len(Xcat) < seq_len:
        raise ValueError(f"Not enough rows for {symbol} in master splits ({len(Xcat)} < {seq_len})")
    x = Xcat[-seq_len:].T
    return x.astype(np.float32), sid


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Offline inference for latest window on a symbol")
    ap.add_argument("--checkpoint", required=True, type=str, help="Path to trained checkpoint (.pt)")
    ap.add_argument("--symbol", required=True, type=str)
    ap.add_argument("--datasets-root", type=str, default="artifacts/datasets/master")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)

    x, sid = build_latest_sequence(args.datasets_root, args.symbol, args.seq_len)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)     # (1, C, L)
    s_t = torch.tensor([sid], dtype=torch.long, device=device)
    y_hat = model(x_t, s_t).item()

    print(json.dumps({"symbol": args.symbol, "y_hat": y_hat}, indent=2))


if __name__ == "__main__":
    main()

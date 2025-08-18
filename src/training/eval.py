#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Data loaders (from your repo)
try:
    from .data_module import build_dataloaders
except Exception as e:
    raise RuntimeError(
        "Could not import data_module. Ensure src/training/data_module.py exists and exports build_dataloaders."
    ) from e

# New model (PatchTST hybrid)
from src.models.cnn_patchtst import CNNPatchTST, CNNPatchTSTConfig


# ------------------------- helpers -------------------------

def resolve_master_dir(datasets_root: Path) -> Path:
    master_ptr = datasets_root / "master"
    if not master_ptr.exists():
        raise FileNotFoundError(f"Missing pointer {master_ptr}.")
    if master_ptr.is_symlink():
        return master_ptr.resolve()
    if master_ptr.is_file():
        txt = master_ptr.read_text().strip()
        p = (datasets_root / txt).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Master pointer file points to missing path: {p}")
        return p
    return master_ptr


def infer_dims(master_dir: Path) -> Tuple[int, int]:
    train_npz = master_dir / "train.npz"
    if not train_npz.exists():
        raise FileNotFoundError(f"Could not find {train_npz}. Did you run the master builder?")
    with np.load(train_npz) as d:
        X = d["X"]
        sym = d["sym_id"]
        F = X.shape[1]
        num_symbols = int(np.max(sym)) + 1 if sym.size > 0 else 1
    return F, num_symbols


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def make_baseline_model(in_channels: int, num_symbols: int, seq_len: int, saved_args: Dict[str, Any]) -> nn.Module:
    """
    Reconstruct the older baseline CNN+Transformer if you still have src/models/cnn_transformer.py.
    """
    try:
        from src.models.cnn_transformer import CNNTransformer, CNNTransformerConfig  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Checkpoint indicates a 'baseline' model, but src/models/cnn_transformer.py "
            "is missing. Restore that file or re-evaluate with a PatchTST checkpoint."
        ) from e

    # Do best-effort mapping from saved train args (if present)
    emb_dim = saved_args.get("emb_dim", 12)
    d_model = saved_args.get("d_model", 128)
    n_heads = saved_args.get("n_heads", 4)
    num_layers = saved_args.get("num_layers", 3)
    dropout = saved_args.get("dropout", 0.1)
    cnn_hidden = saved_args.get("cnn_hidden", 128)
    cnn_kernels = tuple(int(x) for x in str(saved_args.get("cnn_kernels", "5,5,3")).split(","))

    cfg = CNNTransformerConfig(
        in_channels=in_channels,
        num_symbols=num_symbols,
        seq_len=seq_len,
        emb_dim=emb_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
        conv_channels=cnn_hidden,
        kernel_sizes=cnn_kernels,
    )
    return CNNTransformer(cfg)


def rebuild_model(ckpt: Dict[str, Any], num_features: int, num_symbols: int, device: torch.device) -> nn.Module:
    """
    Rebuild the exact model class from checkpoint metadata.
    Prefers explicit CNNPatchTSTConfig; falls back to baseline if needed.
    """
    model_cfg = ckpt.get("model_cfg", {})
    train_args = ckpt.get("train_args", {})
    # If the checkpoint was saved by the new trainer, model_cfg will match CNNPatchTSTConfig keys.
    # Heuristic: PatchTST configs include 'patch_size' and 'd_model' AND 'in_channels'.
    if all(k in model_cfg for k in ("in_channels", "num_symbols", "seq_len", "patch_size", "d_model")):
        # Respect saved config exactly (safer)
        cfg = CNNPatchTSTConfig(**model_cfg)
        model = CNNPatchTST(cfg).to(device)
        return model

    # Else try baseline path
    seq_len = int(train_args.get("seq_len", 64))
    model = make_baseline_model(num_features, num_symbols, seq_len, train_args).to(device)
    return model


@torch.no_grad()
def compute_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []
    for x, s, y in loader:
        x = x.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1, 1)
        yhat = model(x, s)
        preds.append(yhat.detach().cpu())
        targets.append(y.detach().cpu())

    yhat = torch.cat(preds, dim=0).squeeze(1)
    y = torch.cat(targets, dim=0).squeeze(1)

    mse = torch.mean((yhat - y) ** 2).item()
    mae = torch.mean(torch.abs(yhat - y)).item()
    dir_acc = torch.mean((torch.sign(yhat) == torch.sign(y)).float()).item()

    # Top-decile hit rate by |y|
    N = y.numel()
    if N >= 10:
        k = max(1, int(0.1 * N))
        idx = torch.topk(torch.abs(y), k=k, largest=True).indices
        top_hit = torch.mean((torch.sign(yhat[idx]) == torch.sign(y[idx])).float()).item()
    else:
        top_hit = float("nan")

    # Pearson correlation (useful for ranking quality)
    yc = y - y.mean()
    ph = yhat - yhat.mean()
    denom = (torch.sqrt((yc ** 2).sum()) * torch.sqrt((ph ** 2).sum()))
    corr = (float((yc * ph).sum() / denom) if denom.item() > 0 else 0.0)

    return {
        "mse": mse,
        "mae": mae,
        "dir_acc": dir_acc,
        "hit_rate_top_decile": top_hit,
        "pearson_r": corr,
        "n": float(N),
    }


# ------------------------- main -------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to .pt checkpoint")
    p.add_argument("--datasets-root", type=str, default="artifacts/datasets", help="Root where 'master' lives")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=None, help="Override seq_len for loader (usually read from ckpt)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", type=str, default=None, help="Where to write metrics JSON (defaults next to checkpoint)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, device)

    datasets_root = Path(args.datasets_root)
    master_dir = resolve_master_dir(datasets_root)

    # Infer dims
    num_features, num_symbols = infer_dims(master_dir)

    # Decide seq_len for the loader:
    saved_args = ckpt.get("train_args", {})
    seq_len = int(args.seq_len or saved_args.get("seq_len", 64))

    # Build loaders (test only is fine, but we'll reuse builder for consistency)
    loaders = build_dataloaders(
        master_dir=str(master_dir),
        seq_len=seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = loaders["test"]

    # Rebuild and load weights
    model = rebuild_model(ckpt, num_features, num_symbols, device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    # Compute metrics
    metrics = compute_metrics(model, test_loader, device)
    print(json.dumps(metrics, indent=2))

    # Save metrics JSON
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = ckpt_path.parent / "metrics_test.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {out_path}")

if __name__ == "__main__":
    main()

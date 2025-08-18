
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.training.data_module import build_datasets, build_loaders
from src.models.cnn_transformer import CNNTransformer, CNNTransformerConfig


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    datasets_root: str = "artifacts/datasets/master"
    out_dir: str = "artifacts/runs"
    seq_len: int = 64
    stride: int = 1
    batch_size: int = 256
    max_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    grad_clip_norm: float = 1.0
    amp: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    emb_dim: int = 8
    conv_channels: str = "64,64"
    kernel_sizes: str = "5,3"
    causal_conv: bool = False
    use_bn: bool = True
    dropout: float = 0.1
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    ff_ratio: float = 4.0
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    pooling: str = "last"


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train CNN+Transformer on intraday dataset")
    p.add_argument("--datasets-root", type=str, default=TrainConfig.datasets_root)
    p.add_argument("--out-dir", type=str, default=TrainConfig.out_dir)
    p.add_argument("--seq-len", type=int, default=TrainConfig.seq_len)
    p.add_argument("--stride", type=int, default=TrainConfig.stride)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--grad-clip-norm", type=float, default=TrainConfig.grad_clip_norm)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=TrainConfig.amp)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--device", type=str, default=TrainConfig.device)

    # model args
    p.add_argument("--emb-dim", type=int, default=TrainConfig.emb_dim)
    p.add_argument("--conv-channels", type=str, default=TrainConfig.conv_channels, help="comma list, e.g. 64,64")
    p.add_argument("--kernel-sizes", type=str, default=TrainConfig.kernel_sizes, help="comma list, e.g. 5,3")
    p.add_argument("--causal-conv", action="store_true")
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.set_defaults(use_bn=TrainConfig.use_bn)
    p.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    p.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    p.add_argument("--n-heads", type=int, default=TrainConfig.n_heads)
    p.add_argument("--num-layers", type=int, default=TrainConfig.num_layers)
    p.add_argument("--ff-ratio", type=float, default=TrainConfig.ff_ratio)
    p.add_argument("--attn-dropout", type=float, default=TrainConfig.attn_dropout)
    p.add_argument("--ffn-dropout", type=float, default=TrainConfig.ffn_dropout)
    p.add_argument("--pooling", type=str, default=TrainConfig.pooling, choices=["last", "mean"])

    args = p.parse_args()
    # Build TrainConfig
    cfg = TrainConfig(
        datasets_root=args.datasets_root,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        amp=args.amp,
        num_workers=args.num_workers,
        device=args.device,

        emb_dim=args.emb_dim,
        conv_channels=args.conv_channels,
        kernel_sizes=args.kernel_sizes,
        causal_conv=args.causal_conv,
        use_bn=args.use_bn,
        dropout=args.dropout,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        ff_ratio=args.ff_ratio,
        attn_dropout=args.attn_dropout,
        ffn_dropout=args.ffn_dropout,
        pooling=args.pooling,
    )
    return cfg


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=".").decode().strip()
    except Exception:
        return "nogit"


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    count = 0
    for x, s, y in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            y_hat = model(x.to(device), s.to(device))
            loss = torch.mean((y_hat - y.to(device)) ** 2)
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total += loss.item() * y.size(0)
        count += y.size(0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    for x, s, y in tqdm(loader, desc="val", leave=False):
        y_hat = model(x.to(device), s.to(device))
        loss = torch.mean((y_hat - y.to(device)) ** 2)
        total += loss.item() * y.size(0)
        count += y.size(0)
    return total / max(count, 1)


def main():
    cfg = parse_args()
    set_seed(42)
    device = torch.device(cfg.device)

    # datasets
    datasets = build_datasets(cfg.datasets_root, seq_len=cfg.seq_len, stride=cfg.stride, device=None)
    train_loader, val_loader, test_loader = build_loaders(
        datasets,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # model
    conv_channels = tuple(int(x) for x in str(cfg.conv_channels).split(",")) if cfg.conv_channels else tuple()
    kernel_sizes = tuple(int(x) for x in str(cfg.kernel_sizes).split(",")) if cfg.kernel_sizes else tuple()
    assert len(conv_channels) == len(kernel_sizes), "conv_channels and kernel_sizes must be same length"

    model_cfg = CNNTransformerConfig(
        in_channels=datasets.num_features,
        num_symbols=datasets.num_symbols,
        emb_dim=cfg.emb_dim,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        causal_conv=cfg.causal_conv,
        use_bn=cfg.use_bn,
        dropout=cfg.dropout,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        ff_ratio=cfg.ff_ratio,
        attn_dropout=cfg.attn_dropout,
        ffn_dropout=cfg.ffn_dropout,
        pooling=cfg.pooling,
    )
    model = CNNTransformer(model_cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)
    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # output dirs
    run_name = time.strftime("%Y%m%d-%H%M%S") + f"-{_git_sha()}"
    out_dir = Path(cfg.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # save config
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(asdict(model_cfg), f, indent=2)

    # metrics CSV
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["epoch", "train_mse", "val_mse", "lr"])

    best_val = float("inf")
    epochs_no_improve = 0
    best_path = out_dir / "checkpoints" / "best.pt"

    for epoch in range(1, cfg.max_epochs + 1):
        tr = train_loader
        train_mse = train_one_epoch(model, tr, optimizer, scaler, device)
        val_mse = None
        if val_loader is not None:
            val_mse = evaluate(model, val_loader, device)
            scheduler.step(val_mse)
        else:
            val_mse = train_mse

        # log
        with open(csv_path, "a", newline="") as f:
            cw = csv.writer(f); cw.writerow([epoch, train_mse, val_mse, optimizer.param_groups[0]["lr"]])

        # early stopping
        if val_mse < best_val - 1e-8:
            best_val = val_mse
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "model_cfg": asdict(model_cfg)}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch}. Best val={best_val:.6f}")
                break

    print(f"Best checkpoint: {best_path}")
    print(f"Metrics logged to: {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- Your existing datamodule (assumed present from earlier work) ----
# WindowedDataset must yield: x: (B, C, L), sym_id: (B,), y: (B,)
try:
    from .data_module import build_dataloaders, WindowedDataset
except Exception as e:
    print("[train] Could not import data_module. Ensure src/training/data_module.py exists.", file=sys.stderr)
    raise

# ---- New PatchTST hybrid model ----
from src.models.cnn_patchtst import CNNPatchTST, CNNPatchTSTConfig

# ---- Optional: baseline model (keep for A/B). Safe import inside branch. ----
def make_baseline_model(in_channels: int, num_symbols: int, seq_len: int, args) -> nn.Module:
    """
    Constructs the previous baseline CNN+Transformer model if file is present.
    """
    try:
        from src.models.cnn_transformer import CNNTransformer, CNNTransformerConfig  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Baseline model requested (--arch baseline) but src/models/cnn_transformer.py "
            "was not found or failed to import. Either keep that file or use --arch patchtst."
        ) from e

    cfg = CNNTransformerConfig(
        in_channels=in_channels,
        num_symbols=num_symbols,
        seq_len=seq_len,
        emb_dim=args.emb_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        conv_channels=args.cnn_hidden,
        kernel_sizes=tuple(map(int, args.cnn_kernels.split(","))),
    )
    return CNNTransformer(cfg)

# ---- Helpers ----

def infer_dims(master_dir: Path) -> Tuple[int, int]:
    """
    Infer (num_features, num_symbols) from the master/train.npz split.
    """
    train_npz = master_dir / "train.npz"
    if not train_npz.exists():
        raise FileNotFoundError(f"Could not find {train_npz}. Did you run the master builder?")
    with np.load(train_npz) as d:
        X = d["X"]   # (N, F)
        sym = d["sym_id"]  # (N,)
        F = X.shape[1]
        num_symbols = int(np.max(sym)) + 1 if sym.size > 0 else 1
    return F, num_symbols


def resolve_master_dir(datasets_root: Path) -> Path:
    """
    Resolve artifacts/datasets/master symlink/pointer to the actual version directory.
    """
    master_ptr = datasets_root / "master"
    if not master_ptr.exists():
        raise FileNotFoundError(f"Missing pointer {master_ptr}.")
    # If it's a symlink, readlink; if it's a file containing the path; else treat as directory
    if master_ptr.is_symlink():
        return master_ptr.resolve()
    if master_ptr.is_file():
        txt = master_ptr.read_text().strip()
        p = (datasets_root / txt).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Master pointer file points to missing path: {p}")
        return p
    # directory
    return master_ptr


def make_run_dir(artifacts_root: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_root / "runs" / ts
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, train_args: Dict[str, Any], model_cfg: Dict[str, Any]) -> None:
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "configs" / "train_config.json", "w") as f:
        json.dump(train_args, f, indent=2)
    with open(run_dir / "configs" / "model_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)


def huber_loss(delta: float) -> nn.Module:
    return nn.HuberLoss(delta=delta)


# ---- Training / Eval Loops ----

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> Tuple[float, float, float]:
    model.eval()
    losses, maes, dir_hits, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        x, s, y = batch  # x: (B,C,L), s: (B,), y: (B,)
        x = x.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1, 1)

        yhat = model(x, s)  # (B,1)
        loss = loss_fn(yhat, y)
        mae = torch.mean(torch.abs(yhat - y))
        dir_acc = torch.mean((torch.sign(yhat) == torch.sign(y)).float())

        bsz = y.size(0)
        losses += loss.item() * bsz
        maes += mae.item() * bsz
        dir_hits += dir_acc.item() * bsz
        n += bsz
    return losses / n, maes / n, dir_hits / n


def train_one_epoch(model, loader, device, loss_fn, optimizer, scaler, grad_clip):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        x, s, y = batch
        x = x.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            yhat = model(x, s)
            loss = loss_fn(yhat, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


def main():
    parser = argparse.ArgumentParser(description="Train CNN+Transformer for intraday returns")

    # Data / windowing
    parser.add_argument("--datasets-root", type=str, default="artifacts/datasets",
                        help="Root where 'master' pointer lives")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    # Arch selection
    parser.add_argument("--arch", type=str, choices=["patchtst", "baseline"], default="patchtst")

    # CNN (stem)
    parser.add_argument("--cnn-hidden", type=int, default=128)
    parser.add_argument("--cnn-blocks", type=int, default=3)
    parser.add_argument("--cnn-kernels", type=str, default="5,5,3")
    parser.add_argument("--cnn-dilations", type=str, default="1,2,4")
    parser.add_argument("--cnn-dropout", type=float, default=0.1)

    # Tokenizer / Transformer (time)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--patch-stride", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Mixer & embeddings
    parser.add_argument("--channel-mixer", type=str, choices=["mlp", "ivar", "none"], default="mlp")
    parser.add_argument("--emb-dim", type=int, default=12)

    # Optim / train
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision (AMP)")

    # Loss
    parser.add_argument("--loss", type=str, choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.01)

    # Output
    parser.add_argument("--artifacts-root", type=str, default="artifacts", help="Where runs/ and checkpoints/ go")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets_root = Path(args.datasets_root)
    master_dir = resolve_master_dir(datasets_root)
    artifacts_root = Path(args.artifacts_root)
    run_dir = make_run_dir(artifacts_root)

    # Infer dims from master/train.npz
    num_features, num_symbols = infer_dims(master_dir)

    # Build loaders
    loaders = build_dataloaders(
        master_dir=str(master_dir),
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # your WindowedDataset is expected to avoid crossing symbol boundaries
    )
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

    # Build model
    if args.arch == "patchtst":
        cfg = CNNPatchTSTConfig(
            in_channels=num_features,
            num_symbols=num_symbols,
            seq_len=args.seq_len,
            cnn_hidden=args.cnn_hidden,
            cnn_blocks=args.cnn_blocks,
            cnn_kernels=tuple(map(int, args.cnn_kernels.split(","))),
            cnn_dilations=tuple(map(int, args.cnn_dilations.split(","))),
            cnn_dropout=args.cnn_dropout,
            channel_mixer=args.channel_mixer,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.num_layers,
            trf_dropout=args.dropout,
            emb_dim=args.emb_dim,
            pool="last",
        )
        model = CNNPatchTST(cfg)
        model_cfg = model.config_dict()
    else:
        model = make_baseline_model(num_features, num_symbols, args.seq_len, args)
        # Best effort: store a light model config
        model_cfg = {"arch": "baseline", "seq_len": args.seq_len, "d_model": args.d_model,
                     "n_heads": args.n_heads, "num_layers": args.num_layers, "emb_dim": args.emb_dim}

    model.to(device)

    # Loss
    loss_fn = huber_loss(args.huber_delta) if args.loss == "huber" else nn.MSELoss()

    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use-amp and device.type == "cuda")

    # Save configs
    save_config(
        run_dir,
        train_args=vars(args),
        model_cfg=model_cfg,
    )

    # Train
    best_val = float("inf")
    best_path = run_dir / "checkpoints" / "best.pt"
    epochs_no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, loss_fn, optimizer, scaler, args.grad_clip)
        val_loss, val_mae, val_dir = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)

        print(f"[{epoch:03d}/{args.max_epochs}] "
              f"train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}  val_mae={val_mae:.6f}  val_dir={val_dir:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping & checkpointing
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_cfg": model_cfg,
                    "train_args": vars(args),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    # Final test with best checkpoint
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        if args.arch == "patchtst":
            # reconstruct exactly from saved cfg (safer if you later change flags)
            saved_cfg = CNNPatchTSTConfig(**ckpt["model_cfg"])
            model = CNNPatchTST(saved_cfg).to(device)
        else:
            # Baseline reconstruction best-effort (or you can store its cfg similarly)
            model = make_baseline_model(num_features, num_symbols, args.seq_len, args).to(device)
        model.load_state_dict(ckpt["model_state"])
        test_loss, test_mae, test_dir = evaluate(model, test_loader, device, loss_fn)
        print(f"[TEST] loss={test_loss:.6f}  mae={test_mae:.6f}  dir_acc={test_dir:.4f}")
        with open(run_dir / "metrics_test.json", "w") as f:
            json.dump({"loss": test_loss, "mae": test_mae, "dir_acc": test_dir}, f, indent=2)
    else:
        print("No best checkpoint found; skipping test eval.")

if __name__ == "__main__":
    main()

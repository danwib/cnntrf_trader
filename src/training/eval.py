
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from src.training.data_module import build_datasets, build_loaders
from src.models.cnn_transformer import CNNTransformer, CNNTransformerConfig


def load_model(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = CNNTransformerConfig(**ckpt["model_cfg"])
    model = CNNTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def compute_metrics(model, loader, device) -> Dict[str, float]:
    y_true = []
    y_pred = []
    for x, s, y in tqdm(loader, desc="eval"):
        y_hat = model(x.to(device), s.to(device))
        y_true.append(y.numpy())
        y_pred.append(y_hat.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    # hit rate in top decile |y|
    abs_y = np.abs(y_true)
    thr = np.quantile(abs_y, 0.9) if len(abs_y) > 0 else 0.0
    mask = abs_y >= thr
    hit_rate_top_decile = float(np.mean((np.sign(y_true[mask]) == np.sign(y_pred[mask])))) if mask.any() else float('nan')

    return {
        "mse": mse,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "hit_rate_top_decile": hit_rate_top_decile,
        "n_samples": int(len(y_true)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", type=str, default="artifacts/datasets/master")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, default="artifacts/metrics")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    datasets = build_datasets(args.datasets_root, seq_len=args.seq_len, stride=args.stride, device=None)
    _, _, test_loader = build_loaders(datasets, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    if test_loader is None:
        raise SystemExit("No test split found. Ensure artifacts/datasets/master/test.npz exists.")

    model = load_model(args.checkpoint, device)
    metrics = compute_metrics(model, test_loader, device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

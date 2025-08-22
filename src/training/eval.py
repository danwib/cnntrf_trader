#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

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


def read_meta(master_dir: Path) -> Dict[str, Any]:
    meta_path = master_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    # Fallback empty
    return {}


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def make_baseline_model(in_channels: int, num_symbols: int, seq_len: int, saved_args: Dict[str, Any]) -> nn.Module:
    """Reconstruct the older baseline CNN+Transformer if still present."""
    try:
        from src.models.cnn_transformer import CNNTransformer, CNNTransformerConfig  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Checkpoint indicates a 'baseline' model, but src/models/cnn_transformer.py "
            "is missing. Restore that file or evaluate a PatchTST checkpoint."
        ) from e

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
    """Rebuild the exact model class from checkpoint metadata."""
    model_cfg = ckpt.get("model_cfg", {})
    train_args = ckpt.get("train_args", {})
    if all(k in model_cfg for k in ("in_channels", "num_symbols", "seq_len", "patch_size", "d_model")):
        cfg = CNNPatchTSTConfig(**model_cfg)
        model = CNNPatchTST(cfg).to(device)
        return model
    # fallback to baseline
    seq_len = int(train_args.get("seq_len", 64))
    model = make_baseline_model(num_features, num_symbols, seq_len, train_args).to(device)
    return model


@torch.no_grad()
def gather_preds(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    """Collect predictions, targets, and symbol ids."""
    model.eval()
    preds, targets, syms = [], [], []
    for x, s, y in loader:
        x = x.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1, 1)
        yhat = model(x, s)
        preds.append(yhat.detach().cpu())
        targets.append(y.detach().cpu())
        syms.append(s.detach().cpu())
    yhat = torch.cat(preds, dim=0).squeeze(1)
    y = torch.cat(targets, dim=0).squeeze(1)
    sym = torch.cat(syms, dim=0).long()
    return {"yhat": yhat, "y": y, "sym": sym}


def core_metrics(yhat: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((yhat - y) ** 2).item()
    mae = torch.mean(torch.abs(yhat - y)).item()
    dir_acc = torch.mean((torch.sign(yhat) == torch.sign(y)).float()).item()
    # top-decile hit rate by |y|
    N = y.numel()
    if N >= 10:
        k = max(1, int(0.1 * N))
        idx = torch.topk(torch.abs(y), k=k, largest=True).indices
        top_hit = torch.mean((torch.sign(yhat[idx]) == torch.sign(y[idx])).float()).item()
    else:
        top_hit = float("nan")
    # Pearson r
    yc = y - y.mean()
    ph = yhat - yhat.mean()
    denom = (torch.sqrt((yc ** 2).sum()) * torch.sqrt((ph ** 2).sum()))
    corr = (float((yc * ph).sum() / denom) if denom.item() > 0 else 0.0)
    return {"mse": mse, "mae": mae, "dir_acc": dir_acc, "hit_rate_top_decile": top_hit, "pearson_r": corr, "n": float(N)}


def baselines(y: torch.Tensor) -> Dict[str, float]:
    """Simple context baselines."""
    p_up = torch.mean((y > 0).float()).item()
    majority = max(p_up, 1 - p_up)  # always predict the majority sign
    return {"coin_flip": 0.5, "majority_sign": majority, "p_up": p_up}


def thresholded_expectancy(
    yhat: torch.Tensor,
    y: torch.Tensor,
    top_pcts: List[int],
    cost_bps: float = 2.0,
) -> Dict[str, Dict[str, float]]:
    """
    Trade only when |yhat| is in top X% of scores.
    PnL per trade (log-return approximation): long if yhat>0 -> +y; short if yhat<0 -> -y.
    Net subtracts a fixed round-trip cost in bps (bps/1e4).
    """
    out: Dict[str, Dict[str, float]] = {}
    abs_yhat = torch.abs(yhat)
    N = y.shape[0]
    cost = cost_bps / 1e4  # convert bp to decimal return

    for pct in top_pcts:
        q = 1.0 - (pct / 100.0)
        tau = torch.quantile(abs_yhat, q) if N > 0 else torch.tensor(float("inf"))
        sel = abs_yhat >= tau
        n_sel = int(sel.sum().item())
        coverage = n_sel / max(N, 1)

        if n_sel == 0:
            out[str(pct)] = {"coverage": 0.0, "trades": 0.0, "gross_bps": float("nan"),
                             "net_bps": float("nan"), "hit_rate": float("nan")}
            continue

        gross = (torch.sign(yhat[sel]) * y[sel]).mean().item()
        net = gross - cost  # pay cost every round-trip
        hit = torch.mean((torch.sign(yhat[sel]) == torch.sign(y[sel])).float()).item()

        out[str(pct)] = {
            "coverage": coverage,
            "trades": float(n_sel),
            "gross_bps": gross * 1e4,
            "net_bps": net * 1e4,
            "hit_rate": hit,
        }
    return out


def per_symbol_metrics(yhat: torch.Tensor, y: torch.Tensor, sym: torch.Tensor) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for s in torch.unique(sym).tolist():
        mask = (sym == s)
        ys, yhs = y[mask], yhat[mask]
        if ys.numel() == 0:
            continue
        m = core_metrics(yhs, ys)
        out[int(s)] = {"n": m["n"], "mse": m["mse"], "mae": m["mae"], "dir_acc": m["dir_acc"]}
    return out


def write_per_symbol_csv(path: Path, per_sym: Dict[int, Dict[str, float]], symmap: Optional[Dict[int, str]] = None) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["sym_id", "symbol", "n", "mse", "mae", "dir_acc"]
        w.writerow(header)
        for sid, stats in sorted(per_sym.items()):
            sym = symmap.get(sid, "") if symmap else ""
            w.writerow([sid, sym, int(stats["n"]), stats["mse"], stats["mae"], stats["dir_acc"]])


# ------------------------- main -------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split with richer metrics")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to .pt checkpoint")
    p.add_argument("--datasets-root", type=str, default="artifacts/datasets", help="Root where 'master' lives")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=None, help="Override seq_len for loader (usually read from ckpt)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", type=str, default=None, help="Where to write metrics JSON (defaults next to checkpoint)")
    # New options for expectancy
    p.add_argument("--cost-bps", type=float, default=2.0, help="Round-trip cost per trade, in bps")
    p.add_argument("--trade-top-pcts", type=str, default="10,20,30,40",
                   help="Comma list of top-|ŷ| percent thresholds to evaluate (e.g., 10,30,50)")
    p.add_argument("--per-symbol-csv", type=str, default=None, help="Optional CSV path for per-symbol metrics")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path, device)

    datasets_root = Path(args.datasets_root)
    master_dir = resolve_master_dir(datasets_root)
    meta = read_meta(master_dir)

    # Infer dims
    num_features, num_symbols = infer_dims(master_dir)

    # Decide seq_len for the loader:
    saved_args = ckpt.get("train_args", {})
    seq_len = int(args.seq_len or saved_args.get("seq_len", 64))

    # Build loaders (test only is fine)
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

    # Compute predictions
    bundle = gather_preds(model, test_loader, device)
    yhat, y, sym = bundle["yhat"], bundle["y"], bundle["sym"]

    # Core metrics
    metrics = core_metrics(yhat, y)

    # Baselines
    base = baselines(y)
    metrics["baseline_coin_flip"] = base["coin_flip"]
    metrics["baseline_majority_sign"] = base["majority_sign"]
    metrics["p_up"] = base["p_up"]

    # Thresholded, costed expectancy
    top_pcts = [int(s.strip()) for s in args.trade_top_pcts.split(",") if s.strip()]
    exp = thresholded_expectancy(yhat, y, top_pcts=top_pcts, cost_bps=args.cost_bps)

    # Per-symbol breakdown (optional CSV)
    per_sym = per_symbol_metrics(yhat, y, sym)
    symmap = None
    if "sym2id" in meta and isinstance(meta["sym2id"], dict):
        # meta["sym2id"] maps symbol->id; invert it
        symmap = {int(v): k for k, v in meta["sym2id"].items()}
    if args.per_symbol_csv:
        write_per_symbol_csv(Path(args.per_symbol_csv), per_sym, symmap)

    # Assemble report
    report = {
        "checkpoint": str(ckpt_path),
        "master_dir": str(master_dir),
        "seq_len": seq_len,
        "n_test": int(metrics["n"]),
        "metrics": {
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "dir_acc": metrics["dir_acc"],
            "hit_rate_top_decile": metrics["hit_rate_top_decile"],
            "pearson_r": metrics["pearson_r"],
        },
        "baselines": {
            "coin_flip": metrics["baseline_coin_flip"],
            "majority_sign": metrics["baseline_majority_sign"],
            "p_up": metrics["p_up"],
        },
        "expectancy": exp,
        "per_symbol_summary": {
            "count": len(per_sym),
            "avg_dir_acc": float(np.mean([v["dir_acc"] for v in per_sym.values()])) if per_sym else float("nan"),
        },
    }

    # Save metrics JSON
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = ckpt_path.parent / "metrics_test_rich.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote metrics to {out_path}")

    # Optional CSV note
    if args.per_symbol_csv:
        print(f"Wrote per-symbol metrics to {args.per_symbol_csv}")


if __name__ == "__main__":
    main()

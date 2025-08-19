#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from src.training.data_module import build_dataloaders
from src.models.cnn_patchtst import CNNPatchTST, CNNPatchTSTConfig
from src.inference.decision import (
    select_tau_by_grid_from_loader,
    evaluate_loader_cost_aware,
)

def main():
    p = argparse.ArgumentParser("Select tau on val (cost-aware), evaluate on test, and save results.")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--datasets-root", type=str, default="artifacts/datasets")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)

    # trading assumptions
    p.add_argument("--horizon", type=int, default=4, help="label horizon (bars) e.g. y_4 -> 4")
    p.add_argument("--cost-bps", type=float, default=3.0, help="round-trip cost in bps (fees+spread)")
    p.add_argument("--grid", choices=["sigma", "percentile"], default="sigma")
    p.add_argument("--min-trades", type=int, default=200)

    # output
    p.add_argument("--out", type=str, default=None, help="JSON path for threshold + PnL metrics")

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint & rebuild model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = CNNPatchTSTConfig(**ckpt["model_cfg"])
    model = CNNPatchTST(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Build loaders (val/test have shuffle=False in your datamodule)
    loaders = build_dataloaders(
        master_dir=f"{args.datasets_root}/master",
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader, test_loader = loaders["val"], loaders["test"]
    assert val_loader is not None and test_loader is not None, "Need val & test splits."

    # Fit tau on validation
    best_tau, val_best, grid = select_tau_by_grid_from_loader(
        model, val_loader, device=device,
        horizon=args.horizon, cost_bps=args.cost_bps,
        grid=args.grid, min_trades=args.min_trades
    )

    # Evaluate on test with that tau
    test_metrics = evaluate_loader_cost_aware(
        model, test_loader, device=device,
        tau=best_tau, horizon=args.horizon, cost_bps=args.cost_bps
    )

    result = {
        "checkpoint": args.checkpoint,
        "horizon": args.horizon,
        "cost_bps": args.cost_bps,
        "grid": args.grid,
        "best_tau": best_tau,
        "val_metrics": vars(val_best),
        "test_metrics": vars(test_metrics),
    }

    # default output next to checkpoint
    out_path = Path(args.out) if args.out else (Path(args.checkpoint).parent / "threshold_fit.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ok] wrote {out_path}")
    print(f"Chosen tau: {best_tau:.6g}")
    print(f"Val PnL sum: {val_best.pnl_sum:.6g}  trades: {val_best.trades}  win_rate: {val_best.win_rate:.3f}")
    print(f"Test PnL sum: {test_metrics.pnl_sum:.6g}  trades: {test_metrics.trades}  win_rate: {test_metrics.win_rate:.3f}")

if __name__ == "__main__":
    main()


'''How to run:
python -m src.inference.fit_threshold \
  --checkpoint artifacts/runs/<timestamp>/checkpoints/best.pt \
  --datasets-root artifacts/datasets \
  --seq-len 64 --stride 1 \
  --horizon 4 --cost-bps 3.0 --grid sigma --min-trades 200
'''
# src/inference/decision.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import math
import numpy as np
import torch
from torch.utils.data import DataLoader

# -----------------------------
# Core simulation (regression)
# -----------------------------

@dataclass
class PnLMetrics:
    tau: float
    cost_bps: float
    horizon: int
    trades: int
    wins: int
    win_rate: float
    pnl_sum: float
    pnl_mean: float
    pnl_std: float
    sharpe_like: float
    long_share: float
    short_share: float

def _simulate_trades_regression(
    yhat: np.ndarray,     # (N,)
    ytrue: np.ndarray,    # (N,)
    sym: np.ndarray,      # (N,)
    horizon: int,
    tau: float,
    cost_bps: float,
) -> PnLMetrics:
    """
    Enter a position when |yhat| > tau; hold exactly `horizon` bars (non-overlapping per symbol).
    Realized per-trade PnL = sign(yhat)*ytrue - roundtrip_cost.
    Assumes ytrue is the aligned future (log) return for that window.
    """
    assert yhat.shape == ytrue.shape == sym.shape
    N = yhat.shape[0]
    # cost per round-trip in "return" units
    roundtrip_cost = 2.0 * (cost_bps * 1e-4)

    trades = 0
    wins = 0
    pnl_list: List[float] = []
    longs = shorts = 0

    # cooldown per symbol to avoid overlapping trades
    next_idx_allowed: Dict[int, int] = {}

    for i in range(N):
        s = int(sym[i])
        if next_idx_allowed.get(s, -1) > i:
            continue  # still cooling down for this symbol

        score = yhat[i]
        if abs(score) <= tau:
            continue

        sign = 1.0 if score > 0 else -1.0
        pnl = sign * float(ytrue[i]) - roundtrip_cost

        pnl_list.append(pnl)
        trades += 1
        if pnl > 0:
            wins += 1
        if sign > 0: longs += 1
        else: shorts += 1

        # block the next (horizon-1) windows for this symbol
        next_idx_allowed[s] = i + horizon

    if trades == 0:
        return PnLMetrics(
            tau=tau, cost_bps=cost_bps, horizon=horizon, trades=0, wins=0,
            win_rate=float("nan"), pnl_sum=0.0, pnl_mean=float("nan"),
            pnl_std=float("nan"), sharpe_like=float("nan"),
            long_share=float("nan"), short_share=float("nan"),
        )

    pnl_arr = np.asarray(pnl_list, dtype=np.float64)
    pnl_mean = float(pnl_arr.mean())
    pnl_std  = float(pnl_arr.std(ddof=1)) if trades > 1 else 0.0
    sharpe_like = (pnl_mean / pnl_std) * math.sqrt(252.0) if pnl_std > 0 else float("nan")
    win_rate = wins / trades
    long_share = longs / trades
    short_share = shorts / trades

    return PnLMetrics(
        tau=tau, cost_bps=cost_bps, horizon=horizon,
        trades=trades, wins=wins, win_rate=win_rate,
        pnl_sum=float(pnl_arr.sum()), pnl_mean=pnl_mean, pnl_std=pnl_std,
        sharpe_like=sharpe_like, long_share=long_share, short_share=short_share,
    )

# ---------------------------------
# Threshold selection (validation)
# ---------------------------------

def select_tau_by_grid_from_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    horizon: int = 4,
    cost_bps: float = 3.0,
    grid: str = "sigma",                 # "sigma" or "percentile"
    mults: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    percentiles: Tuple[float, ...] = (0, 50, 70, 80, 85, 90, 95, 98),
    min_trades: int = 200,               # avoid picking silly high taus
) -> Tuple[float, PnLMetrics, Dict[float, PnLMetrics]]:
    """
    Runs the model on a validation loader (shuffle=False), grids tau, and returns:
      (best_tau, best_metrics, metrics_by_tau)
    """
    model.eval()
    preds, targets, syms = [], [], []
    with torch.no_grad():
        for x, s, y in loader:
            x = x.to(device, non_blocking=True)
            s = s.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            # model may return (yhat) or (yhat, logits). Handle both.
            out = model(x, s)
            if isinstance(out, (tuple, list)):
                yhat = out[0]
            else:
                yhat = out
            preds.append(yhat.squeeze(1).detach().cpu())
            targets.append(y.detach().cpu())
            syms.append(s.detach().cpu())
    yhat = torch.cat(preds).numpy()
    y    = torch.cat(targets).numpy()
    sym  = torch.cat(syms).numpy()

    # Build candidate taus
    taus: List[float] = []
    if grid == "sigma":
        sigma = float(np.std(yhat))
        taus = [m * sigma for m in mults]
    elif grid == "percentile":
        abs_yhat = np.abs(yhat)
        taus = [float(np.percentile(abs_yhat, p)) for p in percentiles]
    else:
        raise ValueError("grid must be 'sigma' or 'percentile'")

    metrics_by_tau: Dict[float, PnLMetrics] = {}
    best_tau = None
    best_score = -1e18
    best_metrics: Optional[PnLMetrics] = None

    for tau in taus:
        m = _simulate_trades_regression(yhat, y, sym, horizon=horizon, tau=tau, cost_bps=cost_bps)
        metrics_by_tau[tau] = m
        # choose by total PnL, but require a minimum trade count
        score = (m.pnl_sum if m.trades >= min_trades else -1e18)
        if score > best_score:
            best_score = score
            best_tau = tau
            best_metrics = m

    assert best_metrics is not None and best_tau is not None
    return best_tau, best_metrics, metrics_by_tau

# ---------------------------------
# Evaluate on test with chosen tau
# ---------------------------------

@torch.no_grad()
def evaluate_loader_cost_aware(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tau: float,
    horizon: int = 4,
    cost_bps: float = 3.0,
) -> PnLMetrics:
    model.eval()
    preds, targets, syms = [], [], []
    for x, s, y in loader:
        x = x.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        out = model(x, s)
        if isinstance(out, (tuple, list)):
            yhat = out[0]
        else:
            yhat = out
        preds.append(yhat.squeeze(1).detach().cpu())
        targets.append(y.detach().cpu())
        syms.append(s.detach().cpu())
    yhat = torch.cat(preds).numpy()
    y    = torch.cat(targets).numpy()
    sym  = torch.cat(syms).numpy()
    return _simulate_trades_regression(yhat, y, sym, horizon=horizon, tau=tau, cost_bps=cost_bps)

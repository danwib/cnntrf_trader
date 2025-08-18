
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WindowedDataset(Dataset):
    """
    Builds fixed-length sequences from flat arrays (X, y, sym_id).
    Ensures windows do not cross symbol boundaries.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sym_id: np.ndarray,
        seq_len: int = 64,
        stride: int = 1,
        device: Optional[torch.device] = None,
        return_numpy: bool = False,
    ) -> None:
        assert X.ndim == 2, "X must be (N, F)"
        assert y.ndim == 1 and y.shape[0] == X.shape[0], "y must be (N,) and align with X"
        assert sym_id.ndim == 1 and sym_id.shape[0] == X.shape[0], "sym_id must be (N,) and align with X"
        assert seq_len > 0

        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.sym_id = sym_id.astype(np.int32, copy=False)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.device = device
        self.return_numpy = return_numpy

        # Build index lists per symbol
        self._by_sym: Dict[int, np.ndarray] = {}
        for s in np.unique(self.sym_id):
            idxs = np.where(self.sym_id == s)[0]
            # keep original order (assumed to be time ascending)
            self._by_sym[int(s)] = idxs

        # Catalogue windows as (sym, start_pos_in_that_symbol_index_array)
        self._windows: List[Tuple[int, int]] = []
        for s, idxs in self._by_sym.items():
            if len(idxs) < self.seq_len:
                continue
            # slide with given stride
            max_start = len(idxs) - self.seq_len
            starts = range(0, max_start + 1, self.stride)
            self._windows.extend((s, st) for st in starts)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, i: int):
        s, st = self._windows[i]
        idxs = self._by_sym[s][st : st + self.seq_len]
        x = self.X[idxs]                  # (L, F)
        # Channel-first: (F, L)
        x = np.ascontiguousarray(x.T)
        target = self.y[idxs[-1]]
        sym = int(s)
        if self.return_numpy:
            return x, sym, target
        x_t = torch.from_numpy(x)
        y_t = torch.tensor(target, dtype=torch.float32)
        s_t = torch.tensor(sym, dtype=torch.long)
        if self.device is not None:
            x_t = x_t.to(self.device, non_blocking=True)
            y_t = y_t.to(self.device, non_blocking=True)
            s_t = s_t.to(self.device, non_blocking=True)
        return x_t, s_t, y_t


@dataclass
class Datasets:
    train: WindowedDataset
    val: Optional[WindowedDataset]
    test: Optional[WindowedDataset]
    num_symbols: int
    num_features: int


def _load_split(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.load(path)
    X = arr["X"]
    y = arr["y"]
    sid = arr["sym_id"]
    return X, y, sid


def build_datasets(
    root: str | Path = "artifacts/datasets/master",
    seq_len: int = 64,
    stride: int = 1,
    device: Optional[torch.device] = None,
) -> Datasets:
    root = Path(root).resolve()
    train_p = root / "train.npz"
    val_p = root / "val.npz"
    test_p = root / "test.npz"
    assert train_p.exists(), f"Missing {train_p}. Did you run scripts/auto_build_dataset.py --mode master ?"

    Xtr, ytr, sid_tr = _load_split(train_p)
    Xval = yval = sid_val = None
    Xte = yte = sid_te = None

    num_features = Xtr.shape[1]
    # try to compute num_symbols from all splits we have
    sid_all = [sid_tr]
    if val_p.exists():
        Xval, yval, sid_val = _load_split(val_p)
        sid_all.append(sid_val)
    if test_p.exists():
        Xte, yte, sid_te = _load_split(test_p)
        sid_all.append(sid_te)
    num_symbols = int(np.concatenate(sid_all).max()) + 1

    ds_train = WindowedDataset(Xtr, ytr, sid_tr, seq_len=seq_len, stride=stride, device=device)
    ds_val = WindowedDataset(Xval, yval, sid_val, seq_len=seq_len, stride=stride, device=device) if Xval is not None else None
    ds_test = WindowedDataset(Xte, yte, sid_te, seq_len=seq_len, stride=stride, device=device) if Xte is not None else None
    return Datasets(train=ds_train, val=ds_val, test=ds_test, num_symbols=num_symbols, num_features=num_features)


def build_loaders(
    datasets: Datasets,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    train_loader = DataLoader(datasets.train, shuffle=True, **kwargs)
    val_loader = DataLoader(datasets.val, shuffle=False, **kwargs) if datasets.val is not None else None
    test_loader = DataLoader(datasets.test, shuffle=False, **kwargs) if datasets.test is not None else None
    return train_loader, val_loader, test_loader

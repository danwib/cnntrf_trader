# src/data/feature_pipeline.py
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def engineer_basic_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Expects OHLCV columns and UTC DatetimeIndex.
    Produces: returns, log-returns, rolling volatilities, RSI, volume z-score.
    All features are causal (past-only).
    """
    out = df.copy()

    # simple returns / log returns
    out["ret_1"] = out[price_col].pct_change()
    out["logret_1"] = np.log(out[price_col]).diff()

    # realized volatility on the base stream (causal rolling)
    # If base is 15m bars: 15->~3.75h, 60->~15h windows
    out["rv_15"] = out["logret_1"].rolling(15, min_periods=5).std(ddof=0)
    out["rv_60"] = out["logret_1"].rolling(60, min_periods=20).std(ddof=0)

    # momentum (EWMA-based RSI)
    out["rsi_14"] = rsi(out[price_col], 14)

    # volume z-score on trailing window (causal rolling)
    vol_mean = out["volume"].rolling(30, min_periods=5).mean()
    vol_std = out["volume"].rolling(30, min_periods=5).std(ddof=0)
    out["vol_z"] = (out["volume"] - vol_mean) / (vol_std + 1e-9)

    # drop rows that lack enough history for the first features
    out = out.dropna()
    return out


class FeaturePipeline:
    """
    Keeps feature scaling consistent between training and serving.

    IMPORTANT:
    - Call `fit(...)` or `fit_transform(..., train_mask=...)` on TRAIN rows only to avoid leakage.
    - At inference, call `transform(...)` with the same feature columns order.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "target",
        scaler: Optional[StandardScaler] = None,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = scaler if scaler is not None else StandardScaler()

    def _infer_feature_cols(self, df: pd.DataFrame) -> List[str]:
        # numeric columns excluding the target
        return [c for c in df.columns if c != self.target_col and pd.api.types.is_numeric_dtype(df[c])]

    def fit(self, df: pd.DataFrame, train_mask: Optional[pd.Series] = None) -> None:
        if self.feature_cols is None:
            self.feature_cols = self._infer_feature_cols(df)

        if train_mask is None:
            # SAFE DEFAULT: if no mask is provided, assume df already contains only TRAIN rows.
            X_train = df[self.feature_cols].values
        else:
            X_train = df.loc[train_mask, self.feature_cols].values

        # Fit scaler on TRAIN ONLY to avoid leakage
        self.scaler.fit(X_train)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # ensure required columns exist and order is consistent
        assert self.feature_cols is not None, "FeaturePipeline must be fit before transform (feature_cols unknown)."
        return self.scaler.transform(df[self.feature_cols].values)

    def fit_transform(
        self,
        df: pd.DataFrame,
        train_mask: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Fit on TRAIN rows (mask) and return scaled X for ALL rows in df to keep alignment with y if present.
        If the target column exists in df, returns y as well; otherwise y = None.
        """
        if self.feature_cols is None:
            self.feature_cols = self._infer_feature_cols(df)

        if train_mask is None:
            X_train = df[self.feature_cols].values
        else:
            X_train = df.loc[train_mask, self.feature_cols].values

        self.scaler.fit(X_train)

        X_scaled = self.scaler.transform(df[self.feature_cols].values)
        y = df[self.target_col].values if self.target_col in df.columns else None

        meta = {
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "scaler_mean_": self.scaler.mean_.tolist(),
            "scaler_scale_": self.scaler.scale_.tolist(),
        }
        return X_scaled, y, meta

    @classmethod
    def from_meta(cls, meta: Dict) -> "FeaturePipeline":
        obj = cls(feature_cols=meta["feature_cols"], target_col=meta["target_col"])
        # Manually set fitted attributes for sklearn compatibility
        obj.scaler.mean_ = np.array(meta["scaler_mean_"], dtype=float)
        obj.scaler.scale_ = np.array(meta["scaler_scale_"], dtype=float)
        obj.scaler.n_features_in_ = len(meta["feature_cols"])
        return obj

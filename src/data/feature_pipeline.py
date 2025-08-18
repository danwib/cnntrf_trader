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
    """
    out = df.copy()
    out["ret_1"] = out[price_col].pct_change()
    out["logret_1"] = np.log(out[price_col]).diff()
    out["rv_15"] = out["logret_1"].rolling(15, min_periods=5).std()
    out["rv_60"] = out["logret_1"].rolling(60, min_periods=20).std()
    out["rsi_14"] = rsi(out[price_col], 14)
    out["vol_z"] = (
        (out["volume"] - out["volume"].rolling(30, min_periods=5).mean())
        / (out["volume"].rolling(30, min_periods=5).std() + 1e-9)
    )
    out = out.dropna()
    return out


class FeaturePipeline:
    """
    Keeps feature scaling consistent between training and serving.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, target_col: str = "target"):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if self.feature_cols is None:
            self.feature_cols = [
                c for c in df.columns if c != self.target_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        X = df[self.feature_cols].values
        y = df[self.target_col].values
        X_scaled = self.scaler.fit_transform(X)
        meta = {
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "scaler_mean_": self.scaler.mean_.tolist(),
            "scaler_scale_": self.scaler.scale_.tolist(),
        }
        return X_scaled, y, meta

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[self.feature_cols].values)

    @classmethod
    def from_meta(cls, meta: Dict) -> "FeaturePipeline":
        obj = cls(feature_cols=meta["feature_cols"], target_col=meta["target_col"])
        obj.scaler.mean_ = np.array(meta["scaler_mean_"], dtype=float)
        obj.scaler.scale_ = np.array(meta["scaler_scale_"], dtype=float)
        obj.scaler.n_features_in_ = len(meta["feature_cols"])
        return obj

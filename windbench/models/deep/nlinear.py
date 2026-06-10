"""NLinear for multi-horizon wind energy forecasting.

Adapted from Zeng et al. (2023) "Are Transformers Effective for Time Series
Forecasting?" (AAAI 2023), https://arxiv.org/abs/2205.13504.

NLinear subtracts the last time-step value before applying a linear layer,
then adds it back.  This simple normalization removes distribution shift and
often outperforms complex attention-based models on long-horizon tasks.

Adaptation for NWP input
------------------------
The original NLinear operates on a univariate lookback window.  Here we first
project the F weather features at each step to a scalar with a learned linear,
then apply the NLinear normalization and forecast linear over the T-step
lead-time axis.

Input:  (batch, T, F)  — one NWP run
Output: (batch, T)     — energy forecast for every lead-time step
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from windbench.models.base import BaseSeq2SeqModel


class _NLinearNet(nn.Module):
    def __init__(self, T: int, F: int) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(F, 1)   # (T, F) → (T, 1) per timestep
        self.linear = nn.Linear(T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        seq = self.feature_proj(x).squeeze(-1)  # (batch, T)
        last = seq[:, -1:]                       # (batch, 1) — local reference point
        out = self.linear(seq - last) + last     # NLinear normalization
        return out                               # (batch, T)


class NLinearModel(BaseSeq2SeqModel):
    """NLinear seq2seq model: one NWP run → energy forecast for all T lead times.

    A minimal baseline: project NWP features to scalars, subtract the last
    lead-time value, apply a single linear layer, add the reference back.

    Parameters
    ----------
    lr:
        Adam learning rate.
    epochs:
        Training epochs.
    batch_size:
        Mini-batch size.
    device:
        ``"cpu"`` or ``"cuda"``. Auto-detected if not set.
    """

    name = "nlinear"

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: _NLinearNet | None = None
        self._X_mean: np.ndarray | None = None
        self._X_std: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num((X - self._X_mean) / (self._X_std + 1e-8), nan=0.0)

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self._y_mean) / (self._y_std + 1e-8)

    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        return y * (self._y_std + 1e-8) + self._y_mean

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "NLinearModel":
        self._X_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        self._X_std  = np.nanstd(X_train,  axis=(0, 1), keepdims=True)
        self._y_mean = float(np.nanmean(y_train))
        self._y_std  = float(np.nanstd(y_train))

        X = self._normalize_X(X_train.astype(np.float32))
        y = self._normalize_y(y_train.astype(np.float32))

        T, F = X.shape[1], X.shape[2]
        self._model = _NLinearNet(T=T, F=F).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X).to(self.device),
                torch.from_numpy(y).to(self.device),
            ),
            batch_size=self.batch_size, shuffle=True,
        )

        self._model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._model(xb)
                mask = ~torch.isnan(yb)
                loss = ((pred - yb.nan_to_num(0.0)) ** 2 * mask).sum() / mask.sum().clamp(min=1)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._normalize_X(X_test.astype(np.float32))
        self._model.eval()
        with torch.no_grad():
            out = self._model(torch.from_numpy(X).to(self.device)).cpu().numpy()
        return self._denormalize_y(out)

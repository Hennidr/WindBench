"""Sequence-to-sequence LSTM for multi-horizon wind energy forecasting.

Receives one full NWP run as input (T lead-time steps × F weather features)
and predicts energy production at every lead-time step simultaneously.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from windbench.models.base import BaseSeq2SeqModel


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)               # (batch, T, hidden_dim)
        return self.head(out).squeeze(-1)   # (batch, T)


class LSTMModel(BaseSeq2SeqModel):
    """Many-to-many LSTM: one NWP run → energy forecast for all T lead times.

    Parameters
    ----------
    hidden_dim:
        LSTM hidden state size.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout between LSTM layers (only active when num_layers > 1).
    lr:
        Adam learning rate.
    epochs:
        Training epochs.
    batch_size:
        Mini-batch size.
    device:
        ``"cpu"`` or ``"cuda"``. Auto-detected if not set.
    """

    name = "lstm"

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: _LSTMNet | None = None
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
    ) -> "LSTMModel":
        self._X_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        self._X_std  = np.nanstd(X_train,  axis=(0, 1), keepdims=True)
        self._y_mean = float(np.nanmean(y_train))
        self._y_std  = float(np.nanstd(y_train))

        X = self._normalize_X(X_train.astype(np.float32))
        y = self._normalize_y(y_train.astype(np.float32))

        self._model = _LSTMNet(
            X.shape[-1], self.hidden_dim, self.num_layers, self.dropout
        ).to(self.device)
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

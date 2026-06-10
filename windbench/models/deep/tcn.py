"""Temporal Convolutional Network for multi-horizon wind energy forecasting.

Architecture uses stacked dilated causal convolutions (Bai et al., 2018
"An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling", https://arxiv.org/abs/1803.01271).

Each TCN block applies two causal 1-D convolutions with the same dilation,
followed by a residual skip connection.  Dilations grow exponentially
(1, 2, 4, ..., 2^(num_levels-1)) so the receptive field covers the full
T-step NWP run without recurrence.

Input:  (batch, T, F)  — one NWP run (T lead-time steps × F features)
Output: (batch, T)     — energy forecast for every lead-time step
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from windbench.models.base import BaseSeq2SeqModel


class _CausalConv1d(nn.Module):
    """Causal 1-D convolution: pads only the left side so no future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self._pad, 0)))


class _TCNBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TCNNet(nn.Module):
    def __init__(
        self, F: int, hidden_channels: int, num_levels: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(F, hidden_channels)
        blocks: list[nn.Module] = []
        for i in range(num_levels):
            dilation = 2 ** i
            blocks.append(_TCNBlock(hidden_channels, hidden_channels, kernel_size, dilation, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        x = self.input_proj(x)          # (batch, T, hidden_channels)
        x = x.transpose(1, 2)           # (batch, hidden_channels, T)
        x = self.blocks(x)              # (batch, hidden_channels, T)
        x = x.transpose(1, 2)           # (batch, T, hidden_channels)
        return self.head(x).squeeze(-1) # (batch, T)


class TCNModel(BaseSeq2SeqModel):
    """Temporal Convolutional Network: one NWP run → energy forecast for all T lead times.

    Parameters
    ----------
    hidden_channels:
        Number of channels in every TCN layer.
    num_levels:
        Number of TCN blocks; dilation doubles each level (1, 2, 4, ...).
    kernel_size:
        Convolution kernel size within each block.
    dropout:
        Dropout probability applied after each convolution.
    lr:
        Adam learning rate.
    epochs:
        Training epochs.
    batch_size:
        Mini-batch size.
    device:
        ``"cpu"`` or ``"cuda"``. Auto-detected if not set.
    """

    name = "tcn"

    def __init__(
        self,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: _TCNNet | None = None
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
    ) -> "TCNModel":
        self._X_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        self._X_std  = np.nanstd(X_train,  axis=(0, 1), keepdims=True)
        self._y_mean = float(np.nanmean(y_train))
        self._y_std  = float(np.nanstd(y_train))

        X = self._normalize_X(X_train.astype(np.float32))
        y = self._normalize_y(y_train.astype(np.float32))

        self._model = _TCNNet(
            F=X.shape[2],
            hidden_channels=self.hidden_channels,
            num_levels=self.num_levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
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

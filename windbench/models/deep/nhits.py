"""N-HiTS for multi-horizon wind energy forecasting.

Architecture follows Challu et al. (2023) "N-HiTS: Neural Hierarchical
Interpolation for Time Series Forecasting" (AAAI 2023).

Key differences from N-BEATS
-----------------------------
1. **Multi-rate input downsampling**: each stack operates on a *pooled* version
   of the input residual, with pooling kernel size growing across stacks.
2. **Hierarchical interpolation**: each block produces a forecast of size
   ``T // pool_size`` which is upsampled back to T via linear interpolation.

References
----------
* Paper: https://arxiv.org/abs/2201.12886
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from windbench.models.base import BaseSeq2SeqModel


class _NHiTSBlock(nn.Module):
    def __init__(self, T: int, pool_size: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.T = T
        self.pool_size = pool_size
        self.T_pooled = max(1, T // pool_size)

        layers: list[nn.Module] = []
        for i in range(n_layers):
            in_dim = self.T_pooled if i == 0 else hidden_dim
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        self.fc_stack = nn.Sequential(*layers)
        self.theta_backcast = nn.Linear(hidden_dim, T, bias=False)
        self.theta_forecast  = nn.Linear(hidden_dim, self.T_pooled, bias=False)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.max_pool1d(
            residual.unsqueeze(1),
            kernel_size=self.pool_size, stride=self.pool_size, ceil_mode=True,
        ).squeeze(1)
        pooled = pooled[:, : self.T_pooled]

        h = self.fc_stack(pooled)
        backcast = self.theta_backcast(h)
        forecast_coarse = self.theta_forecast(h)

        forecast = F.interpolate(
            forecast_coarse.unsqueeze(1),
            size=self.T, mode="linear", align_corners=False,
        ).squeeze(1)

        return backcast, forecast


class _NHiTSNet(nn.Module):
    def __init__(
        self,
        T: int,
        F: int,
        num_stacks: int,
        num_blocks_per_stack: int,
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(F, 1)
        pool_sizes = _compute_pool_sizes(T, num_stacks)
        blocks: list[nn.Module] = []
        for s in range(num_stacks):
            for _ in range(num_blocks_per_stack):
                blocks.append(_NHiTSBlock(T, pool_sizes[s], hidden_dim, n_layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.input_proj(x).squeeze(-1)   # (batch, T)
        forecast_sum = torch.zeros_like(residual)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        return forecast_sum


def _compute_pool_sizes(T: int, num_stacks: int) -> list[int]:
    if num_stacks == 1:
        return [1]
    return [max(1, int(T ** (1.0 - s / (num_stacks - 1)))) for s in range(num_stacks)]


class NHiTSModel(BaseSeq2SeqModel):
    """N-HiTS seq2seq model: one NWP run → energy forecast for all T lead times.

    Parameters
    ----------
    num_stacks:
        Number of N-HiTS stacks (each has a different pooling rate).
    num_blocks_per_stack:
        Number of blocks within each stack.
    hidden_dim:
        Width of the FC layers inside each block.
    n_layers:
        Number of FC layers per block.
    lr:
        Adam learning rate.
    epochs:
        Training epochs.
    batch_size:
        Mini-batch size.
    device:
        ``"cpu"`` or ``"cuda"``. Auto-detected if not set.
    """

    name = "nhits"

    def __init__(
        self,
        num_stacks: int = 3,
        num_blocks_per_stack: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.num_stacks = num_stacks
        self.num_blocks_per_stack = num_blocks_per_stack
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: _NHiTSNet | None = None
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
    ) -> "NHiTSModel":
        self._X_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        self._X_std  = np.nanstd(X_train,  axis=(0, 1), keepdims=True)
        self._y_mean = float(np.nanmean(y_train))
        self._y_std  = float(np.nanstd(y_train))

        X = self._normalize_X(X_train.astype(np.float32))
        y = self._normalize_y(y_train.astype(np.float32))

        T, F = X.shape[1], X.shape[2]
        self._model = _NHiTSNet(
            T=T, F=F,
            num_stacks=self.num_stacks,
            num_blocks_per_stack=self.num_blocks_per_stack,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
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

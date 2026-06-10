"""N-BEATS for multi-horizon wind energy forecasting.

Architecture follows Oreshkin et al. (2020) "N-BEATS: Neural basis expansion
analysis for interpretable time series forecasting" (ICLR 2020), generic
(non-interpretable) variant.

Adaptation for NWP input
------------------------
The original N-BEATS takes a univariate look-back window as input and produces
a univariate forecast.  Here the input is an NWP run tensor ``(T, F)`` — one
weather-feature vector per lead-time step.  An input projection layer reduces
``(T, F) → (T,)`` per-timestep before the block stack, so the block internals
remain identical to the paper.

Block structure (generic basis)
--------------------------------
Each block receives a residual signal ``r ∈ R^T`` and produces:
  * backcast ``b ∈ R^T``: the part of r the block *explains*
  * forecast ``f ∈ R^T``: the block's contribution to the output

The next block receives ``r - b``.  The final forecast is the sum of all
block forecasts across all stacks.

References
----------
* Paper: https://arxiv.org/abs/1905.10437
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from windbench.models.base import BaseSeq2SeqModel


class _NBEATSBlock(nn.Module):
    def __init__(self, T: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_layers):
            in_dim = T if i == 0 else hidden_dim
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        self.fc_stack = nn.Sequential(*layers)
        self.theta_backcast = nn.Linear(hidden_dim, T, bias=False)
        self.theta_forecast  = nn.Linear(hidden_dim, T, bias=False)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc_stack(residual)
        return self.theta_backcast(h), self.theta_forecast(h)


class _NBEATSNet(nn.Module):
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
        blocks: list[nn.Module] = []
        for _ in range(num_stacks * num_blocks_per_stack):
            blocks.append(_NBEATSBlock(T, hidden_dim, n_layers))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.input_proj(x).squeeze(-1)   # (batch, T)
        forecast_sum = torch.zeros_like(residual)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        return forecast_sum   # (batch, T)


class NBEATSModel(BaseSeq2SeqModel):
    """N-BEATS seq2seq model: one NWP run → energy forecast for all T lead times.

    Parameters
    ----------
    num_stacks:
        Number of N-BEATS stacks.
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

    name = "nbeats"

    def __init__(
        self,
        num_stacks: int = 30,
        num_blocks_per_stack: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 4,
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
        self._model: _NBEATSNet | None = None
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
    ) -> "NBEATSModel":
        self._X_mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
        self._X_std  = np.nanstd(X_train,  axis=(0, 1), keepdims=True)
        self._y_mean = float(np.nanmean(y_train))
        self._y_std  = float(np.nanstd(y_train))

        X = self._normalize_X(X_train.astype(np.float32))
        y = self._normalize_y(y_train.astype(np.float32))

        T, F = X.shape[1], X.shape[2]
        self._model = _NBEATSNet(
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

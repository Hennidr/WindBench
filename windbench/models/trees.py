"""Linear and tree models for multi-horizon wind energy forecasting.

All models receive the same input as the NN models: a single NWP run tensor
(T, F) flattened into T*F features, plus any production-lag features appended
by ``build_seq2seq_arrays`` (n_lags > 0).  They predict all T lead times
simultaneously via multi-output regression.

This unified input structure makes them directly comparable to the NN models:
same samples, same features, same targets, no horizon masking.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from windbench.models.base import BaseSeq2SeqModel


class _RunViewModel(BaseSeq2SeqModel):
    """Shared logic for flattening (n_runs, T, F) → (n_runs, T*F) regression."""

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        n, T, F = X.shape
        return X.reshape(n, T * F)

    def _norm_X(self, X_flat: np.ndarray) -> np.ndarray:
        return np.nan_to_num((X_flat - self._X_mean) / (self._X_std + 1e-8))

    def _norm_y(self, y: np.ndarray) -> np.ndarray:
        return np.nan_to_num((y - self._y_mean) / (self._y_std + 1e-8))

    def _denorm_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * (self._y_std + 1e-8) + self._y_mean

    def _fit_stats(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_flat = self._flatten(X_train)
        self._X_mean = np.nanmean(X_flat, axis=0)
        self._X_std  = np.nanstd(X_flat,  axis=0)
        self._y_mean = np.nanmean(y_train, axis=0)  # (T,)
        self._y_std  = np.nanstd(y_train,  axis=0)  # (T,)

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_norm = self._norm_X(self._flatten(X_test))
        y_norm = self._model.predict(X_norm)
        return self._denorm_y(y_norm).astype(np.float32)


class RandomForestModel(_RunViewModel):
    """Random Forest trained on flattened NWP run features.

    Uses ``MultiOutputRegressor`` to train one RF per lead time, so each
    output head can independently weight features (e.g. lag_1h heavily at
    h=1, NWP features more at h=24).

    Parameters
    ----------
    n_estimators:
        Number of trees per lead-time model.
    max_depth:
        Maximum tree depth (``None`` = unlimited).
    min_samples_leaf:
        Minimum samples per leaf.
    n_jobs:
        Parallelism for both MultiOutputRegressor and each RF (``-1`` = all cores).
    random_state:
        Reproducibility seed.
    """

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: MultiOutputRegressor | None = None
        self._X_mean = self._X_std = self._y_mean = self._y_std = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "RandomForestModel":
        self._fit_stats(X_train, y_train)
        X_norm = self._norm_X(self._flatten(X_train))
        y_norm = self._norm_y(y_train)
        valid = ~np.all(np.isnan(y_train), axis=1)
        base = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=1,
            random_state=self.random_state,
        )
        self._model = MultiOutputRegressor(base, n_jobs=self.n_jobs)
        self._model.fit(X_norm[valid], y_norm[valid])
        return self


class XGBoostModel(_RunViewModel):
    """XGBoost trained on flattened NWP run features.

    Uses ``MultiOutputRegressor`` to predict all T lead times, one estimator
    per output.

    Parameters
    ----------
    n_estimators:
        Boosting rounds per output.
    learning_rate:
        Step-size shrinkage.
    max_depth:
        Maximum tree depth.
    subsample:
        Row subsampling ratio per tree.
    colsample_bytree:
        Feature subsampling ratio per tree.
    random_state:
        Reproducibility seed.
    """

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self._model: MultiOutputRegressor | None = None
        self._X_mean = self._X_std = self._y_mean = self._y_std = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "XGBoostModel":
        import xgboost as xgb

        self._fit_stats(X_train, y_train)
        X_norm = self._norm_X(self._flatten(X_train))
        y_norm = self._norm_y(y_train)
        valid = ~np.all(np.isnan(y_train), axis=1)
        base = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
        )
        self._model = MultiOutputRegressor(base, n_jobs=-1)
        self._model.fit(X_norm[valid], y_norm[valid])
        return self


class RidgeModel(_RunViewModel):
    """Ridge regression on flattened NWP run features.

    Linear baseline that uses the same (T*F) input as tree and NN models.
    Establishes how much skill comes from the NWP signal being linearly
    predictive, independent of model complexity.

    Parameters
    ----------
    alpha:
        L2 regularisation strength.
    """

    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model: MultiOutputRegressor | None = None
        self._X_mean = self._X_std = self._y_mean = self._y_std = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "RidgeModel":
        self._fit_stats(X_train, y_train)
        X_norm = self._norm_X(self._flatten(X_train))
        y_norm = self._norm_y(y_train)
        valid = ~np.all(np.isnan(y_train), axis=1)
        self._model = MultiOutputRegressor(Ridge(alpha=self.alpha, solver="lsqr"), n_jobs=-1)
        self._model.fit(X_norm[valid], y_norm[valid])
        return self

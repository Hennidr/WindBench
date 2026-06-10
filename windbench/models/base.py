"""Abstract base class that all WindBench models must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Unified interface for all forecasting models.

    Subclasses must implement :meth:`fit` and :meth:`predict`.
    """

    name: str = "base"
    horizon: int = 1  # set by the benchmark runner before fit()

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> "BaseModel":
        """Fit the model on training data.

        Parameters
        ----------
        X_train:
            Feature matrix (rows = samples, cols = features).
        y_train:
            Target series aligned with *X_train*.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions for *X_test*.

        Parameters
        ----------
        X_test:
            Feature matrix for the test period.

        Returns
        -------
        np.ndarray
            1-D array of predictions with length ``len(X_test)``.
        """

    def predict_quantiles(
        self,
        X_test: pd.DataFrame,
        quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    ) -> np.ndarray:
        """Generate quantile forecasts for *X_test*.

        Parameters
        ----------
        X_test:
            Feature matrix for the test period.
        quantiles:
            Probability levels in (0, 1).

        Returns
        -------
        np.ndarray
            Array of shape ``(len(X_test), len(quantiles))``.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support probabilistic forecasting.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BaseSeq2SeqModel(ABC):
    """Interface for sequence-to-sequence models operating on NWP run arrays.

    Unlike ``BaseModel``, inputs and outputs are 3-D / 2-D numpy arrays
    indexed by (n_runs, T, n_features) and (n_runs, T) respectively, where
    T is the number of lead-time steps in one NWP run.

    Subclasses must implement :meth:`fit` and :meth:`predict`.
    """

    name: str = "seq2seq_base"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> "BaseSeq2SeqModel":
        """Fit on training run arrays.

        Parameters
        ----------
        X_train : (n_train, T, n_features) float32
        y_train : (n_train, T)             float32, may contain NaN
        X_val, y_val : optional validation arrays of the same shape
        """

    @abstractmethod
    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        """Return predictions of shape (n_test, T).

        Parameters
        ----------
        X_test : (n_test, T, n_features)
        **kwargs : extra context (e.g. ``run_times``) passed by the benchmark
            runner.  Models that do not need extra context should ignore these.
        """

    def predict_at_horizons(
        self,
        X_test: np.ndarray,
        horizons: list[int],
        lead_times: list[int],
        **kwargs,
    ) -> dict[int, np.ndarray]:
        """Slice full-sequence predictions at specific lead-time indices.

        Parameters
        ----------
        X_test : (n_test, T, n_features)
        horizons : lead times to extract, e.g. [1, 6, 24]
        lead_times : the full ordered lead_times list used in build_seq2seq_arrays
        **kwargs : forwarded to :meth:`predict` (e.g. ``run_times``)

        Returns
        -------
        dict mapping each h → 1-D array of shape (n_test,)
        """
        missing = [h for h in horizons if h not in lead_times]
        if missing:
            raise ValueError(
                f"Horizons {missing} not in lead_times. Available: {lead_times[:5]}..."
            )
        y_pred = self.predict(X_test, **kwargs)   # (n_test, T)
        return {h: y_pred[:, lead_times.index(h)] for h in horizons}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

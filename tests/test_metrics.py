"""Tests for windbench.evaluation.metrics."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from windbench.evaluation.metrics import rmse, mae, mape, skill_score, evaluate


def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_mae_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # sqrt((9 + 16) / 2) = sqrt(12.5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_skill_score_perfect():
    y = np.ones(10)
    perfect = np.ones(10)
    baseline = np.zeros(10)
    assert skill_score(y, perfect, baseline) == pytest.approx(1.0)


def test_skill_score_same_as_baseline():
    y = np.ones(10)
    assert skill_score(y, np.zeros(10), np.zeros(10)) == pytest.approx(0.0)


def test_nan_handling():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([np.nan, 2.0, 3.0])
    # Only last 2 elements are valid
    assert rmse(y_true, y_pred) == pytest.approx(0.0)


def test_evaluate_returns_all_metrics():
    y = np.array([1.0, 2.0, 3.0])
    result = evaluate(y, y, y_baseline=y * 0)
    assert "rmse" in result
    assert "mae" in result
    assert "mape" in result
    assert "skill_score" in result

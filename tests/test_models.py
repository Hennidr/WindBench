"""Smoke tests: all models fit and predict without error."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from windbench.data.loader import load_farm
from windbench.data.seq2seq import build_seq2seq_arrays, seq2seq_temporal_split
from windbench.models.arima import ARIMAModel
from windbench.models.trees import RandomForestModel, XGBoostModel

LEAD_TIMES = list(range(1, 25))   # 24 steps matches the synthetic fixture

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def seq2seq_data(sample_df, tmp_path):
    """Small seq2seq arrays: write synthetic data to parquet, load via load_farm."""
    path = tmp_path / "farm.parquet"
    sample_df.to_parquet(path)
    df = load_farm(path, target_col="energy_total")

    weather_cols = [c for c in ["2d", "2t", "10u", "10v", "100u", "100v", "msl", "tp"]
                    if c in df.columns]
    X, y, run_times = build_seq2seq_arrays(
        df,
        weather_cols=weather_cols,
        lead_times=LEAD_TIMES,
        target_col="energy_total",
        min_lead_coverage=0.5,
    )
    X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test = (
        seq2seq_temporal_split(X, y, run_times, val_frac=0.1, test_frac=0.2)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test


# ── Tree models ───────────────────────────────────────────────────────────────

def test_random_forest_fit_predict(seq2seq_data):
    X_train, _, X_test, y_train, _, _, t_train, _, t_test = seq2seq_data
    model = RandomForestModel(n_estimators=10)
    model.fit(X_train, y_train, run_times=t_train)
    preds = model.predict(X_test, run_times=t_test)
    assert preds.shape == (len(X_test), X_test.shape[1])
    assert not np.all(np.isnan(preds)), "RandomForest returned all NaN"


def test_random_forest_predict_at_horizons(seq2seq_data):
    X_train, _, X_test, y_train, _, _, t_train, _, t_test = seq2seq_data
    lead_times = LEAD_TIMES
    eval_horizons = [h for h in [1, 6, 24] if h in lead_times]
    if not eval_horizons:
        pytest.skip("No eval horizons in lead_times for this sample")
    model = RandomForestModel(n_estimators=10)
    model.fit(X_train, y_train, run_times=t_train)
    h_preds = model.predict_at_horizons(X_test, eval_horizons, lead_times, run_times=t_test)
    for h in eval_horizons:
        assert h in h_preds
        assert len(h_preds[h]) == len(X_test)


def test_xgboost_fit_predict(seq2seq_data):
    X_train, _, X_test, y_train, _, _, t_train, _, t_test = seq2seq_data
    model = XGBoostModel(n_estimators=10)
    model.fit(X_train, y_train, run_times=t_train)
    preds = model.predict(X_test, run_times=t_test)
    assert preds.shape == (len(X_test), X_test.shape[1])
    assert not np.all(np.isnan(preds)), "XGBoost returned all NaN"


# ── Statistical ───────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_arima_fit_predict(seq2seq_data):
    X_train, _, X_test, y_train, _, _, t_train, _, t_test = seq2seq_data
    model = ARIMAModel(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model.fit(X_train, y_train, run_times=t_train)
    preds = model.predict(X_test, run_times=t_test)
    assert preds.shape == (len(X_test), X_test.shape[1])


# ── Deep NN ───────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_lstm_fit_predict(seq2seq_data):
    from windbench.models.deep.lstm import LSTMModel
    X_train, _, X_test, y_train, _, _, _, _, _ = seq2seq_data
    model = LSTMModel(hidden_dim=16, num_layers=1, epochs=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test), X_test.shape[1])
    assert not np.all(np.isnan(preds))


@pytest.mark.slow
def test_transformer_fit_predict(seq2seq_data):
    from windbench.models.deep.transformer import TransformerModel
    X_train, _, X_test, y_train, _, _, _, _, _ = seq2seq_data
    model = TransformerModel(d_model=16, nhead=2, num_layers=1, epochs=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test), X_test.shape[1])
    assert not np.all(np.isnan(preds))


@pytest.mark.slow
def test_nbeats_fit_predict(seq2seq_data):
    from windbench.models.deep.nbeats import NBEATSModel
    X_train, _, X_test, y_train, _, _, _, _, _ = seq2seq_data
    model = NBEATSModel(num_stacks=2, num_blocks_per_stack=1, hidden_dim=32, n_layers=2, epochs=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test), X_test.shape[1])
    assert not np.all(np.isnan(preds))


@pytest.mark.slow
def test_nhits_fit_predict(seq2seq_data):
    from windbench.models.deep.nhits import NHiTSModel
    X_train, _, X_test, y_train, _, _, _, _, _ = seq2seq_data
    model = NHiTSModel(num_stacks=2, num_blocks_per_stack=1, hidden_dim=32, n_layers=2, epochs=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test), X_test.shape[1])
    assert not np.all(np.isnan(preds))


@pytest.mark.slow
def test_tcn_fit_predict(seq2seq_data):
    from windbench.models.deep.tcn import TCNModel
    X_train, _, X_test, y_train, _, _, _, _, _ = seq2seq_data
    model = TCNModel(hidden_channels=16, num_levels=2, kernel_size=3, epochs=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test), X_test.shape[1])
    assert not np.all(np.isnan(preds))


@pytest.mark.slow
def test_nlinear_fit_predict(seq2seq_data):
    from windbench.models.deep.nlinear import NLinearModel
    X_train, _, X_test, y_train, _, _, _, _, _ = seq2seq_data
    model = NLinearModel(epochs=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test), X_test.shape[1])
    assert not np.all(np.isnan(preds))


@pytest.mark.slow
def test_dlinear_fit_predict(seq2seq_data):
    from windbench.models.deep.dlinear import DLinearModel
    X_train, _, X_test, y_train, _, _, _, _, _ = seq2seq_data
    model = DLinearModel(kernel_size=5, epochs=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test), X_test.shape[1])
    assert not np.all(np.isnan(preds))


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_mase_metric():
    from windbench.evaluation.metrics import mase
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true  = np.array([3.0, 4.0, 5.0])
    y_pred  = np.array([3.0, 4.0, 5.0])
    assert mase(y_true, y_pred, y_train) == pytest.approx(0.0, abs=1e-6)


def test_mse_metric():
    from windbench.evaluation.metrics import mse
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert mse(y_true, y_pred) == pytest.approx(2.0 / 3.0, rel=1e-5)


# ── Registry ──────────────────────────────────────────────────────────────────

def test_unknown_model_raises():
    from windbench.evaluation.seq2seq_benchmark import _get_seq2seq_model
    with pytest.raises(ValueError, match="Unknown model"):
        _get_seq2seq_model("does_not_exist")

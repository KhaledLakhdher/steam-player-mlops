"""Model quality gate — CI fails if the model regresses below the bar or a naive baseline."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from src.data import generate_synthetic_panel
from src.features import build_training_frame


def _mape(y, pred):
    mask = y > 100
    return float(np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100)


def test_model_beats_naive_baseline():
    frame = build_training_frame(generate_synthetic_panel(n_games=15))
    cutoff = frame["date"].quantile(0.8)
    tr, te = frame[frame["date"] <= cutoff], frame[frame["date"] > cutoff]

    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=0)
    model.fit(tr[config.FEATURE_COLUMNS], tr[config.TARGET])

    y = te[config.TARGET].values
    model_mape = _mape(y, model.predict(te[config.FEATURE_COLUMNS]))
    naive_mape = _mape(y, te["players_lag_1"].values)     # predict "same as yesterday"

    assert model_mape < 30, f"MAPE too high: {model_mape:.1f}%"
    assert model_mape < naive_mape, f"model {model_mape:.1f}% not better than naive {naive_mape:.1f}%"

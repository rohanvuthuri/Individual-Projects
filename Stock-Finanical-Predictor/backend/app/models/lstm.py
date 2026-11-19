"""LSTM network that forecasts the log return over the horizon (→ a price target).

TensorFlow is imported inside the functions because it takes several seconds
to load and the rest of the API shouldn't wait for it.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..modes import Mode

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TEST_FRACTION = 0.2
MAX_EPOCHS = 40
BATCH_SIZE = 64


def _windows(values: np.ndarray, ends: np.ndarray, lookback: int) -> np.ndarray:
    """Stack the `lookback` rows ending at each index in `ends` → (samples, lookback, features)."""
    offsets = np.arange(-lookback + 1, 1)
    return values[ends[:, None] + offsets]


def _build(lookback: int, n_features: int):
    from keras import Input, Sequential
    from keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        Input((lookback, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


def train(X: pd.DataFrame, fwd_ret: pd.Series, mode: Mode) -> dict:
    from keras.callbacks import EarlyStopping
    from keras.utils import set_random_seed

    # Same idea as the Earthquake model: a fixed seed makes runs repeatable.
    set_random_seed(42)

    values = X.to_numpy(dtype=np.float32)
    target = fwd_ret.to_numpy(dtype=np.float32)
    lb = mode.lookback
    ends = np.where(~np.isnan(target))[0]
    ends = ends[ends >= lb - 1]

    # Hold out the most recent samples for testing, with a horizon-sized gap so
    # training labels don't overlap the test period.
    n_test = int(len(ends) * TEST_FRACTION)
    test_ends = ends[-n_test:]
    train_ends = ends[ends < test_ends[0] - mode.horizon]

    # Fit scaling on training rows only so test statistics don't leak in.
    scaler = StandardScaler().fit(values[: train_ends[-1] + 1])
    scaled = scaler.transform(values).astype(np.float32)
    y_scale = float(np.std(target[train_ends])) or 1.0

    stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model = _build(lb, values.shape[1])
    history = model.fit(
        _windows(scaled, train_ends, lb), target[train_ends] / y_scale,
        validation_split=0.15, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        shuffle=True, callbacks=[stop], verbose=0,
    )
    best_epochs = int(np.argmin(history.history["val_loss"])) + 1

    pred = model.predict(_windows(scaled, test_ends, lb), verbose=0).ravel() * y_scale
    actual = target[test_ends]
    up_rate = float((actual > 0).mean())
    accuracy = float(((pred > 0) == (actual > 0)).mean())
    metrics = {
        "accuracy": accuracy,
        "baseline": up_rate,
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "correlation": float(np.corrcoef(pred, actual)[0, 1]) if np.std(pred) > 0 else 0.0,
        "test_samples": int(len(test_ends)),
        "test_start": str(X.index[test_ends[0]].date()),
        "test_end": str(X.index[test_ends[-1]].date()),
        "epochs": best_epochs,
    }
    metrics["beats_baseline"] = accuracy > max(up_rate, 1 - up_rate)

    # Retrain on everything (including the test period) for the live forecast,
    # using the epoch count that worked best above.
    all_ends = ends
    scaler = StandardScaler().fit(values[: all_ends[-1] + 1])
    scaled = scaler.transform(values).astype(np.float32)
    y_scale = float(np.std(target[all_ends])) or 1.0
    set_random_seed(42)
    final = _build(lb, values.shape[1])
    final.fit(
        _windows(scaled, all_ends, lb), target[all_ends] / y_scale,
        epochs=best_epochs, batch_size=BATCH_SIZE, shuffle=True, verbose=0,
    )

    return {
        "model": final,
        "scaler": scaler,
        "y_scale": y_scale,
        "features": list(X.columns),
        "trained_at": datetime.now(timezone.utc),
        "metrics": metrics,
    }


def predict(bundle: dict, X_live: pd.DataFrame, lookback: int) -> float:
    """Forecast log return over the horizon from the most recent window."""
    values = bundle["scaler"].transform(X_live[bundle["features"]].to_numpy(dtype=np.float32))
    window = values[-lookback:][None, :, :].astype(np.float32)
    return float(bundle["model"](window, training=False).numpy().ravel()[0] * bundle["y_scale"])

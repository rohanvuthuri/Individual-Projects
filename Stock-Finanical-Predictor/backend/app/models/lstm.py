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
from .validation import direction_metrics, walk_forward

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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


def _fit(values: np.ndarray, target: np.ndarray, train_ends: np.ndarray, lb: int, epochs: int | None):
    """Train one network on the windows ending at `train_ends`.

    With epochs=None, early stopping on a validation slice picks the epoch count.
    Scaling is fit on training rows only so later statistics don't leak in.
    Returns (model, scaled values, y_scale, epochs used).
    """
    from keras.callbacks import EarlyStopping
    from keras.utils import set_random_seed

    # Same idea as the Earthquake model: a fixed seed makes runs repeatable.
    set_random_seed(42)
    scaler = StandardScaler().fit(values[: train_ends[-1] + 1])
    scaled = scaler.transform(values).astype(np.float32)
    y_scale = float(np.std(target[train_ends])) or 1.0

    model = _build(lb, values.shape[1])
    x, y = _windows(scaled, train_ends, lb), target[train_ends] / y_scale
    if epochs is None:
        stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = model.fit(x, y, validation_split=0.15, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                            shuffle=True, callbacks=[stop], verbose=0)
        epochs = int(np.argmin(history.history["val_loss"])) + 1
    else:
        model.fit(x, y, epochs=epochs, batch_size=BATCH_SIZE, shuffle=True, verbose=0)
    return model, scaler, scaled, y_scale, epochs


def train(X: pd.DataFrame, fwd_ret: pd.Series, mode: Mode) -> dict:
    values = X.to_numpy(dtype=np.float32)
    target = fwd_ret.to_numpy(dtype=np.float32)
    lb = mode.lookback
    ends = np.where(~np.isnan(target))[0]
    ends = ends[ends >= lb - 1]

    # Walk-forward with the same folds as gradient boosting, so both models
    # have out-of-sample predictions for the same periods (needed for stacking).
    oof = np.full(len(X), np.nan)
    fold_epochs = []
    for tr, te in walk_forward(len(ends), mode.horizon):
        train_ends, test_ends = ends[tr], ends[te]
        model, _, scaled, y_scale, epochs = _fit(values, target, train_ends, lb, epochs=None)
        # Calling the model directly (not .predict) avoids TensorFlow re-tracing warnings per fold.
        oof[test_ends] = model(_windows(scaled, test_ends, lb), training=False).numpy().ravel() * y_scale
        fold_epochs.append(epochs)

    oof_pred = pd.Series(oof, index=X.index).dropna()
    actual = fwd_ret[oof_pred.index]
    metrics = direction_metrics(actual > 0, oof_pred > 0, oof_pred)
    metrics.update({
        "rmse": float(np.sqrt(np.mean((oof_pred - actual) ** 2))),
        "correlation": float(np.corrcoef(oof_pred, actual)[0, 1]) if oof_pred.std() > 0 else 0.0,
        "epochs": int(np.median(fold_epochs)),
    })

    # Retrain on everything for the live forecast, using the typical best epoch count.
    final, scaler, _, y_scale, _ = _fit(values, target, ends, lb, epochs=metrics["epochs"])
    return {
        "model": final,
        "scaler": scaler,
        "y_scale": y_scale,
        "features": list(X.columns),
        "trained_at": datetime.now(timezone.utc),
        "metrics": metrics,
        "oof_pred": oof_pred,  # out-of-fold forecasts, used by the stacking meta-model
    }


def predict(bundle: dict, X_live: pd.DataFrame, lookback: int) -> float:
    """Forecast log return over the horizon from the most recent window."""
    values = bundle["scaler"].transform(X_live[bundle["features"]].to_numpy(dtype=np.float32))
    window = values[-lookback:][None, :, :].astype(np.float32)
    return float(bundle["model"](window, training=False).numpy().ravel()[0] * bundle["y_scale"])

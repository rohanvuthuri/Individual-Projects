"""Walk-forward splits and scoring shared by every model.

All models use the same folds so their out-of-sample predictions line up on
the same rows; the stacking meta-model is trained on exactly those rows.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

N_SPLITS = 4


def walk_forward(n: int, horizon: int, n_splits: int = N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over n time-ordered rows. The gap equals the
    horizon so no training label overlaps the test period."""
    return list(TimeSeriesSplit(n_splits=n_splits, gap=horizon).split(np.arange(n)))


def direction_metrics(y_up: pd.Series, predicted_up: pd.Series, score: pd.Series) -> dict:
    """Accuracy of up/down calls against the naive "always guess the common
    direction" baseline. `score` is anything that ranks how bullish a call is."""
    y = y_up.astype(int)
    accuracy = float((predicted_up.astype(int) == y).mean())
    baseline = float(y.mean())  # share of test periods where the price went up
    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "auc": float(roc_auc_score(y, score)) if y.nunique() > 1 else None,
        "beats_baseline": accuracy > max(baseline, 1 - baseline),
        "test_samples": int(len(y)),
        "test_start": str(y.index[0].date()),
        "test_end": str(y.index[-1].date()),
    }

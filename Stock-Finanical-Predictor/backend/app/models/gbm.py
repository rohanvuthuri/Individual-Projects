"""Gradient-boosting classifier: probability the price is higher after the horizon."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from ..data import to_chart_time
from ..features import FEATURE_LABELS
from ..modes import Mode

N_SPLITS = 5
MAX_CURVE_POINTS = 300


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )


def train(X: pd.DataFrame, fwd_ret: pd.Series, close: pd.Series, mode: Mode) -> dict:
    """Walk-forward backtest, then a final fit on every labeled row.

    The gap between each train and test fold equals the horizon, so no training
    label "sees" prices that fall inside the test period.
    """
    labeled = fwd_ret.notna()
    Xl, yl = X[labeled], (fwd_ret[labeled] > 0).astype(int)

    splitter = TimeSeriesSplit(n_splits=N_SPLITS, gap=mode.horizon)
    oof_prob = pd.Series(np.nan, index=Xl.index)
    last_fold = None
    for train_idx, test_idx in splitter.split(Xl):
        model = _new_model().fit(Xl.iloc[train_idx], yl.iloc[train_idx])
        oof_prob.iloc[test_idx] = model.predict_proba(Xl.iloc[test_idx])[:, 1]
        last_fold = (model, test_idx)

    tested = oof_prob.notna()
    y_test, p_test = yl[tested], oof_prob[tested]
    metrics = {
        "accuracy": float(accuracy_score(y_test, p_test > 0.5)),
        "baseline": float(y_test.mean()),  # accuracy of always guessing "up"
        "auc": float(roc_auc_score(y_test, p_test)) if y_test.nunique() > 1 else None,
        "test_samples": int(tested.sum()),
        "test_start": str(y_test.index[0].date()),
        "test_end": str(y_test.index[-1].date()),
    }
    metrics["beats_baseline"] = metrics["accuracy"] > max(metrics["baseline"], 1 - metrics["baseline"])

    model, test_idx = last_fold
    perm = permutation_importance(
        model, Xl.iloc[test_idx], yl.iloc[test_idx],
        n_repeats=5, random_state=42, scoring="roc_auc",
    )
    importances = sorted(
        (
            {"feature": name, "label": FEATURE_LABELS.get(name, name), "importance": float(score)}
            for name, score in zip(X.columns, perm.importances_mean)
        ),
        key=lambda d: d["importance"],
        reverse=True,
    )

    final = _new_model().fit(Xl, yl)
    return {
        "model": final,
        "features": list(X.columns),
        "trained_at": datetime.now(timezone.utc),
        "metrics": metrics,
        "importances": importances[:8],
        "equity_curve": _equity_curve(oof_prob[tested], close),
    }


def _equity_curve(prob: pd.Series, close: pd.Series) -> list[dict]:
    """Growth of $1 holding the stock only on bars where the model says "up",
    against simply holding it the whole out-of-sample period."""
    next_ret = close.pct_change().shift(-1).reindex(prob.index).fillna(0)
    position = (prob > 0.5).astype(float)
    strategy = (1 + position * next_ret).cumprod()
    buy_hold = (1 + next_ret).cumprod()

    step = max(1, len(prob) // MAX_CURVE_POINTS)
    idx = list(range(0, len(prob), step))
    if idx[-1] != len(prob) - 1:
        idx.append(len(prob) - 1)
    times = to_chart_time(prob.index[idx])
    return [
        {"time": t, "strategy": float(strategy.iloc[i]), "buy_hold": float(buy_hold.iloc[i])}
        for t, i in zip(times, idx)
    ]


def predict(bundle: dict, X_live: pd.DataFrame) -> float:
    return float(bundle["model"].predict_proba(X_live[bundle["features"]])[:, 1][-1])

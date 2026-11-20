"""Stacking meta-model: learns how much to trust each base model.

It is trained only on the base models' out-of-fold predictions (made on data
those models never trained on), so it learns how they behave on unseen data,
not how well they memorised the past.

    P(up) = sigmoid(b + w_gbm(vol) * gbm + w_lstm(vol) * lstm)
    w_model(vol) = a_model + c_model * vol      (trust can shift with volatility)

Guards against overfitting, which stacking is prone to with this little data
(overlapping long-horizon labels leave only a handful of independent outcomes):
  * the base weights a_gbm, a_lstm are kept >= 0: it can ignore a model but not
    bet against it, which is almost always a fit to noise;
  * volatility only adjusts trust, it is not a predictor on its own;
  * scaled inputs are clipped so unusual days can't produce extreme outputs;
  * `earned_trust` is only true when the meta-model's own walk-forward backtest
    beats the naive guess with real ranking skill; otherwise the app doesn't use it.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from ..modes import Mode
from .validation import direction_metrics, walk_forward

MIN_ROWS = 150
META_SPLITS = 3
MIN_AUC = 0.52  # ranking skill clearly better than a coin flip
CLIP_Z = 2.5
L2 = 0.05
FEATURES = ["gbm", "lstm", "gbm×volatility", "lstm×volatility"]


class ConstrainedLogit:
    """Logistic regression with L2 and non-negative weights on the base models."""

    def fit(self, D: np.ndarray, y: np.ndarray) -> "ConstrainedLogit":
        def loss(params):
            b, w = params[0], params[1:]
            z = b + D @ w
            p = 1 / (1 + np.exp(-z))
            err = p - y
            nll = np.mean(np.logaddexp(0, z) - y * z)
            grad = np.concatenate([[err.mean()], D.T @ err / len(y) + 2 * L2 * w])
            return nll + L2 * w @ w, grad

        base_rate = np.clip(y.mean(), 1e-3, 1 - 1e-3)
        start = np.zeros(D.shape[1] + 1)
        start[0] = np.log(base_rate / (1 - base_rate))
        bounds = [(None, None), (0, None), (0, None), (None, None), (None, None)]
        result = minimize(loss, start, jac=True, method="L-BFGS-B", bounds=bounds)
        self.intercept_, self.coef_ = float(result.x[0]), result.x[1:]
        return self

    def predict_proba(self, D: np.ndarray) -> np.ndarray:
        p = 1 / (1 + np.exp(-(self.intercept_ + D @ self.coef_)))
        return np.column_stack([1 - p, p])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _design(gbm_prob, lstm_pred, vol, scaler: StandardScaler | None = None):
    base = np.column_stack([
        _logit(np.asarray(gbm_prob, dtype=float)),
        np.asarray(lstm_pred, dtype=float),
        np.asarray(vol, dtype=float),
    ])
    if scaler is None:
        scaler = StandardScaler().fit(base)
    z = np.clip(scaler.transform(base), -CLIP_Z, CLIP_Z)
    design = np.column_stack([z[:, 0], z[:, 1], z[:, 0] * z[:, 2], z[:, 1] * z[:, 2]])
    return design, scaler, z[:, 2]


def train(gbm_oof: pd.Series, lstm_oof: pd.Series, X: pd.DataFrame, fwd_ret: pd.Series, mode: Mode) -> dict | None:
    """Returns None when the base models don't share enough out-of-fold rows."""
    idx = gbm_oof.index.intersection(lstm_oof.index).intersection(fwd_ret.dropna().index)
    if len(idx) < MIN_ROWS:
        return None
    g, l, vol = gbm_oof[idx], lstm_oof[idx], X.loc[idx, "vol_20"]
    y = (fwd_ret[idx] > 0).astype(int)
    if y.nunique() < 2:
        return None

    # Nested walk-forward: score the meta-model on stacked rows it didn't train on.
    oof = pd.Series(np.nan, index=idx)
    for tr, te in walk_forward(len(idx), mode.horizon, n_splits=META_SPLITS):
        if y.iloc[tr].nunique() < 2:
            continue
        D_tr, scaler, _ = _design(g.iloc[tr], l.iloc[tr], vol.iloc[tr])
        model = ConstrainedLogit().fit(D_tr, y.iloc[tr].to_numpy())
        D_te, _, _ = _design(g.iloc[te], l.iloc[te], vol.iloc[te], scaler)
        oof.iloc[te] = model.predict_proba(D_te)[:, 1]

    tested = oof.notna()
    if tested.sum() == 0:
        return None
    y_t, p_t = y[tested], oof[tested]
    metrics = direction_metrics(y_t, p_t > 0.5, p_t)

    # All three models scored on exactly the same test periods.
    comparison = {
        "gbm": float(((g[tested] > 0.5).astype(int) == y_t).mean()),
        "lstm": float(((l[tested] > 0).astype(int) == y_t).mean()),
        "stacked": metrics["accuracy"],
        "naive": max(metrics["baseline"], 1 - metrics["baseline"]),
        "samples": metrics["test_samples"],
    }
    earned_trust = metrics["beats_baseline"] and (metrics["auc"] or 0) > MIN_AUC

    D, scaler, _ = _design(g, l, vol)
    final = ConstrainedLogit().fit(D, y.to_numpy())
    return {
        "model": final,
        "scaler": scaler,
        "metrics": metrics,
        "comparison": comparison,
        "earned_trust": earned_trust,
        "coefficients": dict(zip(FEATURES, map(float, final.coef_))),
    }


def predict(stack: dict, gbm_prob: float, lstm_pred: float, context_row: pd.DataFrame) -> tuple[float, dict]:
    """Returns (probability up, effective weight on each model right now).

    The weights include the volatility adjustment at today's volatility, so
    they show which model the meta-model leans on in current conditions.
    """
    D, _, vol_z = _design([gbm_prob], [lstm_pred], context_row["vol_20"], stack["scaler"])
    prob = float(stack["model"].predict_proba(D)[0, 1])
    coef = stack["coefficients"]
    v = float(vol_z[0])
    weights = {
        "gbm": coef["gbm"] + coef["gbm×volatility"] * v,
        "lstm": coef["lstm"] + coef["lstm×volatility"] * v,
    }
    return prob, weights

"""Combines the two models into one Buy / Hold / Sell call.

Buy or Sell needs both models to agree. While the LSTM is still training the
call is based on gradient boosting alone and marked provisional.
"""

BUY_PROB = 0.55
SELL_PROB = 0.45


def _skill(metrics: dict) -> float:
    """0 when the backtest is no better than guessing, 1 at a 10-point edge."""
    edge = metrics["accuracy"] - max(metrics["baseline"], 1 - metrics["baseline"])
    return min(max(edge / 0.10, 0.0), 1.0)


def combine(prob_up: float, gbm_metrics: dict, lstm_return: float | None, lstm_metrics: dict | None,
            typical_move: float) -> dict:
    gbm_dir = 1 if prob_up >= BUY_PROB else -1 if prob_up <= SELL_PROB else 0
    gbm_strength = min(abs(prob_up - 0.5) * 4, 1.0)  # 0.75 probability → full strength
    gbm_weight = 0.5 + 0.5 * _skill(gbm_metrics)

    if lstm_return is None or lstm_metrics is None:
        signal = {1: "Buy", -1: "Sell", 0: "Hold"}[gbm_dir]
        return {
            "signal": signal,
            "confidence": round(gbm_strength * gbm_weight * 0.5, 3),
            "provisional": True,
            "agreement": None,
        }

    lstm_dir = 1 if lstm_return > 0 else -1
    lstm_strength = min(abs(lstm_return) / typical_move, 1.0) if typical_move > 0 else 0.0
    lstm_weight = 0.5 + 0.5 * _skill(lstm_metrics)

    agree = gbm_dir == lstm_dir
    if agree and gbm_dir == 1:
        signal = "Buy"
    elif agree and gbm_dir == -1:
        signal = "Sell"
    else:
        signal = "Hold"

    confidence = (gbm_strength * gbm_weight + lstm_strength * lstm_weight) / 2
    if not agree:
        confidence *= 0.5

    if gbm_dir == 0:
        agreement = "neutral"  # gradient boosting has no strong view either way
    else:
        agreement = "agree" if agree else "disagree"

    return {
        "signal": signal,
        "confidence": round(confidence, 3),
        "provisional": False,
        "agreement": agreement,
    }

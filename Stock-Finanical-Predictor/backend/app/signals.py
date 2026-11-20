"""Turns the model outputs into one Buy / Hold / Sell call.

Three ways to decide, best first:
  stacked    the meta-model's probability (it has learned how far to trust each model)
  agreement  Buy/Sell only when both base models agree (meta-model unavailable)
  gbm_only   gradient boosting alone while the LSTM is still training (provisional)
"""

BUY_PROB = 0.55
SELL_PROB = 0.45


def _skill(metrics: dict) -> float:
    """0 when the backtest is no better than guessing, 1 at a 10-point edge."""
    edge = metrics["accuracy"] - max(metrics["baseline"], 1 - metrics["baseline"])
    return min(max(edge / 0.10, 0.0), 1.0)


def _direction(prob: float) -> int:
    return 1 if prob >= BUY_PROB else -1 if prob <= SELL_PROB else 0


def _strength(prob: float) -> float:
    return min(abs(prob - 0.5) * 4, 1.0)  # 0.75 probability → full strength


SIGNAL = {1: "Buy", -1: "Sell", 0: "Hold"}


def combine(prob_up: float, gbm_metrics: dict, lstm_return: float | None, lstm_metrics: dict | None,
            typical_move: float, stacked_prob: float | None = None, stack_metrics: dict | None = None) -> dict:
    gbm_dir = _direction(prob_up)

    if lstm_return is None or lstm_metrics is None:
        return {
            "signal": SIGNAL[gbm_dir],
            "confidence": round(_strength(prob_up) * (0.5 + 0.5 * _skill(gbm_metrics)) * 0.5, 3),
            "provisional": True,
            "agreement": None,
            "method": "gbm_only",
        }

    lstm_dir = 1 if lstm_return > 0 else -1
    agree = gbm_dir == lstm_dir
    agreement = "neutral" if gbm_dir == 0 else "agree" if agree else "disagree"

    if stacked_prob is not None and stack_metrics is not None:
        return {
            "signal": SIGNAL[_direction(stacked_prob)],
            "confidence": round(_strength(stacked_prob) * (0.5 + 0.5 * _skill(stack_metrics)), 3),
            "provisional": False,
            "agreement": agreement,
            "method": "stacked",
        }

    gbm_part = _strength(prob_up) * (0.5 + 0.5 * _skill(gbm_metrics))
    lstm_strength = min(abs(lstm_return) / typical_move, 1.0) if typical_move > 0 else 0.0
    lstm_part = lstm_strength * (0.5 + 0.5 * _skill(lstm_metrics))
    confidence = (gbm_part + lstm_part) / 2
    if not agree:
        confidence *= 0.5
    return {
        "signal": SIGNAL[gbm_dir] if agree else "Hold",
        "confidence": round(confidence, 3),
        "provisional": False,
        "agreement": agreement,
        "method": "agreement",
    }

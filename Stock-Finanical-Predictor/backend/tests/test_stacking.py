import numpy as np
import pandas as pd
import pytest

from app.features import build_dataset
from app.models import stacking
from app.modes import MODES


@pytest.fixture
def stacked_inputs(daily_prices):
    """Out-of-fold predictions where gradient boosting knows something and the
    LSTM is pure noise, over the later part of the history."""
    mode = MODES["swing"]
    X, fwd_ret = build_dataset(daily_prices, mode)
    labeled = fwd_ret.dropna()
    oof_rows = labeled.index[len(labeled) // 4:]
    rng = np.random.default_rng(1)
    truth = np.sign(labeled[oof_rows].to_numpy())
    gbm_oof = pd.Series(1 / (1 + np.exp(-(1.5 * truth + rng.normal(0, 1, len(oof_rows))))), index=oof_rows)
    lstm_oof = pd.Series(rng.normal(0, 0.02, len(oof_rows)), index=oof_rows)
    return gbm_oof, lstm_oof, X, fwd_ret, mode


def test_meta_model_trusts_the_informative_model(stacked_inputs):
    stack = stacking.train(*stacked_inputs)
    assert stack is not None
    coef = stack["coefficients"]
    assert abs(coef["gbm"]) > 3 * abs(coef["lstm"])
    assert coef["gbm"] > 0

    cmp = stack["comparison"]
    assert cmp["stacked"] > cmp["naive"]
    assert cmp["stacked"] >= cmp["lstm"]
    assert stack["metrics"]["beats_baseline"]
    assert stack["earned_trust"]


def test_meta_model_never_bets_against_a_model(stacked_inputs):
    gbm_oof, _, X, fwd_ret, mode = stacked_inputs
    # An LSTM that is reliably wrong: an unconstrained model would learn to invert it.
    contrarian = -(gbm_oof - 0.5)
    stack = stacking.train(gbm_oof, contrarian, X, fwd_ret, mode)
    assert stack["coefficients"]["gbm"] > 0
    assert stack["coefficients"]["lstm"] == pytest.approx(0, abs=1e-6)


def test_noise_models_do_not_earn_trust(stacked_inputs):
    gbm_oof, lstm_oof, X, fwd_ret, mode = stacked_inputs
    rng = np.random.default_rng(7)
    noise = pd.Series(rng.uniform(0.3, 0.7, len(gbm_oof)), index=gbm_oof.index)
    stack = stacking.train(noise, lstm_oof, X, fwd_ret, mode)
    assert not stack["earned_trust"]


def test_predict_returns_probability_and_weights(stacked_inputs):
    gbm_oof, lstm_oof, X, fwd_ret, mode = stacked_inputs
    stack = stacking.train(gbm_oof, lstm_oof, X, fwd_ret, mode)
    bullish, weights = stacking.predict(stack, 0.9, 0.0, X.iloc[[-1]])
    bearish, _ = stacking.predict(stack, 0.1, 0.0, X.iloc[[-1]])
    assert 0 <= bearish < 0.5 < bullish <= 1
    assert weights.keys() == {"gbm", "lstm"}
    assert abs(weights["gbm"]) > abs(weights["lstm"])


def test_too_few_shared_rows_returns_none(stacked_inputs):
    gbm_oof, lstm_oof, X, fwd_ret, mode = stacked_inputs
    # The two models' predictions barely overlap in time.
    assert stacking.train(gbm_oof.iloc[:100], lstm_oof.iloc[-100:], X, fwd_ret, mode) is None

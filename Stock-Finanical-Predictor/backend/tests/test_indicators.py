import numpy as np
import pandas as pd
import pytest

from app import indicators as ind


def test_sma_matches_hand_calculation():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ind.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert ind.sma(s, 3).isna().sum() == 2


def test_rsi_extremes_and_balance():
    rising = pd.Series(np.arange(1.0, 40.0))
    assert ind.rsi(rising).dropna().eq(100).all()

    falling = pd.Series(np.arange(40.0, 1.0, -1))
    assert ind.rsi(falling).dropna().lt(1e-9).all()

    # Equal-sized gains and losses → RSI hovers around 50.
    zigzag = pd.Series([100.0 + (i % 2) for i in range(200)])
    assert ind.rsi(zigzag).iloc[-1] == pytest.approx(50, abs=5)


def test_rsi_matches_hand_calculation():
    # First 14 moves: seven +1s and seven -1s → avg gain 0.5, avg loss 0.5 → RSI 50.
    # Next move +1: avg gain (0.5*13 + 1)/14 = 7.5/14, avg loss 6.5/14
    #   → RS = 7.5/6.5 → RSI = 100 - 100/(1 + 15/13) = 53.571...
    deltas = [1, -1] * 7 + [1]
    s = pd.Series(np.cumsum([100.0] + deltas))
    values = ind.rsi(s).dropna()
    assert values.iloc[0] == pytest.approx(50.0)
    assert values.iloc[1] == pytest.approx(100 - 100 / (1 + 15 / 13))


def test_macd_is_zero_for_flat_prices():
    flat = pd.Series([50.0] * 100)
    m = ind.macd(flat).dropna()
    assert np.allclose(m.to_numpy(), 0)


def test_vwap_restarts_each_session(intraday_prices):
    v = ind.vwap(intraday_prices)
    first_bars = intraday_prices.groupby(intraday_prices.index.normalize()).head(1)
    typical = (first_bars["High"] + first_bars["Low"] + first_bars["Close"]) / 3
    assert np.allclose(v.loc[first_bars.index], typical)


def test_drawdown_is_never_positive(daily_prices):
    dd = ind.drawdown(daily_prices["Close"], 252)
    assert (dd <= 1e-12).all()

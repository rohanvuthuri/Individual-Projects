import numpy as np
import pandas as pd
import pytest

from app.data import NotEnoughData
from app.features import build_dataset, build_features, forward_log_return
from app.modes import MODES


@pytest.mark.parametrize("mode_key", ["long", "swing"])
def test_features_do_not_look_ahead(daily_prices, mode_key):
    """Deleting the future must not change any feature in the past."""
    mode = MODES[mode_key]
    full = build_features(daily_prices, mode)
    cut = 1000
    partial = build_features(daily_prices.iloc[:cut], mode)
    pd.testing.assert_frame_equal(full.iloc[:cut], partial)


def test_intraday_features_do_not_look_ahead(intraday_prices):
    mode = MODES["day"]
    full = build_features(intraday_prices, mode)
    cut = 78 * 20 + 30  # partway through a session
    partial = build_features(intraday_prices.iloc[:cut], mode)
    pd.testing.assert_frame_equal(full.iloc[:cut], partial)


def test_label_is_forward_return(daily_prices):
    mode = MODES["swing"]
    y = forward_log_return(daily_prices, mode)
    close = daily_prices["Close"]
    assert y.iloc[0] == pytest.approx(np.log(close.iloc[mode.horizon] / close.iloc[0]))
    assert y.iloc[-mode.horizon:].isna().all()
    assert y.iloc[: -mode.horizon].notna().all()


def test_intraday_labels_stop_at_session_close(intraday_prices):
    mode = MODES["day"]
    y = forward_log_return(intraday_prices, mode)
    # The last `horizon` bars of every session would cross overnight → no label.
    per_session = y.groupby(intraday_prices.index.normalize())
    assert all(g.iloc[-mode.horizon:].isna().all() for _, g in per_session)
    assert all(g.iloc[: -mode.horizon].notna().all() for _, g in list(per_session)[:-1])


def test_short_history_raises(daily_prices):
    with pytest.raises(NotEnoughData):
        build_dataset(daily_prices.iloc[:300], MODES["long"])

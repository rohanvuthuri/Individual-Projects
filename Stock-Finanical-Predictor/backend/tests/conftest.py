import numpy as np
import pandas as pd
import pytest


def make_prices(n: int, freq: str = "B", seed: int = 0) -> pd.DataFrame:
    """Random-walk OHLCV bars indexed in New York time, like yfinance returns."""
    rng = np.random.default_rng(seed)
    if freq == "5min":
        days = pd.bdate_range("2026-01-05", periods=n // 78 + 1, tz="America/New_York")
        index = pd.DatetimeIndex([
            d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=5 * i)
            for d in days for i in range(78)
        ])[:n]
    else:
        index = pd.bdate_range("2018-01-02", periods=n, tz="America/New_York")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) * 1.003,
        "Low": np.minimum(open_, close) * 0.997,
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=index)


@pytest.fixture
def daily_prices() -> pd.DataFrame:
    return make_prices(1500)


@pytest.fixture
def intraday_prices() -> pd.DataFrame:
    return make_prices(78 * 30, freq="5min")

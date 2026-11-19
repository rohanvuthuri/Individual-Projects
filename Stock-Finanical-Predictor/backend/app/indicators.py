"""Technical indicators written with plain pandas.

Every function only looks backwards (rolling windows, exponential averages),
so the value at row t never depends on rows after t.
"""

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_mean(x: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing: starts at the simple average of the first `window`
    values, then each step is (previous * (window - 1) + new) / window."""
    seed = x.rolling(window, min_periods=window).mean()
    start = seed.notna().to_numpy().argmax() if seed.notna().any() else len(x)
    seeded = x.copy().astype(float)
    seeded.iloc[: start + 1] = np.nan
    if start < len(x):
        seeded.iloc[start] = seed.iloc[start]
    return seeded.ewm(alpha=1 / window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_mean(gain, window)
    avg_loss = wilder_mean(loss, window)
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # No losses at all in the window means RSI is 100.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    upper, lower = mid + k * std, mid - k * std
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_mid": mid,
        "bb_lower": lower,
        "bb_pctb": (close - lower) / (upper - lower),
        "bb_width": (upper - lower) / mid,
    })


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return wilder_mean(true_range, window)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price that restarts every trading session."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    session = df.index.normalize()
    pv = (typical * df["Volume"]).groupby(session).cumsum()
    vol = df["Volume"].groupby(session).cumsum()
    return pv / vol.replace(0, np.nan)


def volatility(close: pd.Series, window: int) -> pd.Series:
    return np.log(close).diff().rolling(window, min_periods=window).std()


def drawdown(close: pd.Series, window: int) -> pd.Series:
    """How far below the rolling high the price is (0 = at the high, -0.2 = 20% below)."""
    high = close.rolling(window, min_periods=1).max()
    return close / high - 1

"""Turns price bars into model inputs (features) and answers (labels).

Features at row t only use bars up to and including t. The label at row t is
the return from t to t + horizon, so the last `horizon` rows have no label;
those are the rows we make live predictions for.
"""

import numpy as np
import pandas as pd

from . import indicators as ind
from .data import NotEnoughData
from .modes import Mode

MIN_TRAIN_ROWS = 300

FEATURE_LABELS = {
    "ret_1": "1-bar return",
    "ret_5": "5-bar return",
    "ret_20": "20-bar return",
    "ret_60": "60-bar return",
    "vol_20": "20-bar volatility",
    "vol_60": "60-bar volatility",
    "rsi_14": "RSI (14)",
    "macd": "MACD line",
    "macd_hist": "MACD histogram",
    "bb_pctb": "Bollinger %B",
    "bb_width": "Bollinger width",
    "atr_pct": "ATR (% of price)",
    "dist_sma_20": "Distance from SMA 20",
    "dist_sma_50": "Distance from SMA 50",
    "dist_sma_200": "Distance from SMA 200",
    "drawdown_252": "Drawdown from 252-bar high",
    "volume_ratio": "Volume vs 20-bar average",
    "obv_trend": "On-balance volume trend",
    "dist_vwap": "Distance from VWAP",
    "bar_of_session": "Time of day",
}


def build_features(df: pd.DataFrame, mode: Mode) -> pd.DataFrame:
    close = df["Close"]
    f = pd.DataFrame(index=df.index)

    for n in (1, 5, 20, 60):
        f[f"ret_{n}"] = close.pct_change(n)
    f["vol_20"] = ind.volatility(close, 20)
    f["vol_60"] = ind.volatility(close, 60)
    f["rsi_14"] = ind.rsi(close, 14)

    m = ind.macd(close)
    f["macd"] = m["macd"] / close
    f["macd_hist"] = m["macd_hist"] / close

    bb = ind.bollinger(close)
    f["bb_pctb"] = bb["bb_pctb"]
    f["bb_width"] = bb["bb_width"]
    f["atr_pct"] = ind.atr(df) / close

    for n in (20, 50, 200):
        f[f"dist_sma_{n}"] = close / ind.sma(close, n) - 1
    f["drawdown_252"] = ind.drawdown(close, 252)

    vol = df["Volume"].astype(float)
    f["volume_ratio"] = vol / vol.rolling(20, min_periods=20).mean()
    f["obv_trend"] = ind.obv(df).diff(20) / vol.rolling(20, min_periods=20).sum()

    if mode.interval.endswith("m"):
        f["dist_vwap"] = close / ind.vwap(df) - 1
        f["bar_of_session"] = df.groupby(df.index.normalize()).cumcount()

    return f.replace([np.inf, -np.inf], np.nan)


def forward_log_return(df: pd.DataFrame, mode: Mode) -> pd.Series:
    close = df["Close"]
    fwd = np.log(close.shift(-mode.horizon) / close)
    if mode.interval.endswith("m"):
        # Day traders close out by the end of the session, so ignore labels
        # whose horizon would run past the close into the next day.
        session = pd.Series(df.index.normalize(), index=df.index)
        fwd = fwd.where(session.shift(-mode.horizon) == session)
    return fwd


def build_dataset(df: pd.DataFrame, mode: Mode) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (features, forward log return). Rows with incomplete features are dropped;
    the label is NaN where the future isn't known yet."""
    X = build_features(df, mode)
    y = forward_log_return(df, mode)
    ok = X.notna().all(axis=1)
    X, y = X[ok], y[ok]
    if y.notna().sum() < MIN_TRAIN_ROWS:
        raise NotEnoughData(
            f"Not enough price history to train a {mode.label.lower()} model "
            f"(need at least {MIN_TRAIN_ROWS} usable bars)."
        )
    return X, y

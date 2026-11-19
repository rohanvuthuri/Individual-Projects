"""FastAPI server for the Stock Financial Predictor.

Run from the backend folder:  uvicorn app.main:app --reload
"""

import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import data, indicators as ind
from .data import NotEnoughData, TickerNotFound
from .features import build_dataset
from .fundamentals import get_fundamentals
from .models import gbm, lstm, registry
from .modes import MODES, get_mode
from .signals import combine

app = FastAPI(title="Stock Financial Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(TickerNotFound)
def _not_found(_: Request, exc: TickerNotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(NotEnoughData)
def _not_enough(_: Request, exc: NotEnoughData):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ValueError)
def _bad_request(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _points(series: pd.Series) -> list[dict]:
    times = data.to_chart_time(series.index)
    return [
        {"time": t, "value": round(float(v), 4)}
        for t, v in zip(times, series.to_numpy())
        if not (isinstance(v, float) and math.isnan(v))
    ]


def _overlays(df: pd.DataFrame, mode_key: str) -> list[dict]:
    close = df["Close"]
    if mode_key == "long":
        lines = {
            "SMA 50": ind.sma(close, 50),
            "SMA 200": ind.sma(close, 200),
            "52-week high": close.rolling(252, min_periods=1).max(),
            "52-week low": close.rolling(252, min_periods=1).min(),
        }
    elif mode_key == "swing":
        bb = ind.bollinger(close)
        lines = {
            "EMA 20": ind.ema(close, 20),
            "EMA 50": ind.ema(close, 50),
            "Bollinger upper": bb["bb_upper"],
            "Bollinger lower": bb["bb_lower"],
        }
    else:
        lines = {
            "VWAP": ind.vwap(df),
            "EMA 9": ind.ema(close, 9),
            "EMA 21": ind.ema(close, 21),
        }
    return [{"label": label, "series": s} for label, s in lines.items()]


@app.get("/api/modes")
def modes():
    return [
        {"key": m.key, "label": m.label, "interval": m.interval, "horizon_label": m.horizon_label}
        for m in MODES.values()
    ]


@app.get("/api/search")
def search(q: str = Query("", max_length=50)):
    return data.search(q)


@app.get("/api/quote/{ticker}")
def quote(ticker: str):
    return data.get_quote(ticker)


@app.get("/api/history/{ticker}")
def history(ticker: str, mode: str = "swing"):
    m = get_mode(mode)
    df = data.get_prices(ticker, m)
    close = df["Close"]
    macd = ind.macd(close)
    overlays = _overlays(df, m.key)

    # Indicators are computed on the full history so they're warmed up, then trimmed.
    tail = slice(-m.chart_bars, None)
    view = df.iloc[tail]
    times = data.to_chart_time(view.index)
    return {
        "ticker": data.normalize_ticker(ticker),
        "mode": m.key,
        "interval": m.interval,
        "bars": [
            {"time": t, "open": round(o, 4), "high": round(h, 4), "low": round(lo, 4),
             "close": round(c, 4), "volume": float(v)}
            for t, o, h, lo, c, v in zip(
                times, view["Open"], view["High"], view["Low"], view["Close"], view["Volume"]
            )
        ],
        "overlays": [{"label": o["label"], "points": _points(o["series"].iloc[tail])} for o in overlays],
        "rsi": _points(ind.rsi(close).iloc[tail]),
        "macd": {
            "macd": _points(macd["macd"].iloc[tail]),
            "signal": _points(macd["macd_signal"].iloc[tail]),
            "hist": _points(macd["macd_hist"].iloc[tail]),
        },
    }


@app.get("/api/fundamentals/{ticker}")
def fundamentals(ticker: str):
    data.normalize_ticker(ticker)
    return get_fundamentals(ticker)


@app.get("/api/predict/{ticker}")
def predict(ticker: str, mode: str = "swing"):
    m = get_mode(mode)
    ticker = data.normalize_ticker(ticker)
    df = data.get_prices(ticker, m)
    X, fwd_ret = build_dataset(df, m)
    last_price = float(df["Close"].iloc[-1])
    typical_move = float(fwd_ret.dropna().abs().median())

    g = registry.get_gbm(ticker, m, X, fwd_ret, df["Close"])
    prob_up = gbm.predict(g, X.iloc[[-1]])

    status, l_bundle, l_error = registry.get_lstm(ticker, m, X, fwd_ret)
    lstm_out: dict = {"status": status, "error": l_error}
    lstm_return = None
    if l_bundle is not None:
        lstm_return = lstm.predict(l_bundle, X, m.lookback)
        lstm_out.update({
            "predicted_return": math.expm1(lstm_return),
            "target_price": last_price * math.exp(lstm_return),
            "metrics": l_bundle["metrics"],
            "trained_at": l_bundle["trained_at"].isoformat(),
        })

    decision = combine(
        prob_up, g["metrics"], lstm_return,
        l_bundle["metrics"] if l_bundle else None, typical_move,
    )
    return {
        "ticker": ticker,
        "mode": m.key,
        "horizon_label": m.horizon_label,
        "horizon_bars": m.horizon,
        "as_of": df.index[-1].isoformat(),
        "last_price": last_price,
        "typical_move": float(np.expm1(typical_move)),
        "expected_range": {
            "low": last_price * math.exp(-typical_move),
            "high": last_price * math.exp(typical_move),
        },
        **decision,
        "gbm": {
            "prob_up": prob_up,
            "metrics": g["metrics"],
            "importances": g["importances"],
            "equity_curve": g["equity_curve"],
            "trained_at": g["trained_at"].isoformat(),
        },
        "lstm": lstm_out,
    }

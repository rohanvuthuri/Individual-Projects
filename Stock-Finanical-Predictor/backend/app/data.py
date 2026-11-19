"""Downloads prices from Yahoo Finance (via yfinance) with a small in-memory cache."""

import re
import threading
import time
from typing import Any, Callable

import pandas as pd
import yfinance as yf

from .modes import Mode

TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")


class TickerNotFound(Exception):
    pass


class NotEnoughData(Exception):
    pass


_cache: dict[tuple, tuple[float, Any]] = {}
_lock = threading.Lock()


def _cached(key: tuple, ttl_seconds: float, load: Callable[[], Any]) -> Any:
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl_seconds:
            return hit[1]
    value = load()
    with _lock:
        _cache[key] = (now, value)
    return value


def normalize_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if not TICKER_RE.match(t):
        raise TickerNotFound(f"'{ticker}' is not a valid ticker symbol")
    return t


def get_prices(ticker: str, mode: Mode) -> pd.DataFrame:
    """OHLCV bars for the mode, oldest first, indexed by exchange-local time."""
    ticker = normalize_ticker(ticker)

    def load() -> pd.DataFrame:
        df = yf.Ticker(ticker).history(period=mode.period, interval=mode.interval, auto_adjust=True)
        if df is None or df.empty:
            raise TickerNotFound(f"No price data found for '{ticker}'")
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        return df[~df.index.duplicated(keep="last")].sort_index()

    return _cached(("prices", ticker, mode.key), mode.data_ttl.total_seconds(), load)


def get_info(ticker: str) -> dict:
    ticker = normalize_ticker(ticker)

    def load() -> dict:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        return info

    return _cached(("info", ticker), 6 * 3600, load)


def get_quote(ticker: str) -> dict:
    ticker = normalize_ticker(ticker)

    def load() -> dict:
        daily = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=True)
        if daily is None or daily.empty:
            raise TickerNotFound(f"No price data found for '{ticker}'")
        info = get_info(ticker)
        price = info.get("regularMarketPrice") or float(daily["Close"].iloc[-1])
        prev = info.get("regularMarketPreviousClose") or (
            float(daily["Close"].iloc[-2]) if len(daily) > 1 else price
        )
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName") or ticker,
            "exchange": info.get("fullExchangeName") or info.get("exchange"),
            "currency": info.get("currency", "USD"),
            "price": float(price),
            "previous_close": float(prev),
            "change": float(price - prev),
            "change_pct": float((price / prev - 1) * 100) if prev else 0.0,
            "market_state": info.get("marketState"),
        }

    return _cached(("quote", ticker), 30, load)


def search(query: str) -> list[dict]:
    query = query.strip()
    if not query:
        return []

    def load() -> list[dict]:
        try:
            quotes = yf.Search(query, max_results=8, news_count=0).quotes
        except Exception:
            return []
        return [
            {
                "symbol": q["symbol"],
                "name": q.get("longname") or q.get("shortname") or q["symbol"],
                "exchange": q.get("exchDisp") or q.get("exchange"),
                "type": q.get("quoteType"),
            }
            for q in quotes
            if q.get("symbol") and q.get("quoteType") in {"EQUITY", "ETF", "INDEX", "MUTUALFUND", "CRYPTOCURRENCY"}
        ]

    return _cached(("search", query.lower()), 3600, load)


def to_chart_time(index: pd.DatetimeIndex) -> list[int]:
    """Unix seconds of the exchange-local wall clock.

    lightweight-charts shows times in UTC, so we drop the timezone first so the
    chart displays New York time for US stocks instead of UTC.
    """
    if index.tz is not None:
        index = index.tz_localize(None)
    # as_unit: pandas may store times in s, ms, us or ns depending on the source.
    return index.as_unit("s").asi8.astype(int).tolist()

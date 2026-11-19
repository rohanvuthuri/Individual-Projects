"""Long-term scorecard from a company's current financial ratios.

These are today's values only (yfinance has no point-in-time history), so they
are shown to the user but never used to train the models; training on them
would leak future information into the past.
"""

from typing import Callable

from .data import get_info

Rating = str  # "good" | "neutral" | "poor" | "info"


def _band(good: Callable[[float], bool], poor: Callable[[float], bool]) -> Callable[[float], Rating]:
    return lambda v: "good" if good(v) else "poor" if poor(v) else "neutral"


def _pe(v: float) -> Rating:
    if v <= 0:
        return "poor"
    return "good" if v < 15 else "poor" if v > 30 else "neutral"


# (info key, label, format, rater, explanation)
METRICS: list[tuple[str, str, str, Callable[[float], Rating] | None, str]] = [
    ("trailingPE", "P/E (trailing)", "ratio", _pe, "Price paid per $1 of last year's earnings. Under 15 is cheap, over 30 is expensive."),
    ("forwardPE", "P/E (forward)", "ratio", _pe, "Price per $1 of next year's expected earnings."),
    ("trailingPegRatio", "PEG ratio", "ratio", _band(lambda v: 0 < v < 1, lambda v: v > 2 or v <= 0), "P/E divided by growth rate. Under 1 suggests growth is cheap."),
    ("priceToBook", "Price / book", "ratio", _band(lambda v: 0 < v < 1.5, lambda v: v >= 5 or v <= 0), "Price compared with the company's net assets."),
    ("debtToEquity", "Debt / equity", "pct_raw", _band(lambda v: v < 50, lambda v: v >= 150), "Debt as a % of shareholder equity. Lower means less financial risk."),
    ("returnOnEquity", "Return on equity", "pct", _band(lambda v: v > 0.15, lambda v: v < 0.05), "Profit generated per $1 of shareholder equity."),
    ("profitMargins", "Profit margin", "pct", _band(lambda v: v > 0.15, lambda v: v < 0.05), "Share of revenue kept as profit."),
    ("revenueGrowth", "Revenue growth (YoY)", "pct", _band(lambda v: v > 0.10, lambda v: v < 0), "Sales growth versus a year ago."),
    ("earningsGrowth", "Earnings growth (YoY)", "pct", _band(lambda v: v > 0.10, lambda v: v < 0), "Profit growth versus a year ago."),
    ("dividendYield", "Dividend yield", "pct_raw", None, "Yearly dividends as a % of the share price."),
    ("beta", "Beta", "ratio", None, "Volatility versus the market. 1 moves with the market, above 1 swings more."),
]


def get_fundamentals(ticker: str) -> dict:
    info = get_info(ticker)
    metrics = []
    for key, label, fmt, rate, explanation in METRICS:
        value = info.get(key)
        if not isinstance(value, (int, float)):
            continue
        metrics.append({
            "key": key,
            "label": label,
            "value": float(value),
            "format": fmt,
            "rating": rate(float(value)) if rate else "info",
            "explanation": explanation,
        })

    rated = [m for m in metrics if m["rating"] != "info"]
    score = (
        (sum(m["rating"] == "good" for m in rated) - sum(m["rating"] == "poor" for m in rated)) / len(rated)
        if rated else 0.0
    )
    summary = "Strong" if score >= 0.3 else "Weak" if score <= -0.3 else "Mixed"

    return {
        "available": bool(metrics),
        "summary": summary if rated else None,
        "score": score,
        "metrics": metrics,
        "profile": {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "week52_low": info.get("fiftyTwoWeekLow"),
            "week52_high": info.get("fiftyTwoWeekHigh"),
            "analyst_target": info.get("targetMeanPrice"),
            "analyst_rating": info.get("recommendationKey"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
        },
    }

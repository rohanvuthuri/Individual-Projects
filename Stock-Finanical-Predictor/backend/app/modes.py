"""Settings for each investing style.

Every mode answers the same question ("will the price be higher after N bars?")
but with a different bar size, amount of history and horizon N.
"""

from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    interval: str          # yfinance bar size
    period: str            # how much history yfinance returns
    horizon: int           # bars ahead we predict
    horizon_label: str     # human-readable horizon for the UI
    lookback: int          # bars the LSTM sees per sample
    model_max_age: timedelta  # retrain models older than this
    data_ttl: timedelta    # how long downloaded prices stay cached
    chart_bars: int        # bars sent to the chart


MODES: dict[str, Mode] = {
    "long": Mode(
        key="long",
        label="Long-term investing",
        interval="1d",
        period="10y",
        horizon=126,
        horizon_label="~6 months",
        lookback=60,
        model_max_age=timedelta(days=7),
        data_ttl=timedelta(hours=6),
        chart_bars=1260,
    ),
    "swing": Mode(
        key="swing",
        label="Short-term / swing",
        interval="1d",
        period="5y",
        horizon=10,
        horizon_label="~2 weeks",
        lookback=60,
        model_max_age=timedelta(days=1),
        data_ttl=timedelta(hours=1),
        chart_bars=250,
    ),
    "day": Mode(
        key="day",
        label="Day trading",
        interval="5m",
        period="60d",  # yfinance's limit for 5-minute bars
        horizon=12,
        horizon_label="~1 hour",
        lookback=48,
        model_max_age=timedelta(hours=1),
        data_ttl=timedelta(minutes=1),
        chart_bars=78 * 5,  # ~5 trading sessions
    ),
}


def get_mode(key: str) -> Mode:
    try:
        return MODES[key]
    except KeyError:
        raise ValueError(f"Unknown mode '{key}'. Use one of: {', '.join(MODES)}")

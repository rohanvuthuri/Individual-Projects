import pandas as pd
import pytest

from app.data import to_chart_time


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_chart_time_is_unix_seconds_of_local_wall_clock(unit):
    index = pd.DatetimeIndex(
        ["2026-11-18 09:30", "2026-11-18 09:35"], tz="America/New_York"
    ).as_unit(unit)
    # 09:30 New York shown as 09:30 on a UTC-based chart.
    expected = int(pd.Timestamp("2026-11-18 09:30", tz="UTC").timestamp())
    assert to_chart_time(index) == [expected, expected + 300]

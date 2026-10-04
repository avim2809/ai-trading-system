from __future__ import annotations

import numpy as np
import pandas as pd

from firm.data.qa import expected_trading_days, run_all


def _bars():
    import datetime as dt

    idx = expected_trading_days(dt.date(2018, 1, 2), dt.date(2019, 12, 31))
    rng = np.random.default_rng(0)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.005, len(idx)))
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "adjusted_close": px, "volume": 1e6}, index=idx.rename("date"))
    df["segment"] = 0
    df.iloc[400, df.columns.get_loc("adjusted_close")] *= 1.5
    return df.drop(df.index[200])


def test_qa_never_modifies_input_and_reports():
    series = {"AAA": _bars()}
    before = {k: v.copy(deep=True) for k, v in series.items()}
    f = run_all(series)
    for k, bars in series.items():
        pd.testing.assert_frame_equal(bars, before[k])
    checks = {(x.check, x.severity) for x in f}
    assert ("missing_days", "flag") in checks and ("spike", "flag") in checks
    assert all(x.symbol == "AAA" for x in f)

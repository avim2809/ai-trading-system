from __future__ import annotations

import numpy as np
import pandas as pd

from firm.data.qa import check_roll_gaps


def _series(adjust: bool):
    idx = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(3)
    px = 100 + np.cumsum(rng.normal(0, 0.5, 120))
    roll_pos = 60
    raw = px.copy()
    raw[roll_pos:] += 6.0  # contract switch: next contract trades 6 points higher
    adj = raw.copy()
    if adjust:
        adj[roll_pos:] -= 6.0  # Panama (additive) back-adjustment absorbs the gap
    return pd.DataFrame({"adjusted": adj}, index=idx.rename("date")), pd.DataFrame({"date": [idx[roll_pos]]})


def test_unadjusted_roll_jump_flagged():
    bars, rolls = _series(adjust=False)
    f = check_roll_gaps(bars, rolls)
    assert len(f) == 1 and f[0].severity == "flag" and f[0].date == rolls["date"].iloc[0].date()


def test_panama_adjusted_passes():
    bars, rolls = _series(adjust=True)
    assert check_roll_gaps(bars, rolls) == []

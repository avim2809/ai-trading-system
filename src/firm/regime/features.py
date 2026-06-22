"""Stationary feature engineering for HMM regime detection.

Raw prices are non-stationary, so the HMM is trained on stationarised
features (Chen, Yi & Zhao, 2020, §3.3):

* daily log return                ``r_t = ln(P_t / P_{t-1})``
* 5-day cumulative log return     ``R_{5,t} = ln(P_t / P_{t-5})``
* 14-period Average True Range    intraday volatility proxy
* 20-day volume-spike ratio       ``Volume_t / MA_20(Volume)``

The ATR computation mirrors the inline pattern already used in
:mod:`firm.strategies.volatility_breakout` and :mod:`firm.strategies.gann`
so the whole codebase computes true range the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Column order of the engineered feature matrix. Index 0 (``log_ret``) is the
#: return column used for mean-return regime labelling (see
#: :meth:`firm.regime.model.GaussianRegimeModel.label_map`).
REGIME_FEATURES: list[str] = ["log_ret", "log_ret_5d", "atr_14", "vol_spike"]

_ATR_PERIOD = 14
_VOL_MA_PERIOD = 20


def compute_regime_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Engineer the stationary HMM feature matrix from a single series' OHLCV.

    Args:
        ohlcv: DataFrame for **one** symbol, sorted (or sortable) by ``date``,
            with at least a ``close`` column.  ``high``/``low`` are used for
            ATR when present (otherwise ATR falls back to the absolute daily
            close-to-close move, keeping the feature defined for close-only
            series such as a synthetic market proxy).  ``volume`` drives the
            volume-spike ratio when present (otherwise it is held at 1.0).

    Returns:
        A DataFrame indexed like the input (rows with incomplete rolling
        windows dropped) with the four :data:`REGIME_FEATURES` columns.  The
        column order is guaranteed so callers can rely on column 0 being the
        return feature.
    """
    if ohlcv.empty or "close" not in ohlcv.columns:
        return pd.DataFrame(columns=REGIME_FEATURES)

    df = ohlcv.copy()
    if "date" in df.columns:
        df = df.sort_values("date")
    close = pd.to_numeric(df["close"], errors="coerce")

    log_ret = np.log(close / close.shift(1))
    log_ret_5d = np.log(close / close.shift(5))

    # True range: max(H-L, |H-C_prev|, |L-C_prev|).  Fall back to |ΔClose|
    # when high/low are unavailable so the feature stays defined.
    prev_close = close.shift(1)
    if "high" in df.columns and "low" in df.columns:
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
    else:
        true_range = (close - prev_close).abs()
    atr_14 = true_range.rolling(_ATR_PERIOD).mean()

    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        vol_ma = volume.rolling(_VOL_MA_PERIOD).mean()
        vol_spike = volume / vol_ma.replace(0.0, np.nan)
    else:
        vol_spike = pd.Series(1.0, index=df.index)

    features = pd.DataFrame(
        {
            "log_ret": log_ret,
            "log_ret_5d": log_ret_5d,
            "atr_14": atr_14,
            "vol_spike": vol_spike,
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features[REGIME_FEATURES]


def apply_laplace_smoothing(transmat: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Additive (Laplace) smoothing of an HMM transition matrix (§6.2).

    Guarantees every transition probability is strictly positive so that the
    forward algorithm cannot collapse to a zero-probability state on an
    out-of-sample observation.  Rows are renormalised to sum to 1.
    """
    smoothed = np.asarray(transmat, dtype=float) + epsilon
    return smoothed / smoothed.sum(axis=1, keepdims=True)

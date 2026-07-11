"""Machine-learning return-prediction strategy.

Financial intuition:
    Non-linear combinations of technical and fundamental features can
    capture cross-sectional return predictability that simple factor models
    miss.  Walk-forward training ensures the model only learns from past
    data, preserving point-in-time integrity (Gu, Kelly & Xiu, 2020).

Data inputs:
    Adjusted close prices and volumes from PitView.prices() for feature
    engineering (lagged returns, volatility, volume changes, momentum
    indicators).

Signal logic:
    1. Build a feature matrix per symbol using only data up to asof:
       - Lagged returns at multiple horizons (1d, 5d, 21d, 63d).
       - Realized volatility (21d, 63d).
       - Volume change ratio (5d mean / 21d mean).
       - Momentum (price vs 50d MA, 200d MA).
    2. Target: forward N-day return, constructed strictly from historical
       data where t + N <= asof - buffer (NO look-ahead).
    3. Train a GradientBoostingRegressor (or Ridge) on the historical
       training window using walk-forward methodology.
    4. Predict current-period returns for each symbol.
    5. Z-score predictions cross-sectionally to produce the signal.

Portfolio construction approach:
    Long high-predicted-return symbols, short low-predicted-return symbols.
    Scale positions by prediction confidence / z-score magnitude.

Risk notes:
    ML models can overfit, especially with small universes.  The walk-
    forward buffer prevents look-ahead but does not prevent overfitting
    to in-sample patterns.  Regularization and feature selection are
    essential.  Retrain frequency should be tuned to balance staleness
    vs. stability.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

logger = logging.getLogger(__name__)


def _build_features(pivot: pd.DataFrame) -> pd.DataFrame:
    """Construct feature matrix from a date × symbol price pivot table.

    Returns a stacked DataFrame indexed by (date, symbol) with feature columns.
    """
    ret_1d = pivot.pct_change(1)
    ret_5d = pivot.pct_change(5)
    ret_21d = pivot.pct_change(21)
    ret_63d = pivot.pct_change(63)

    vol_21d = ret_1d.rolling(21).std()
    vol_63d = ret_1d.rolling(63).std()

    ma50 = pivot.rolling(50).mean()
    ma200 = pivot.rolling(200).mean()
    price_vs_ma50 = (pivot / ma50) - 1.0
    price_vs_ma200 = (pivot / ma200) - 1.0

    features = {}
    for col in pivot.columns:
        sym_feat = pd.DataFrame(
            {
                "ret_1d": ret_1d[col],
                "ret_5d": ret_5d[col],
                "ret_21d": ret_21d[col],
                "ret_63d": ret_63d[col],
                "vol_21d": vol_21d[col],
                "vol_63d": vol_63d[col],
                "price_vs_ma50": price_vs_ma50[col],
                "price_vs_ma200": price_vs_ma200[col],
            }
        )
        sym_feat["symbol"] = col
        features[col] = sym_feat

    if not features:
        return pd.DataFrame()

    result = pd.concat(features.values())
    result.index.name = "date"
    return result.reset_index()


def _build_macro_features(pit_view) -> pd.DataFrame:
    """Build date-indexed macro feature columns from FRED data in the PIT store.

    Returns a DataFrame with a DatetimeIndex and columns:
      yield_curve_slope — daily T10Y2Y spread (level, in %)
      cpi_mom_3m        — 3-month CPI momentum (forward-filled to daily)

    Returns an empty DataFrame when the PIT store has no macro data, so the
    strategy degrades gracefully when FRED_API_KEY is absent.
    """
    if not hasattr(pit_view, "macro"):
        return pd.DataFrame()

    frames: dict[str, pd.Series] = {}

    yc = pit_view.macro("T10Y2Y", lookback_days=400)
    if not yc.empty:
        frames["yield_curve_slope"] = yc

    cpi = pit_view.macro("CPIAUCSL", lookback_days=400)
    if not cpi.empty:
        # 3-month (63 calendar-day) momentum: pct change over 91 days
        frames["cpi_mom_3m"] = cpi.pct_change(91)

    if not frames:
        return pd.DataFrame()

    macro_df = pd.DataFrame(frames)
    macro_df.index = pd.to_datetime(macro_df.index)
    return macro_df


@register("ml_prediction")
class MLPredictionStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("ml_prediction", params)
        self._model = None
        self._last_train_date = None

    def generate(self, pit_view: PitView) -> list[Signal]:
        model_type: str = self.params.get("model_type", "ridge")
        train_lookback_days: int = self.params.get("train_lookback_days", 504)
        predict_horizon: int = self.params.get("predict_horizon", 5)
        retrain_frequency: int = self.params.get("retrain_frequency", 21)
        buffer_days: int = self.params.get("buffer_days", 5)

        universe = pit_view.universe
        if not universe:
            return []

        needed_days = train_lookback_days + predict_horizon + buffer_days + 210
        prices_df = pit_view.prices(symbols=universe, lookback_days=needed_days)
        if prices_df.empty:
            return []

        pivot = (
            prices_df.pivot_table(index="date", columns="symbol", values="adj_close")
            .sort_index()
        )
        if len(pivot) < 252:
            return []

        feature_df = _build_features(pivot)
        if feature_df.empty:
            return []

        feature_df["date"] = pd.to_datetime(feature_df["date"])

        # Merge macro features (yield curve, CPI momentum) when available.
        macro_df = _build_macro_features(pit_view)
        macro_cols: list[str] = []
        if not macro_df.empty:
            macro_df = macro_df.reset_index().rename(columns={"index": "date"})
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            macro_cols = [c for c in macro_df.columns if c != "date"]
            feature_df = feature_df.merge(macro_df, on="date", how="left")
            # Forward-fill so monthly CPI is available on every trading day,
            # then zero-fill any remaining NaNs at the start of the series.
            for col in macro_cols:
                feature_df[col] = feature_df[col].ffill().fillna(0.0)

        fwd_ret = pivot.pct_change(predict_horizon).shift(-predict_horizon)
        fwd_ret_stacked = fwd_ret.stack().reset_index()
        fwd_ret_stacked.columns = ["date", "symbol", "fwd_return"]
        fwd_ret_stacked["date"] = pd.to_datetime(fwd_ret_stacked["date"])

        merged = feature_df.merge(fwd_ret_stacked, on=["date", "symbol"], how="left")

        feature_cols = [
            "ret_1d", "ret_5d", "ret_21d", "ret_63d",
            "vol_21d", "vol_63d", "price_vs_ma50", "price_vs_ma200",
            *macro_cols,
        ]

        asof_ts = pd.Timestamp(pit_view.asof)
        # Size the train window in TRADING days (rows), consistent with how the
        # forward-return label is built (pct_change/shift by trading rows).  A
        # calendar-day Timedelta would shrink the gap below the intended
        # no-look-ahead buffer and shorten the training set.
        trading_dates = pd.Index(sorted(merged["date"].unique()))
        gap = buffer_days + predict_horizon  # trading days to leave before asof
        cutoff_pos = len(trading_dates) - 1 - gap
        if cutoff_pos < 0:
            return []
        cutoff = trading_dates[cutoff_pos]
        train_start = trading_dates[max(0, cutoff_pos - train_lookback_days)]

        train_mask = (merged["date"] >= train_start) & (merged["date"] <= cutoff)
        train_data = merged.loc[train_mask].dropna(subset=feature_cols + ["fwd_return"])

        if len(train_data) < 50:
            return []

        X_train = train_data[feature_cols].values
        y_train = train_data["fwd_return"].values

        needs_retrain = (
            self._model is None
            or self._last_train_date is None
            or (asof_ts - self._last_train_date).days >= retrain_frequency
        )

        if needs_retrain:
            self._model = self._fit_model(model_type, X_train, y_train)
            self._last_train_date = asof_ts

        latest_dates = merged.groupby("symbol")["date"].max()
        pred_rows = []
        for symbol in universe:
            if symbol not in latest_dates.index:
                continue
            sym_latest = merged[
                (merged["symbol"] == symbol)
                & (merged["date"] == latest_dates[symbol])
            ]
            if sym_latest.empty:
                continue
            row = sym_latest[feature_cols].iloc[-1:]
            if row.isna().any(axis=1).iloc[0]:
                continue
            pred_rows.append((symbol, row.values))

        if not pred_rows:
            return []

        symbols = [s for s, _ in pred_rows]
        X_pred = np.vstack([r for _, r in pred_rows])

        try:
            predictions = self._model.predict(X_pred)
        except Exception:
            return []

        pred_series = pd.Series(predictions, index=symbols)
        mean = pred_series.mean()
        std = pred_series.std()
        if std == 0 or np.isnan(std):
            z_scores = pred_series * 0.0
        else:
            z_scores = ((pred_series - mean) / std).clip(-3, 3)

        signals: list[Signal] = []
        for symbol, z in z_scores.items():
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="ml_prediction",
                    score=float(z),
                    confidence=min(abs(float(z)) / 3.0, 1.0),
                    horizon=f"{predict_horizon}d",
                    asof=pit_view.asof,
                    meta={
                        "predicted_return": float(pred_series[symbol]),
                        "model_type": model_type,
                        "train_samples": len(train_data),
                    },
                )
            )
        return signals

    @staticmethod
    def _fit_model(model_type: str, X: np.ndarray, y: np.ndarray):
        """Train a sklearn model; supports 'ridge' and 'gbr'."""
        if model_type == "gbr":
            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        else:
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0)

        finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[finite_mask], y[finite_mask]
        if len(X) < 10:
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0)

        model.fit(X, y)
        return model

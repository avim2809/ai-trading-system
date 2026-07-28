"""Custom commission scheme for realistic cost modelling.

Commission is configured from ``config/settings.yaml`` backtest section
(``commission_pct``). Slippage (``slippage_pct``) is applied separately at
the broker via backtrader's native ``broker.set_slippage_perc`` in
``BacktestEngine.setup`` rather than folded into the commission scheme.

Bid-ask spread cost (``spread_pct``) IS folded in here rather than handled
like slippage: it's a per-trade, size-independent cost (the cost of
crossing the quoted spread) rather than a price-impact-from-size effect, so
it behaves exactly like an extra commission rate on every fill.
"""

from __future__ import annotations

import backtrader as bt


class PercentageCommission(bt.CommInfoBase):
    """Percentage-of-trade-value commission model, optionally including an
    additional bid-ask spread cost rate."""

    params = (
        ("commission", 0.001),  # 10 bps default
        ("spread_pct", 0.0),    # additional per-trade bid-ask spread cost
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        # CommInfoBase.__init__ silently divides `commission` by 100 when
        # commtype=COMM_PERC and percabs is falsy (its default) — i.e. it
        # assumes callers pass a *percentage points* value (e.g. "0.1" for
        # 0.1%). Every caller here passes a *fraction* (e.g. 0.001 for
        # 0.1%), matching commission_pct/slippage_pct's convention
        # throughout this codebase. Without percabs=True, real commission
        # charged on every fill was silently 100x smaller than configured —
        # a long-standing bug that made every backtest look far cheaper to
        # trade than the configured commission_pct actually implied.
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * (self.p.commission + self.p.spread_pct)

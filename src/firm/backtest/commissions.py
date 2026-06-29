"""Custom commission scheme for realistic cost modelling.

Commission is configured from ``config/settings.yaml`` backtest section
(``commission_pct``). Slippage (``slippage_pct``) is applied separately at
the broker via backtrader's native ``broker.set_slippage_perc`` in
``BacktestEngine.setup`` rather than folded into the commission scheme.
"""

from __future__ import annotations

import backtrader as bt


class PercentageCommission(bt.CommInfoBase):
    """Percentage-of-trade-value commission model."""

    params = (
        ("commission", 0.001),  # 10 bps default
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.commission

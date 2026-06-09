"""Custom commission and slippage schemes for realistic cost modelling.

Both models are configured from ``config/settings.yaml`` backtest section
(``commission_pct`` and ``slippage_pct``).
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


class FixedSlippage(bt.CommInfoBase):
    """Fixed percentage slippage applied as additional transaction cost.

    Backtrader doesn't expose a first-class ``SlippagePerc`` base in all
    versions.  This piggy-backs on the commission infrastructure: the
    slippage cost is added to the commission so the broker deducts it
    automatically.
    """

    params = (
        ("slippage", 0.0005),  # 5 bps default
        ("commission", 0.0),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * (self.p.commission + self.p.slippage)

"""Custom backtrader sizers.

``TargetWeightSizer`` converts target portfolio weights (from the agent
pipeline) into share quantities suitable for ``bt.Strategy.buy / sell``.
"""

from __future__ import annotations

import backtrader as bt


class TargetWeightSizer(bt.Sizer):
    """Size positions to achieve a target portfolio weight.

    The caller sets ``params.target_weight`` before placing the order.
    The sizer computes the number of shares required to move the position
    from its current market value to the desired weight of total
    portfolio value.
    """

    params = (("target_weight", 0.0),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        portfolio_value = self.strategy.broker.getvalue()
        target_value = portfolio_value * self.p.target_weight
        current_position = self.strategy.getposition(data).size
        current_value = current_position * data.close[0]
        diff_value = target_value - current_value
        if abs(diff_value) < 1.0:
            return 0
        return int(abs(diff_value / data.close[0]))

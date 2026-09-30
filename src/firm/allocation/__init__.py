"""Capital-sleeve allocation: multi-asset satellite sleeves alongside the
main 13-strategy engine (e.g. the BTC trend sleeve, ``btc_trend.py``).

Not to be confused with the existing per-strategy capital-sleeves feature
(``capital_allocation_mode: sleeved`` in the engine config, see
``docs/capital_sleeves_plan.md``) -- this package is for whole-sleeve
satellite allocations sized as a fraction of total NAV, per the ``Sleeve``
ABC in ``sleeves.py``.
"""

"""Research infrastructure (credibility plan): forward-data seal and fail-closed data access.

Importing this package has no side effects. It must never be imported by live modules
(``firm.live.*``, ``firm.api.*``, ``firm.runtime``, ``firm.data.pit_store``); see
``docs/HOLDOUT_POLICY.md`` and ``tests/test_live_import_isolation.py``.
"""

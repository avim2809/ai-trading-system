#!/usr/bin/env python
"""CLI to build a point-in-time (date, symbol) panel into the Parquet cache.

Thin runnable wrapper around :func:`firm.scripts_entry.fetch_data_main`. Works
without installation by adding ``src/`` to ``sys.path``.

Examples:
    python scripts/fetch_data.py --symbols AAPL,MSFT --start 2020-01-01 --end 2021-12-31
    python scripts/fetch_data.py --symbols AAPL --start 2020-01-01 --end 2020-12-31 \
        --no-sentiment --prices-provider tiingo
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from a checkout without `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.scripts_entry import fetch_data_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(fetch_data_main())

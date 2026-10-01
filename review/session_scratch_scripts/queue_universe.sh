#!/bin/bash
# Wait for the running extras download to finish (rate limit is shared), then fetch ETFs + US universe.
cd /local/store/git/ai-trading-system
while pgrep -f "fetch_eodhd_extras.py --only forex,crypto,actions" >/dev/null; do sleep 20; done
nice -n 10 .venv/bin/python3 scripts/fetch_eodhd_extras.py --only etfs,us_universe --workers 8

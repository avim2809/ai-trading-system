#!/bin/bash
# When the forex/crypto/actions run ends, restart the universe run alone at 15 req/s (resumable).
cd /local/store/git/ai-trading-system
S=/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad
while pgrep -f "fetch_eodhd_extras.py --only forex,crypto,actions" >/dev/null; do sleep 15; done
pid=$(pgrep -f "fetch_eodhd_extras.py --only etfs,us_universe --workers 8 --max-rps 8")
[ -n "$pid" ] && kill $pid && sleep 5
nice -n 10 .venv/bin/python3 scripts/fetch_eodhd_extras.py --only etfs,us_universe --workers 10 --max-rps 15 > $S/eodhd_universe2.log 2>&1
grep -E "items to fetch|extras final|Traceback" $S/eodhd_universe2.log | cut -c25-400

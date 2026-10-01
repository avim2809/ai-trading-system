#!/bin/bash
# Wait for the first allocation cycle on :8001 (after 15:30Z) and dump its outcome.
for i in $(seq 1 60); do
  out=$(curl -s -m 15 http://127.0.0.1:8001/api/live/status)
  ts=$(echo "$out" | python3 -c "import sys,json; d=json.load(sys.stdin); print((d.get('last_cycle') or {}).get('timestamp',''))" 2>/dev/null)
  st=$(echo "$out" | python3 -c "import sys,json; d=json.load(sys.stdin); print((d.get('allocation') or {}).get('day_status'))" 2>/dev/null)
  if [[ "$ts" > "2026-09-30T15:29" && "$st" != "None" && "$st" != "submitting" ]]; then
    sleep 60   # let market orders fill
    curl -s -m 15 http://127.0.0.1:8001/api/live/status
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT waiting for first allocation cycle"; curl -s -m 15 http://127.0.0.1:8001/api/live/status

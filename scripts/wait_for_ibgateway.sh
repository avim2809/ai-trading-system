#!/usr/bin/env bash
# Blocks (briefly) until IB Gateway's API port is accepting connections.
#
# ai-trading.service's After=/Wants=ibgateway.service only waits for the IB
# Gateway *process* to fork (systemd Type=simple) — not for IBC to finish its
# headless login and actually open the API port, which takes 30-60s and can
# run longer (2FA prompt, IBC/Gateway update). Without this, firm-api can
# start and attempt IBKRBroker.connect() before the port exists, fail, and
# (since nothing currently retries after boot) leave the live engine stopped
# until a human notices — see docs/PROJECT_CONTEXT.md "Broker & host failover".
#
# Always exits 0: this is a readiness *delay*, not a hard gate. firm-api
# serves far more than live trading (dashboard, backtests, RAG), so blocking
# the whole API indefinitely on IB Gateway would trade one failure mode for a
# worse one. If the timeout is hit, firm-api starts anyway and falls back to
# its existing documented behavior (IBKRBroker.connect()'s own 3x retry,
# then bootstrap_live_from_yaml() leaves the API running without a live
# engine rather than crashing) — this script just makes hitting that
# fallback far less likely, not impossible.
set -uo pipefail

HOST="${IBKR_HOST:-127.0.0.1}"
PORT="${IBKR_PORT:-4002}"
TIMEOUT_SECONDS="${IBGATEWAY_WAIT_TIMEOUT:-90}"
POLL_INTERVAL=2

elapsed=0
while ! nc -z "$HOST" "$PORT" 2>/dev/null; do
    if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
        echo "wait_for_ibgateway: timed out after ${TIMEOUT_SECONDS}s waiting for $HOST:$PORT — starting firm-api anyway (existing connect-retry/fallback behavior applies)"
        exit 0
    fi
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
done

echo "wait_for_ibgateway: $HOST:$PORT accepting connections after ${elapsed}s"
exit 0

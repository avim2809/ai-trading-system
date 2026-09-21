#!/usr/bin/env bash
# Backs up this host's live-trading state -- kill-switch state, live_state.db
# (portfolio/attribution history, cycle counter), decision memory, dynamic
# universe state, pending approvals, execution audit trail -- to a second
# directory on the same disk.
#
# This is *not* protection against a disk failure: there is only one disk on
# this host (see docs/PROJECT_CONTEXT.md "Durable live state" / "Losing the
# host entirely"). It only protects against an accidental deletion, a bad
# script, or a botched manual edit inside data/ or data_alpaca/ specifically
# -- still strictly better than the "no backup exists at all" status quo,
# but not a substitute for real off-box storage. Wiring up an off-box
# destination needs credentials/infrastructure (remote storage, a second
# host) this script has no business guessing at; see the same doc section
# for what that would take.
#
# Excludes data*/{vectordb,cache,logs,models} -- large and fully
# reproducible (RAG ingestion, provider re-fetch, model re-training, and log
# rotation respectively) -- so each backup stays a few MB instead of ~150MB.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_ROOT="${LIVE_STATE_BACKUP_DIR:-/local/store/backups/ai-trading-live-state}"
# Daily runs: 14 days gives two weeks of recovery points at negligible disk
# cost (each archive is a few MB -- see the exclude list above).
RETAIN_COUNT="${LIVE_STATE_BACKUP_RETAIN:-14}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${BACKUP_ROOT}/live_state_${TIMESTAMP}.tar.gz"

cd "$REPO_DIR"

targets=()
for d in data data_alpaca; do
    [ -d "$d" ] && targets+=("$d")
done
if [ "${#targets[@]}" -eq 0 ]; then
    echo "backup_live_state: no data/data_alpaca directory found here -- nothing to back up" >&2
    exit 0
fi

mkdir -p "$BACKUP_ROOT"

exclude_args=()
for d in "${targets[@]}"; do
    for sub in vectordb cache logs models; do
        exclude_args+=("--exclude=${d}/${sub}")
    done
done

tar -czf "$DEST" "${exclude_args[@]}" "${targets[@]}"

echo "backup_live_state: wrote $DEST ($(du -h "$DEST" | cut -f1))"

# Retention: keep only the newest $RETAIN_COUNT archives so this can run
# unattended indefinitely without slowly eating the same disk it's meant to
# protect against losing data on.
mapfile -t existing < <(ls -1t "${BACKUP_ROOT}"/live_state_*.tar.gz 2>/dev/null)
if [ "${#existing[@]}" -gt "$RETAIN_COUNT" ]; then
    for old in "${existing[@]:$RETAIN_COUNT}"; do
        rm -f "$old"
        echo "backup_live_state: pruned $old"
    done
fi

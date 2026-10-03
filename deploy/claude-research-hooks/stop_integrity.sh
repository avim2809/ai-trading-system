#!/usr/bin/env bash
# Stop hook for RESEARCH sessions only (ticket P0-04; owner decisions OD-06, OD-07).
#
# TEMPLATE: the owner installs this root-owned (mode 0755) at /etc/claude-code/hooks/stop_integrity.sh and registers it
# only in the research unix user's own root-owned ~/.claude/settings.json. Never from ${CLAUDE_PROJECT_DIR}.
#
# Contract (Claude Code blocks a Stop only on exit code 2 or JSON {"decision":"block"}; any other non-zero exit is a
# non-blocking error, so EVERY failure below maps to exit 2):
#   1. stop_hook_active=true on stdin  -> exit 0 (block at most once per turn, so a persistent failure cannot loop).
#   2. Not inside a research worktree  -> exit 0 immediately. In the live checkout (ops sessions, and the live .venv)
#      the hook never runs tests and never blocks.
#   3. Interpreter: <worktree>/.venv/bin/python, then the shared research venv <VENV_ROOT>/<depset>/bin/python
#      (<depset> from <worktree>/.research-depset, default "core"). It must import firm from <worktree>/src. The live
#      .venv is refused explicitly (its editable install points at the live src).
#   4. No safe interpreter: the research user fails closed (exit 2 with instructions); an ops session skips and appends
#      the skip to a durable log that the red-team review checks.
#   5. Run `nice -n 10 python -m pytest -q tests/integrity -x` under a timeout; any failure or timeout -> exit 2 with the
#      summary on stderr.
set -u

RESEARCH_WT_ROOT="/local/store/research/ai-trading-system/.claude/worktrees"
VENV_ROOT="/local/store/research-venvs"
LIVE_VENV_PREFIX="/local/store/git/ai-trading-system/.venv"
SKIP_LOG="/var/log/claude-research/stop_hook_skips.log"
RESEARCH_USER=research
TIMEOUT_SECONDS=110

stdin_json="$(cat 2>/dev/null || true)"
if printf '%s' "$stdin_json" | grep -Eq '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then
    exit 0
fi

project_dir="${CLAUDE_PROJECT_DIR:-}"
[ -n "$project_dir" ] || exit 0
wt="$(realpath -m "$project_dir" 2>/dev/null || true)"
root="$(realpath -m "$RESEARCH_WT_ROOT" 2>/dev/null || true)"
case "$wt" in
    "$root"/*) ;;
    *) exit 0 ;;
esac
# the worktree itself, even if the session started in a subdirectory
rel="${wt#"$root"/}"
wt="$root/${rel%%/*}"

depset="core"
if [ -r "$wt/.research-depset" ]; then
    depset="$(tr -d '[:space:]' < "$wt/.research-depset")"
    [ -n "$depset" ] || depset="core"
fi

is_safe() {  # $1 = candidate interpreter
    local py="$1" real
    [ -x "$py" ] || return 1
    real="$(realpath -m "$py" 2>/dev/null || true)"
    case "$real" in
        "$LIVE_VENV_PREFIX"/*) return 1 ;;
    esac
    PYTHONPATH="$wt/src" "$py" -c "import firm,sys; sys.exit(0 if firm.__file__.startswith('$wt/src') else 1)" \
        >/dev/null 2>&1
}

py=""
for candidate in "$wt/.venv/bin/python" "$VENV_ROOT/$depset/bin/python"; do
    if is_safe "$candidate"; then
        py="$candidate"
        break
    fi
done

if [ -z "$py" ]; then
    if [ "$(id -un)" = "$RESEARCH_USER" ]; then
        echo "stop_integrity: no safe interpreter for $wt (looked at $wt/.venv and $VENV_ROOT/$depset). The live .venv is refused. Ask the owner to create the shared research venv (scripts/new_research_worktree.sh --create-venv)." >&2
        exit 2
    fi
    mkdir -p "$(dirname "$SKIP_LOG")" 2>/dev/null || true
    printf '%s skipped integrity check: no safe interpreter for %s (user %s)\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$wt" "$(id -un)" >> "$SKIP_LOG" 2>/dev/null \
        || echo "stop_integrity: could not append to $SKIP_LOG" >&2
    exit 0
fi

cd "$wt" || exit 2
out="$(PYTHONPATH="$wt/src" PYTHONDONTWRITEBYTECODE=1 nice -n 10 timeout "$TIMEOUT_SECONDS" \
    "$py" -m pytest -q tests/integrity -x -p no:cacheprovider 2>&1)"
rc=$?
if [ "$rc" -ne 0 ]; then
    {
        echo "stop_integrity: tests/integrity failed (exit $rc; 124 = timeout). Fix the cause or report it; do not edit tests/integrity."
        printf '%s\n' "$out" | tail -n 40
    } >&2
    exit 2
fi
exit 0

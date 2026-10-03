#!/usr/bin/env bash
# Create one git worktree per ticket inside the RESEARCH clone (PLAN.md section 8, P0-06).
#
#   scripts/new_research_worktree.sh <TICKET_ID> [dependency-set] [--create-venv]
#
# Run as the research user from inside /local/store/research/ai-trading-system. It
#   * refuses to run inside the live checkout (a worktree there would share the refs the
#     running services are served from),
#   * refuses when less than FIRM_MIN_FREE_GB (default 20) GB are free on /,
#   * creates <clone>/.claude/worktrees/<TICKET_ID> on branch <prefix>/<TICKET_ID>
#     (prefix from FIRM_WORKTREE_BRANCH_PREFIX, default "ticket"; branches start at the
#     current HEAD, or at FIRM_WORKTREE_BASE if set),
#   * writes <worktree>/.research-depset (read by the research Stop hook, P0-04),
#   * checks the shared research venv /local/store/research-venvs/<set>/ (outside every repo,
#     NON-editable install) and, only with --create-venv, builds it and installs pytest-xdist
#     into that venv. Nothing is ever installed into the live .venv.
#
# Tests then run with the worktree's source first on the path:
#   PYTHONPATH=<worktree>/src nice -n 10 ionice -c3 <venv>/bin/python -m pytest -q -n 2 ...
# (xdist at most -n 2; heavy runs may add `prlimit --as=<bytes>`).
set -euo pipefail

LIVE_CHECKOUT="${FIRM_LIVE_CHECKOUT:-/local/store/git/ai-trading-system}"
MIN_FREE_GB="${FIRM_MIN_FREE_GB:-20}"
VENV_ROOT="${FIRM_RESEARCH_VENV_ROOT:-/local/store/research-venvs}"
BRANCH_PREFIX="${FIRM_WORKTREE_BRANCH_PREFIX:-ticket}"

usage() {
    echo "usage: $0 <TICKET_ID> [dependency-set] [--create-venv]" >&2
    exit 2
}

ID=""
SET_NAME=""
CREATE_VENV=0
for arg in "$@"; do
    case "$arg" in
        --create-venv) CREATE_VENV=1 ;;
        -h|--help) usage ;;
        -*) echo "unknown option: $arg" >&2; usage ;;
        *)
            if [ -z "$ID" ]; then ID="$arg"
            elif [ -z "$SET_NAME" ]; then SET_NAME="$arg"
            else usage
            fi
            ;;
    esac
done
[ -n "$ID" ] || usage
SET_NAME="${SET_NAME:-core}"
case "$ID" in *[!A-Za-z0-9._-]*) echo "invalid ticket id: $ID" >&2; exit 2 ;; esac
case "$SET_NAME" in *[!A-Za-z0-9._-]*) echo "invalid dependency set: $SET_NAME" >&2; exit 2 ;; esac

TOP="$(git rev-parse --show-toplevel)"
if [ "$(realpath "$TOP")" = "$(realpath -m "$LIVE_CHECKOUT")" ]; then
    echo "never inside the live checkout ($LIVE_CHECKOUT)" >&2
    exit 1
fi

free_gb="$(df -BG --output=avail / | tail -1 | tr -dc 0-9)"
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "need ${MIN_FREE_GB}GB free on /, have ${free_gb}GB" >&2
    exit 1
fi

cd "$TOP"
WT=".claude/worktrees/$ID"
BRANCH="$BRANCH_PREFIX/$ID"
if [ -e "$WT" ]; then
    echo "worktree path already exists: $TOP/$WT" >&2
    exit 1
fi
git worktree add "$WT" -b "$BRANCH" ${FIRM_WORKTREE_BASE:+"$FIRM_WORKTREE_BASE"}
echo "$SET_NAME" > "$WT/.research-depset"

VENV="$VENV_ROOT/$SET_NAME"
if [ ! -x "$VENV/bin/python" ]; then
    if [ "$CREATE_VENV" -eq 1 ]; then
        python3 -m venv "$VENV"
        "$VENV/bin/python" -m pip install "$TOP"          # non-editable on purpose
        "$VENV/bin/python" -m pip install pytest-xdist     # this venv only, never pyproject/.venv
    else
        echo "NOTE: shared research venv $VENV is missing; re-run with --create-venv (or ask the owner)." >&2
    fi
fi

ABS_WT="$TOP/$WT"
echo "worktree: $ABS_WT (branch $BRANCH, depset $SET_NAME)"
echo "run tests with:"
echo "  cd $ABS_WT && PYTHONPATH=\$PWD/src nice -n 10 ionice -c3 $VENV/bin/python -m pytest -q -n 2 -p no:cacheprovider <paths>"
echo "then check: PYTHONPATH=\$PWD/src $VENV/bin/python -c 'import firm; print(firm.__file__)'  (must be under $ABS_WT/src)"

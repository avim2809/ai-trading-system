#!/bin/bash
# Overnight launch: Step 1 source run (C0 + signal books) and the corrected-engine
# replication of the 9/28 design (C1, C2, C3, P). Waits for the US close.
set -u
SP=/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad
W=$SP/wt
OUT=$SP/runs/step1
mkdir -p $OUT
target=$(date -d "today 23:05" +%s); now=$(date +%s)
if [ $now -lt $target ]; then sleep $((target - now)); fi
cd /local/store/git/ai-trading-system
{
  date -u +%FT%TZ
  echo "worktree HEAD $(git -C $W rev-parse HEAD)"; git -C $W status --short
  echo "standalone fp $(python $W/scripts/standalone_strategy_preregistered_bars.py)"
  echo "combination fp $(python $W/scripts/combination_preregistered_bars.py)"
  sha256sum $W/scripts/standalone_strategy_preregistered_bars.py $W/scripts/run_standalone_strategy_evaluation.py \
    $W/scripts/combination_preregistered_bars.py $W/scripts/run_combination_evaluation.py \
    $W/src/firm/backtest/datafeeds.py $W/src/firm/backtest/run.py
} > $OUT/launch_provenance.txt 2>&1
for c in C0 C1_robust_attribution C2_robust_standalone C3_confidence P_random_weights; do mkdir -p $SP/datadir_$c; done
FIRM_DATA_DIR=$SP/datadir_C0 PYTHONPATH=$W/src nice -n 10 python $W/scripts/run_standalone_strategy_evaluation.py run --out-dir $OUT > $OUT/C0_legacy_optimal.log 2>&1 &
for c in C1_robust_attribution C2_robust_standalone C3_confidence P_random_weights; do
  FIRM_DATA_DIR=$SP/datadir_$c PYTHONPATH=$W/src nice -n 10 python $W/scripts/run_combination_evaluation.py run --candidate $c --out-dir $OUT > $OUT/$c.log 2>&1 &
done
wait
date -u +%FT%TZ >> $OUT/launch_provenance.txt
ls -la $OUT

# Gann Research Archive

Frozen copy of the Jul 2026 Gann cycles research programme. **Not used in production**
(`main` keeps `gann` disabled; see `docs/gann_research_closeout.md`).

Branch: `research/gann-archive`

## Raw URLs for external audit (paste into agent chat)

### Study scripts

```
https://raw.githubusercontent.com/avim2809/ai-trading-system/research/gann-archive/scripts/gann_multiasset_study.py
https://raw.githubusercontent.com/avim2809/ai-trading-system/research/gann-archive/scripts/gann_correct_cycles_study.py
https://raw.githubusercontent.com/avim2809/ai-trading-system/research/gann-archive/scripts/gann_ic_study.py
https://raw.githubusercontent.com/avim2809/ai-trading-system/research/gann-archive/scripts/gann_followup_study.py
https://raw.githubusercontent.com/avim2809/ai-trading-system/research/gann-archive/scripts/gann_swing_event_study.py
```

### Results (archived JSON)

```
https://raw.githubusercontent.com/avim2809/ai-trading-system/research/gann-archive/research/gann-archive/results/gann_multiasset/results.json
https://raw.githubusercontent.com/avim2809/ai-trading-system/research/gann-archive/research/gann-archive/results/gann_correct_cycles/results.json
```

### Strategy + closeout (on `main`)

```
https://raw.githubusercontent.com/avim2809/ai-trading-system/main/src/firm/strategies/gann.py
https://raw.githubusercontent.com/avim2809/ai-trading-system/main/docs/gann_research_closeout.md
```

## Run (optional)

```bash
git checkout research/gann-archive
pip install yfinance  # multi-asset study only
python scripts/gann_multiasset_study.py
python scripts/gann_correct_cycles_study.py
```

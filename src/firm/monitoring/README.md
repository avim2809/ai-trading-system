# firm.monitoring

- `allocation_forward.py` (P5-06): allocation forward test, frozen I1-I5 bars. Never merged with the G-PAPER thresholds.
- `decay.py` (P5-03): CUSUM and rolling t-test.
- `fidelity.py` (P5-04): candidate fidelity monitor. Daily-return corr / annualised tracking error vs a shadow replay
  (`firm.backtest.vector_engine` on the candidate's own snapshot), implementation shortfall vs `firm.costs.model`
  (commission and exchange fees included, arrival mid as reference), durable position breaks, missed reviews.
  Thresholds come from `config/gates.yaml` `g_paper`. corr/TE count towards a breach only once the full 63-day window exists.
- `shadow_loader.py` (P5-04): monitor-only snapshot reader (not `data_access`; not importable from research/live code).
- CLI: `scripts/daily_reconcile.py {run,report,export}`. Daily state lives in the gitignored monitor state dir; `export`
  (owner-run) is the only writer of the sealed candidate directory. No timer is installed (OD-15, per ticket at P6).

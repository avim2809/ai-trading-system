// Shared formatting for firm.eval.metrics-style per-strategy metric dicts
// (backtest report.strategies, live GET /live/attribution) — used by
// Compare.tsx (run-vs-run) and StrategyAttributionTable.tsx (sleeve-vs-sleeve).

export function formatMetric(key: string, value: number): string {
  if (value == null || isNaN(value)) return '—'
  const pctKeys = ['total_return', 'cagr', 'max_drawdown', 'volatility', 'turnover']
  if (pctKeys.some((k) => key.toLowerCase().includes(k))) {
    return `${(value * 100).toFixed(2)}%`
  }
  return value.toFixed(4)
}

// Metrics where a *smaller* value is the better outcome (all reported as
// positive magnitudes by firm.eval.metrics — see max_drawdown/annualized_
// volatility/conditional_value_at_risk) — everything else defaults to
// higher-is-better (Sharpe, CAGR, alpha, hit rate, ...).
const LOWER_IS_BETTER = ['max_drawdown', 'annualized_volatility', 'volatility', 'cvar', 'turnover']

export function isLowerBetter(key: string): boolean {
  const k = key.toLowerCase()
  return LOWER_IS_BETTER.some((m) => k.includes(m))
}

// Sign-based coloring, matching MetricCard's convention for pct/ratio
// values (emerald when positive, red when negative) — used for metrics
// that can go either way (total_return, sharpe_ratio, cagr). max_drawdown
// is always reported as a positive magnitude (see firm.eval.metrics), so
// callers render it with a fixed loss-red instead of this helper.
export function signColor(value: number | undefined): string {
  if (value == null || isNaN(value)) return 'text-slate-500'
  if (value > 0) return 'text-emerald-400'
  if (value < 0) return 'text-red-400'
  return 'text-slate-300'
}

// Below this many daily NAV/return observations, annualized ratios (Sharpe,
// CAGR) are an extreme extrapolation from a handful of points, not a
// meaningful estimate — shown as "n/a" rather than a wild, misleading
// number. Shared between LiveDashboard's own portfolio-level gate and
// StrategyAttributionTable's per-sleeve gate (sleeved mode's exact metrics
// carry an `n_days` sample size alongside the ratios).
export const MIN_OBSERVATIONS_FOR_RATIOS = 10
